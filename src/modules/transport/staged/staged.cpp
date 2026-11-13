/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Staged RDMA transport for NVSHMEM.
 * Uses ibverbs RC QPs with staged bounce buffers.
 * Data path: GPU --(copy)--> local bounce --(RDMA Write)--> remote bounce --(copy)--> GPU
 * Bounce buffers live in pinned host memory (SYSMEM, default) or in GPU memory registered for
 * GPUDirect RDMA (VIDMEM), selected by NVSHMEM_STAGED_BOUNCE_LOCATION.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <infiniband/verbs.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "non_abi/device/transport/nvshmem_common_transport.h"
#include "non_abi/device/transport/transport_constants.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"
#include "non_abi/c/nvshmemi_error_macros.h"
#include "staged_config.h"
#include "staged_gdrcopy.h"
#include "transport_common.h"
#include "transport_ib_common.h"

namespace nvshmemi {

static struct nvshmemt_ibv_function_table ftable;
static void* ibv_handle;

staged_qp_t::staged_qp_t(struct ibv_context* context, struct ibv_pd* pd, void* server_bounce)
    : context_(context), pd_(pd), server_bounce_(server_bounce) {}

staged_qp_t::~staged_qp_t() { close(); }

int staged_qp_t::close() noexcept {
    int status = 0;
    if (qp_) {
        int verbs_status = ftable.destroy_qp(qp_);
        if (verbs_status != 0) {
            /* A live QP may still reference its CQs and control receive MR. */
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_destroy_qp failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        qp_ = nullptr;
    }
    if (send_cq_) {
        int verbs_status = ftable.destroy_cq(send_cq_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ftable.destroy_cq(send) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            send_cq_ = nullptr;
        }
    }
    if (recv_cq_) {
        int verbs_status = ftable.destroy_cq(recv_cq_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ftable.destroy_cq(recv) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            recv_cq_ = nullptr;
        }
    }
    if (ctrl_recv_mr_) {
        int verbs_status = ftable.dereg_mr(ctrl_recv_mr_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ftable.dereg_mr(control) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            ctrl_recv_mr_ = nullptr;
        }
    }
    return status;
}

staged_rdma_state_t::~staged_rdma_state_t() { close(); }

int staged_rdma_state_t::close() noexcept {
    int status = 0;
    for (const auto& qp : qps_.qps) {
        int qp_status = qp->close();
        if (status == 0 && qp_status != 0) {
            status = qp_status;
        }
    }
    if (status != 0) {
        /* A surviving QP may still reference the bounce MR, PD, and device context. */
        return status;
    }
    qps_.qps.clear();

    if (ib_.bounce_handle.mr) {
        if (nvshmemt_ib_common_release_mem_handle(
                &ftable, reinterpret_cast<nvshmem_mem_handle_t*>(&ib_.bounce_handle), 0) != 0) {
            /* A registered MR still references its allocation and PD. */
            NVSHMEMI_ERROR_PRINT("[STAGED] bounce deregistration failed during cleanup");
            return NVSHMEMX_ERROR_INTERNAL;
        }
        ib_.bounce_handle = {};
    }
    if (ib_.bounce_region) {
        const bool vidmem = ib_.bounce_in_vidmem;
        cudaError_t cuda_status =
            vidmem ? cudaFree(ib_.bounce_region) : cudaFreeHost(ib_.bounce_region);
        if (cuda_status != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] %s(bounce) failed during cleanup: %s",
                                 vidmem ? "cudaFree" : "cudaFreeHost",
                                 cudaGetErrorString(cuda_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            ib_.bounce_region = nullptr;
            ib_.client_bounce = nullptr;
            ib_.bounce_region_size = 0;
        }
    }
    if (ib_.protection_domain) {
        int verbs_status = ftable.dealloc_pd(ib_.protection_domain);
        if (verbs_status != 0) {
            /* The device context must outlive a PD that could not be released. */
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_dealloc_pd failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return status == 0 ? NVSHMEMX_ERROR_INTERNAL : status;
        }
        ib_.protection_domain = nullptr;
    }
    if (ib_.context) {
        int verbs_status = ftable.close_device(ib_.context);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_close_device failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return status == 0 ? NVSHMEMX_ERROR_INTERNAL : status;
        }
        ib_.context = nullptr;
    }
    return status;
}

staged_cuda_state_t::~staged_cuda_state_t() {
    for (cudaEvent_t event : client_events) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    if (client_stream) {
        cuStreamDestroy(client_stream);
    }
    if (server_stream) {
        cuStreamDestroy(server_stream);
    }
    if (green_ctx) {
        cuGreenCtxDestroy(green_ctx);
    }
}

int staged_qp_t::initialize(int ib_port, uint16_t pkey_index, size_t index) {
    ctrl_recv_mr_ =
        ftable.reg_mr(pd_, ctrl_recv_bufs_.data(),
                      ctrl_recv_bufs_.size() * sizeof(staged_ctrl_msg_t), IBV_ACCESS_LOCAL_WRITE);
    if (!ctrl_recv_mr_) {
        NVSHMEMI_ERROR_PRINT("[STAGED] failed to register control receive buffers for QP %zu: %s",
                             index, strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    send_cq_ = ftable.create_cq(context_, STAGED_CQ_DEPTH, nullptr, nullptr, 0);
    recv_cq_ = ftable.create_cq(context_, STAGED_CQ_DEPTH, nullptr, nullptr, 0);
    if (!send_cq_ || !recv_cq_) {
        NVSHMEMI_ERROR_PRINT("[STAGED] failed to create CQs for QP %zu", index);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    struct ibv_qp_init_attr qp_attr{};
    qp_attr.send_cq = send_cq_;
    qp_attr.recv_cq = recv_cq_;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = STAGED_SQ_DEPTH;
    qp_attr.cap.max_recv_wr = STAGED_RQ_DEPTH;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.cap.max_inline_data = 128;
    qp_ = ftable.create_qp(pd_, &qp_attr);
    if (!qp_) {
        NVSHMEMI_ERROR_PRINT("[STAGED] failed to create QP %zu: %s", index, strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    struct ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = static_cast<uint8_t>(ib_port);
    attr.pkey_index = pkey_index;
    attr.qp_access_flags =
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    int status = ftable.modify_qp(
        qp_, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (status) {
        NVSHMEMI_ERROR_PRINT("[STAGED] QP %zu INIT failed: %s", index, strerror(status));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

int staged_qp_t::post_receive(size_t index) {
    staged_ctrl_msg_t& message = ctrl_recv_bufs_.at(index);
    struct ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(&message);
    sge.length = sizeof(message);
    sge.lkey = ctrl_recv_mr_->lkey;
    struct ibv_recv_wr wr{}, *bad_wr = nullptr;
    wr.wr_id = index;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    return ibv_post_recv(qp_, &wr, &bad_wr);
}

staged_worker_state_t::staged_worker_state_t(staged_operation_state_t& operations,
                                             staged_response_state_t& responses)
    : operations_(operations), responses_(responses) {}

void staged_worker_state_t::fail(int status) {
    if (status == 0) {
        return;
    }
    int expected = 0;
    first_error_.compare_exchange_strong(expected, status, std::memory_order_relaxed);
    status = first_error_.load(std::memory_order_relaxed);
    stop_requested_.store(true, std::memory_order_relaxed);
    operations_.fail(status);
    responses_.fail(status);
}

void staged_worker_state_t::request_stop() {
    stop_requested_.store(true, std::memory_order_relaxed);
    operations_.stop();
    responses_.stop();
}

staged_worker_thread_t::~staged_worker_thread_t() {
    state_.request_stop();
    join();
}

void staged_worker_thread_t::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

staged_worker_threads_t::~staged_worker_threads_t() {
    state_.request_stop();
    client_thread_.join();
    server_thread_.join();
}

transport_staged_state_t::transport_staged_state_t()
    : operations(NVSHMEMX_ERROR_INTERNAL),
      responses(NVSHMEMX_ERROR_INTERNAL),
      workers(operations, responses) {}

namespace {

/* ── helpers ───────────────────────────────────────────────────── */

static size_t staged_total_qps(const transport_staged_state_t* s) { return s->rdma.total_qps(); }

static staged_qp_t& staged_get_qp(transport_staged_state_t* s, int qp_slot, int pe) {
    return s->rdma.qp(qp_slot, pe);
}

/* Staged uses a single NIC port: the first one the shared IB enumeration accepts, which honors
 * NVSHMEM_HCA_LIST and picks a usable GID. Devices it opened but staged does not use are released.
 */
static int staged_open_hca(transport_staged_state_t* s) {
    int num_devices = 0;
    std::unique_ptr<struct ibv_device*, decltype(ftable.free_device_list)> dev_list(
        ftable.get_device_list(&num_devices), ftable.free_device_list);
    if (!dev_list || num_devices == 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED] no IB devices found");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    std::vector<nvshmemt_ib_common_device> devices(static_cast<size_t>(num_devices));
    std::array<int, MAX_NUM_PES_PER_NODE> dev_ids{};
    std::array<int, MAX_NUM_PES_PER_NODE> port_ids{};
    nvshmemt_ib_common_state ib{};
    ib.devices = devices.data();
    ib.dev_ids = dev_ids.data();
    ib.port_ids = port_ids.data();
    ib.n_raw_devices = num_devices;
    ib.options = &s->options;
    ib.log_level = s->log_level;

    nvshmemt_ib_hca_filter filter{};
    int status = nvshmemt_ib_common_parse_hca_filter(filter, ib);
    if (status == 0) {
        status = nvshmemt_ib_common_enumerate_devices(&ftable, ib, sizeof(devices[0]), filter,
                                                      dev_list.get(), num_devices);
    }
    if (status != 0) {
        return status;
    }
    nvshmemt_ib_common_warn_missing_hcas(filter);

    const int dev_id = ib.dev_ids[0];
    const int port = ib.port_ids[0];
    for (int i = 0; i < num_devices; ++i) {
        if (i == dev_id) {
            continue;
        }
        if (devices[i].pd) {
            ftable.dealloc_pd(devices[i].pd);
        }
        if (devices[i].context) {
            ftable.close_device(devices[i].context);
        }
    }

    const nvshmemt_ib_common_device& device = devices[dev_id];
    const char* name = ftable.get_device_name(device.dev);
    if (filter.user_selection && !filter.exclude_list && ib.n_dev_ids > 1) {
        NVSHMEMI_WARN_PRINT("[STAGED] NVSHMEM_HCA_LIST matches %d ports; staged uses only %s:%d",
                            ib.n_dev_ids, name, port);
    }
    INFO(s->log_level, "[STAGED] using device '%s' port %d", name, port);

    s->rdma.set_context(device.context);
    s->rdma.set_protection_domain(device.pd);
    s->rdma.set_ib_port(port);
    s->rdma.set_local_gid_index(static_cast<uint8_t>(device.gid_info[port - 1].local_gid_index));
    s->rdma.set_local_gid(device.gid_info[port - 1].local_gid);
    return 0;
}

static int staged_init_green_context(transport_staged_state_t* s) {
    if (s->cuda.green_ctx) {
        return 0;
    }

    CUcontext current_ctx = nullptr;
    CUresult cuerr = cuCtxGetCurrent(&current_ctx);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetCurrent failed (CUresult=%d)", static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    if (!current_ctx) {
        NVSHMEMI_ERROR_PRINT("no current CUDA context for green context init");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    CUdevice cu_dev;
    cuerr = cuCtxGetDevice(&cu_dev);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetDevice failed (CUresult=%d)", static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    CUdevResource ctx_sm_resource{};
    cuerr = cuCtxGetDevResource(current_ctx, &ctx_sm_resource, CU_DEV_RESOURCE_TYPE_SM);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetDevResource(SM) failed (CUresult=%d)",
                             static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    unsigned int min_count = static_cast<unsigned int>(s->options.STAGED_GREEN_CTX_SMS);

    CUdevResource green_resource{};
    unsigned int nb_groups = 1;
    cuerr = cuDevSmResourceSplitByCount(&green_resource, &nb_groups, &ctx_sm_resource, nullptr, 0,
                                        min_count);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuDevSmResourceSplitByCount failed (CUresult=%d)",
                             static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    if (nb_groups == 0 || green_resource.type != CU_DEV_RESOURCE_TYPE_SM) {
        NVSHMEMI_ERROR_PRINT("[STAGED] CUDA green context SM split produced no usable partition");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    CUdevResourceDesc green_desc = nullptr;
    cuerr = cuDevResourceGenerateDesc(&green_desc, &green_resource, 1);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuDevResourceGenerateDesc failed (CUresult=%d)",
                             static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    cuerr = cuGreenCtxCreate(&s->cuda.green_ctx, green_desc, cu_dev, CU_GREEN_CTX_DEFAULT_STREAM);
    if (cuerr != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuGreenCtxCreate failed (CUresult=%d)", static_cast<int>(cuerr));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    INFO(s->log_level, "[STAGED] green_ctx sm_count=%u min_sms=%u ctx_sms=%u",
         green_resource.sm.smCount, min_count, ctx_sm_resource.sm.smCount);
    return 0;
}

static int staged_cuda_copy_sync(transport_staged_state_t* /* s */, void* dst, const void* src,
                                 size_t bytes, cudaStream_t stream, const char* label) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s cudaMemcpyAsync failed: %s", label,
                             cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s cudaStreamSynchronize failed: %s", label,
                             cudaGetErrorString(err));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

/* Seeing CHUNK_DONE or RESPONSE does not make the NIC's earlier writes to a VIDMEM bounce visible.
 * Unless the GPU orders GPUDirect RDMA writes, staged_copy() flushes them before reading the
 * bounce. */
static bool staged_is_gdr_flush_required(int device) {
    int ordering = 0;
    return cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device) !=
               cudaSuccess ||
           ordering < cudaGPUDirectRDMAWritesOrderingOwner;
}

static bool staged_is_gdr_flush_supported(int device) {
    int options = 0;
    return cudaDeviceGetAttribute(&options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device) ==
               cudaSuccess &&
           (options & cudaFlushGPUDirectRDMAWritesOptionHost);
}

static int staged_allocate_bounce_region(transport_staged_state_t* s, size_t num_qps) {
    auto& rdma = s->rdma;
    if (num_qps == 0 || rdma.bounce_region() || rdma.bounce_mr()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (s->bounce_bytes > SIZE_MAX / static_cast<size_t>(s->pipeline_depth)) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    const size_t ring_bytes = s->bounce_bytes * static_cast<size_t>(s->pipeline_depth);
    const size_t ring_count = num_qps + 1;
    if (ring_bytes == 0 || ring_count > SIZE_MAX / ring_bytes) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    size_t bounce_region_size = ring_count * ring_bytes;
    const bool vidmem = rdma.bounce_in_vidmem();
    if (vidmem) {
        s->cuda.flush_gdr_writes = staged_is_gdr_flush_required(s->cuda.device);
        if (s->cuda.flush_gdr_writes && !staged_is_gdr_flush_supported(s->cuda.device)) {
            NVSHMEMI_ERROR_PRINT(
                "[STAGED] NVSHMEM_STAGED_BOUNCE_LOCATION=VIDMEM requires GPUDirect RDMA write "
                "ordering or flush support on device %d",
                s->cuda.device);
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        INFO(s->log_level, "[STAGED] VIDMEM bounce: GPUDirect RDMA write flush %s",
             s->cuda.flush_gdr_writes ? "required" : "not required");
        if (bounce_region_size > SIZE_MAX - STAGED_VIDMEM_ALIGNMENT) {
            return NVSHMEMX_ERROR_OUT_OF_MEMORY;
        }
        bounce_region_size = ROUNDUP(bounce_region_size, STAGED_VIDMEM_ALIGNMENT);
    }

    void* bounce_region = nullptr;
    cudaError_t cerr = vidmem ? cudaMalloc(&bounce_region, bounce_region_size)
                              : cudaHostAlloc(&bounce_region, bounce_region_size, 0);
    if (cerr != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s failed for %zu bytes: %s",
                             vidmem ? "cudaMalloc" : "cudaHostAlloc", bounce_region_size,
                             cudaGetErrorString(cerr));
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    /* GPU memory registers through DMA-BUF when available, else through peer memory. */
    bool dmabuf_support = false;
    int status = 0;
    if (vidmem) {
        status = nvshmemt_ib_common_check_dmabuf_support(dmabuf_support, s->cuda.syms,
                                                         s->options.IB_DISABLE_DMABUF);
    }
    nvshmemt_ib_common_mem_handle handle{};
    if (status == 0) {
        status = nvshmemt_ib_common_reg_mem_handle(
            &ftable, nullptr, rdma.protection_domain(),
            reinterpret_cast<nvshmem_mem_handle_t*>(&handle), bounce_region, bounce_region_size,
            false, dmabuf_support, s->cuda.syms, s->log_level, false, false, nullptr);
    }
    if (status == 0 && vidmem && s->cuda.copy_policy == staged_copy_policy_t::GDRCOPY) {
        status = staged_gdrcopy_map(s, bounce_region, bounce_region_size, bounce_region, nullptr);
        if (status) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, reinterpret_cast<nvshmem_mem_handle_t*>(&handle), s->log_level);
        }
    }
    if (status) {
        NVSHMEMI_ERROR_PRINT("[STAGED] failed to register %zu bytes of bounce storage",
                             bounce_region_size);
        if (vidmem) {
            cudaFree(bounce_region);
        } else {
            cudaFreeHost(bounce_region);
        }
        return status;
    }

    rdma.set_bounce_storage(bounce_region, bounce_region_size, bounce_region, handle);
    return 0;
}

/* Report local endpoint-setup failures before entering the endpoint exchange.  Every rank must
 * participate here: otherwise an allocation failure on one PE can leave the other PEs blocked in
 * the subsequent all-to-all. */
static int staged_collective_setup_status(nvshmem_transport_t transport, int local_status,
                                          const char* phase) {
    if (!transport || !transport->boot_handle || !transport->boot_handle->allgather) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] cannot collectively report %s failure: bootstrap allgather is unavailable",
            phase);
        return local_status ? local_status : NVSHMEMX_ERROR_INTERNAL;
    }

    const int n = transport->boot_handle->pg_size;
    if (n <= 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] invalid bootstrap PE count %d during %s",
                             transport->my_pe, n, phase);
        return local_status ? local_status : NVSHMEMX_ERROR_INTERNAL;
    }

    std::vector<int> peer_statuses(static_cast<size_t>(n));
    int rc = transport->boot_handle->allgather(&local_status, peer_statuses.data(),
                                               sizeof(local_status), transport->boot_handle);
    if (rc != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] bootstrap allgather failed while checking %s",
                             transport->my_pe, phase);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    for (int pe = 0; pe < n; ++pe) {
        if (peer_statuses[pe] == 0) {
            continue;
        }
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] %s failed on PE%d (status %d)", transport->my_pe, phase,
                             pe, peer_statuses[pe]);
        return peer_statuses[pe];
    }

    return 0;
}

static int staged_create_qp_resources(transport_staged_state_t* s, int num_pes, int qps_per_pe) {
    auto& rdma = s->rdma;
    if (num_pes <= 0 || qps_per_pe <= 0 || !rdma.qps_empty()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    rdma.configure_qps(num_pes, qps_per_pe);
    const size_t num_qps = staged_total_qps(s);
    int status = staged_allocate_bounce_region(s, num_qps);
    if (status) {
        return status;
    }

    INFO(s->log_level, "[STAGED] allocating %zu QPs (%d per PE) and %zu bytes of %s bounce storage",
         num_qps, rdma.qps_per_pe(), rdma.bounce_region_size(),
         rdma.bounce_in_vidmem() ? "GPU" : "pinned host");

    if (s->cuda.copy_policy == staged_copy_policy_t::STREAM) {
        CUresult cuerr = cuGreenCtxStreamCreate(&s->cuda.client_stream, s->cuda.green_ctx,
                                                CU_STREAM_NON_BLOCKING, 0);
        if (cuerr != CUDA_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("cuGreenCtxStreamCreate(client) failed (CUresult=%d)",
                                 static_cast<int>(cuerr));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        cuerr = cuGreenCtxStreamCreate(&s->cuda.server_stream, s->cuda.green_ctx,
                                       CU_STREAM_NON_BLOCKING, 0);
        if (cuerr != CUDA_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("cuGreenCtxStreamCreate(server) failed (CUresult=%d)",
                                 static_cast<int>(cuerr));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        for (int i = 0; i < s->pipeline_depth; i++) {
            cudaError_t cerr =
                cudaEventCreateWithFlags(&s->cuda.client_events[i], cudaEventDisableTiming);
            if (cerr != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED] failed to create CUDA event for slot %d: %s", i,
                                     cudaGetErrorString(cerr));
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }

    const size_t ring_bytes = s->bounce_bytes * static_cast<size_t>(s->pipeline_depth);
    rdma.reserve_qps(num_qps);
    for (size_t q = 0; q < num_qps; ++q) {
        /* Layout: [shared client slot ring | one server slot ring per QP]. */
        void* server_bounce = static_cast<char*>(rdma.bounce_region()) + (q + 1) * ring_bytes;
        auto qp =
            std::make_unique<staged_qp_t>(rdma.context(), rdma.protection_domain(), server_bounce);
        status = qp->initialize(rdma.ib_port(), static_cast<uint16_t>(s->options.IB_PKEY_INDEX), q);
        if (status) {
            return status;
        }
        rdma.add_qp(std::move(qp));
    }

    return 0;
}

static void staged_fail_control(transport_staged_state_t* s) {
    s->workers.fail(NVSHMEMX_ERROR_INTERNAL);
}

static void staged_stop_workers(transport_staged_state_t* s) {
    if (!s) {
        return;
    }
    s->workers.request_stop();
    s->worker_threads.reset();
}

class staged_worker_cuda_scope_t {
   public:
    staged_worker_cuda_scope_t(const staged_cuda_state_t& cuda, const char* worker)
        : worker_(worker) {
        cudaError_t error = cudaGetDevice(&previous_device_);
        if (error != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED %s] cudaGetDevice failed: %s", worker_,
                                 cudaGetErrorString(error));
            return;
        }

        error = cudaSetDevice(cuda.device);
        if (error != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED %s] cudaSetDevice(%d) failed: %s", worker_, cuda.device,
                                 cudaGetErrorString(error));
            return;
        }
        device_set_ = true;

        if (cuda.copy_policy == staged_copy_policy_t::STREAM) {
            previous_capture_mode_ = cudaStreamCaptureModeRelaxed;
            error = cudaThreadExchangeStreamCaptureMode(&previous_capture_mode_);
            if (error != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED %s] cudaThreadExchangeStreamCaptureMode failed: %s",
                                     worker_, cudaGetErrorString(error));
                return;
            }
            capture_mode_set_ = true;
        }
        status_ = 0;
    }

    ~staged_worker_cuda_scope_t() {
        if (capture_mode_set_) {
            cudaError_t error = cudaThreadExchangeStreamCaptureMode(&previous_capture_mode_);
            if (error != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED %s] failed to restore CUDA stream capture mode: %s",
                                     worker_, cudaGetErrorString(error));
            }
        }
        if (device_set_) {
            cudaError_t error = cudaSetDevice(previous_device_);
            if (error != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED %s] failed to restore CUDA device %d: %s", worker_,
                                     previous_device_, cudaGetErrorString(error));
            }
        }
    }

    staged_worker_cuda_scope_t(const staged_worker_cuda_scope_t&) = delete;
    staged_worker_cuda_scope_t& operator=(const staged_worker_cuda_scope_t&) = delete;
    staged_worker_cuda_scope_t(staged_worker_cuda_scope_t&&) = delete;
    staged_worker_cuda_scope_t& operator=(staged_worker_cuda_scope_t&&) = delete;

    int status() const { return status_; }

   private:
    const char* worker_;
    int previous_device_ = -1;
    cudaStreamCaptureMode previous_capture_mode_ = cudaStreamCaptureModeGlobal;
    int status_ = NVSHMEMX_ERROR_INTERNAL;
    bool device_set_ = false;
    bool capture_mode_set_ = false;
};

/* Poll CQ until at least one completion or a transport failure stops the workers. */
static int poll_cq_blocking(transport_staged_state_t* s, struct ibv_cq* cq, struct ibv_wc* wc) {
    int n = 0;
    while (!s->workers.stopping() && (n = ibv_poll_cq(cq, 1, wc)) == 0) {
    }
    return s->workers.stopping() ? -1 : n;
}

static int send_ctrl(transport_staged_state_t* s, staged_qp_t& qp, const staged_ctrl_msg_t* msg) {
    return qp.serialize_send([&](struct ibv_qp* qp_handle, struct ibv_cq* send_cq) -> int {
        if (s->workers.stopping()) {
            return NVSHMEMX_ERROR_INTERNAL;
        }

        struct ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(msg);
        sge.length = sizeof(staged_ctrl_msg_t);
        sge.lkey = 0;
        struct ibv_send_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        int ret = ibv_post_send(qp_handle, &wr, &bad_wr);
        if (ret != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_post_send for control message failed: %s",
                                 strerror(ret));
            staged_fail_control(s);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        /* Poll for send completion. */
        struct ibv_wc wc;
        int n = poll_cq_blocking(s, send_cq, &wc);
        if (n < 0) {
            if (!s->workers.stopping()) {
                NVSHMEMI_ERROR_PRINT("[STAGED] ibv_poll_cq failed while sending a control message");
            }
            staged_fail_control(s);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("[STAGED] control message send failed: %s",
                                 ftable.wc_status_str(wc.status));
            staged_fail_control(s);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return 0;
    });
}

static int post_rdma_write_wait(transport_staged_state_t* s, staged_qp_t& qp, void* local_buf,
                                size_t len, uint64_t remote_addr, uint32_t rkey) {
    return qp.serialize_send([&](struct ibv_qp* qp_handle, struct ibv_cq* send_cq) -> int {
        if (s->workers.stopping()) {
            return NVSHMEMX_ERROR_INTERNAL;
        }
        struct ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(local_buf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey = s->rdma.bounce_mr()->lkey;
        struct ibv_send_wr wr{}, *bad_wr = nullptr;
        wr.wr_id = len;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;
        int ret = ibv_post_send(qp_handle, &wr, &bad_wr);
        if (ret != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_post_send for RDMA Write failed: %s", strerror(ret));
            return NVSHMEMX_ERROR_INTERNAL;
        }

        struct ibv_wc wc;
        if (poll_cq_blocking(s, send_cq, &wc) < 0) {
            if (!s->workers.stopping()) {
                NVSHMEMI_ERROR_PRINT("[STAGED] ibv_poll_cq failed while waiting for RDMA Write");
            }
            return NVSHMEMX_ERROR_INTERNAL;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("[STAGED] RDMA Write failed: %s", ftable.wc_status_str(wc.status));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return 0;
    });
}

static size_t staged_slot_offset(transport_staged_state_t* s, int slot) {
    return static_cast<size_t>(slot) * s->bounce_bytes;
}

static bool staged_is_gpu_memory(const void* ptr) {
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        /* Host pointers succeed (cudaMemoryTypeUnregistered), so this is a CUDA error. Treat the
         * pointer as GPU memory: the GPU paths report errors, where a host memcpy could crash. */
        NVSHMEMI_ERROR_PRINT("[STAGED] cudaPointerGetAttributes(%p) failed: %s", ptr,
                             cudaGetErrorString(err));
        (void)cudaGetLastError();
        return true;
    }
    return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
}

/* Copy bytes from src to dst and wait until dst holds them; either side may be host or GPU memory.
 * Reading the VIDMEM bounce means the NIC wrote it, so flush those writes first when needed. */
static int staged_copy(nvshmem_transport_t transport, void* dst, const void* src, size_t bytes,
                       cudaStream_t stream, const char* label) {
    auto* s = static_cast<transport_staged_state_t*>(transport->state);
    if (bytes == 0) {
        return 0;
    }

    if (s->cuda.flush_gdr_writes && s->rdma.bounce_contains(src)) {
        cudaError_t err = cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner);
        if (err != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] %s GPUDirect RDMA write flush failed: %s", label,
                                 cudaGetErrorString(err));
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }

    const bool dst_on_gpu = staged_is_gpu_memory(dst);
    const bool src_on_gpu = staged_is_gpu_memory(src);
    if (!dst_on_gpu && !src_on_gpu) {
        memcpy(dst, src, bytes);
        return 0;
    }
    if (s->cuda.copy_policy == staged_copy_policy_t::STREAM) {
        return staged_cuda_copy_sync(s, dst, src, bytes, stream, label);
    }
    return staged_gdrcopy_copy(s, dst, dst_on_gpu, src, src_on_gpu, bytes, label);
}

static const char* staged_rma_name(const rma_verb_t& verb) {
    switch (verb.desc) {
        case NVSHMEMI_OP_P:
            return "p";
        case NVSHMEMI_OP_PUT:
            return verb.is_nbi ? "put_nbi" : "put";
        case NVSHMEMI_OP_G:
            return "g";
        case NVSHMEMI_OP_GET:
            return verb.is_nbi ? "get_nbi" : "get";
        default:
            return "unknown";
    }
}

static int staged_complete_response(transport_staged_state_t* s, const staged_ctrl_msg_t& msg) {
    staged_response_t response{};
    response.value = msg.value;
    response.status = static_cast<int>(msg.status);
    return s->responses.complete(msg.request_id, response);
}

static int staged_wait_response(transport_staged_state_t* s,
                                staged_response_state_t::ticket response_ticket,
                                staged_response_t* response) {
    staged_response_t received{};
    int status = s->responses.wait(response_ticket, &received);
    if (status != 0) {
        return status;
    }
    if (response) {
        *response = received;
    }
    return received.status;
}

template <typename T>
static int staged_compute_amo_value(T old_value, const staged_ctrl_msg_t& msg, T* new_value_out) {
    T new_value = old_value;
    uint32_t op_flags = msg.flags & STAGED_CTRL_AMO_OP_MASK;
    bool is_float = (op_flags & NVSHMEMI_AMO_FLOAT_BIT) != 0;
    nvshmemi_amo_t op =
        static_cast<nvshmemi_amo_t>(op_flags & ~static_cast<uint32_t>(NVSHMEMI_AMO_FLOAT_BIT));
    switch (op) {
        case NVSHMEMI_AMO_INC:
        case NVSHMEMI_AMO_FETCH_INC:
            new_value = old_value + static_cast<T>(1);
            break;
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP:
            new_value = static_cast<T>(msg.value);
            break;
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_FETCH_ADD:
            new_value = is_float ? nvshmemt_float_atomic_add<T>(old_value, msg.value)
                                 : old_value + static_cast<T>(msg.value);
            break;
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND:
            new_value = old_value & static_cast<T>(msg.value);
            break;
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR:
            new_value = old_value | static_cast<T>(msg.value);
            break;
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR:
            new_value = old_value ^ static_cast<T>(msg.value);
            break;
        case NVSHMEMI_AMO_COMPARE_SWAP:
            new_value =
                (old_value == static_cast<T>(msg.cmp)) ? static_cast<T>(msg.value) : old_value;
            break;
        case NVSHMEMI_AMO_FETCH:
            new_value = old_value;
            break;
        default:
            NVSHMEMI_ERROR_PRINT("[STAGED] AMO verb %u not implemented yet",
                                 msg.flags & STAGED_CTRL_AMO_OP_MASK);
            return NVSHMEMX_ERROR_INTERNAL;
    }

    *new_value_out = new_value;
    return 0;
}

template <typename T>
static int staged_apply_amo_t(nvshmem_transport_t transport, void* target,
                              const staged_ctrl_msg_t& msg, uint64_t* old_value_out,
                              cudaStream_t stream) {
    static_assert(sizeof(T) <= 8, "staged AMO only supports up to 8-byte elements");

    T old_value{};
    T new_value{};
    int status =
        staged_copy(transport, &old_value, target, sizeof(old_value), stream, "AMO load target");
    if (status) {
        return status;
    }

    status = staged_compute_amo_value(old_value, msg, &new_value);
    if (status) {
        return status;
    }

    status =
        staged_copy(transport, target, &new_value, sizeof(new_value), stream, "AMO store target");
    if (status) {
        return status;
    }

    *old_value_out = static_cast<uint64_t>(old_value);
    return 0;
}

static int staged_apply_amo(nvshmem_transport_t transport, void* target,
                            const staged_ctrl_msg_t& msg, uint64_t* old_value,
                            cudaStream_t stream) {
    if (msg.bytes == sizeof(uint16_t)) {
        return staged_apply_amo_t<uint16_t>(transport, target, msg, old_value, stream);
    } else if (msg.bytes == sizeof(uint32_t)) {
        return staged_apply_amo_t<uint32_t>(transport, target, msg, old_value, stream);
    } else if (msg.bytes == sizeof(uint64_t)) {
        return staged_apply_amo_t<uint64_t>(transport, target, msg, old_value, stream);
    }

    NVSHMEMI_ERROR_PRINT("[STAGED] AMO size %lu not implemented yet",
                         static_cast<unsigned long>(msg.bytes));
    return NVSHMEMX_ERROR_INTERNAL;
}

/* ── server thread: handles incoming control messages ──────────── */
static void staged_server_loop(nvshmem_transport_t transport, staged_startup_latch_t& startup) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    staged_worker_cuda_scope_t cuda_scope(s->cuda, "server");
    int status = cuda_scope.status();
    if (status) {
        startup.report(status);
        staged_fail_control(s);
        return;
    }

    const size_t num_qps = staged_total_qps(s);
    for (size_t q = 0; q < num_qps; ++q) {
        const int pe = static_cast<int>(q % static_cast<size_t>(s->rdma.num_pes()));
        if (pe == transport->my_pe) {
            continue;
        }
        staged_qp_t& qp = s->rdma.qp_at(q);
        for (int i = 0; i < STAGED_RQ_DEPTH; ++i) {
            if (qp.post_receive(static_cast<size_t>(i)) != 0) {
                NVSHMEMI_ERROR_PRINT("[STAGED server] failed to post initial control receive");
                startup.report(NVSHMEMX_ERROR_INTERNAL);
                staged_fail_control(s);
                return;
            }
        }
    }

    startup.report(0);

    while (!s->workers.stopping()) {
        bool made_progress = false;
        for (size_t q = 0; q < num_qps; ++q) {
            const int pe = static_cast<int>(q % static_cast<size_t>(s->rdma.num_pes()));
            if (pe == transport->my_pe) {
                continue;
            }
            staged_qp_t& qp = s->rdma.qp_at(q);

            struct ibv_wc wc;
            int n = ibv_poll_cq(qp.recv_cq(), 1, &wc);
            if (n < 0) {
                NVSHMEMI_ERROR_PRINT(
                    "[STAGED server] ibv_poll_cq failed while receiving control "
                    "messages");
                staged_fail_control(s);
                return;
            }
            if (n == 0) {
                continue;
            }
            made_progress = true;

            if (wc.status != IBV_WC_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("[STAGED server] control receive failed: %s",
                                     ftable.wc_status_str(wc.status));
                staged_fail_control(s);
                return;
            }

            int buf_idx = static_cast<int>(wc.wr_id);
            if (buf_idx < 0 || buf_idx >= STAGED_RQ_DEPTH) {
                NVSHMEMI_ERROR_PRINT("[STAGED server] invalid control receive buffer index %d",
                                     buf_idx);
                staged_fail_control(s);
                return;
            }
            staged_ctrl_msg_t& msg = qp.ctrl_message(static_cast<size_t>(buf_idx));

            if (msg.op == staged_ctrl_op_t::RESPONSE) {
                if (staged_complete_response(s, msg) != 0) {
                    NVSHMEMI_ERROR_PRINT(
                        "[STAGED server] received an unknown or duplicate response id %lu",
                        static_cast<unsigned long>(msg.request_id));
                    staged_fail_control(s);
                    return;
                }
            } else if (msg.op == staged_ctrl_op_t::CHUNK_DONE) {
                /* A chunk was RDMA-written to this peer's bounce buffer. */
                char* dst = reinterpret_cast<char*>(msg.addr);
                size_t slot_offset = static_cast<size_t>(msg.value);
                size_t len = static_cast<size_t>(msg.bytes);
                TRACE(s->log_level,
                      "[STAGED PE%d server] chunk_done req=%lu bytes=%zu slot=%zu dst=%p",
                      transport->my_pe, static_cast<unsigned long>(msg.request_id), len,
                      slot_offset, dst);
                int status = staged_copy(transport, dst,
                                         static_cast<char*>(qp.server_bounce()) + slot_offset, len,
                                         s->cuda.server_stream, "PUT H2D");
                TRACE(s->log_level, "[STAGED PE%d server] h2d_done req=%lu bytes=%zu status=%d",
                      transport->my_pe, static_cast<unsigned long>(msg.request_id), len, status);

                staged_ctrl_msg_t ack{};
                ack.op = staged_ctrl_op_t::RESPONSE;
                ack.request_id = msg.request_id;
                ack.status = status;
                if (send_ctrl(s, qp, &ack)) {
                    return;
                }
            } else if (msg.op == staged_ctrl_op_t::GET_REQ) {
                char* src = reinterpret_cast<char*>(msg.addr);
                size_t len = static_cast<size_t>(msg.bytes);
                bool rdma_write_failed = false;

                int status = staged_copy(transport, qp.server_bounce(), src, len,
                                         s->cuda.server_stream, "GET D2H");
                if (!status) {
                    status = post_rdma_write_wait(s, qp, qp.server_bounce(), len, msg.reply_addr,
                                                  msg.reply_rkey);
                    rdma_write_failed = status != 0;
                }

                staged_ctrl_msg_t resp{};
                resp.op = staged_ctrl_op_t::RESPONSE;
                resp.request_id = msg.request_id;
                resp.status = status;
                if (send_ctrl(s, qp, &resp)) {
                    return;
                }
                if (rdma_write_failed) {
                    staged_fail_control(s);
                    return;
                }
            } else if (msg.op == staged_ctrl_op_t::AMO) {
                void* target = reinterpret_cast<void*>(msg.addr);
                uint64_t old_value = 0;
                int status = 0;
                {
                    std::lock_guard<std::mutex> lk(s->memory.amo_mutex);
                    status =
                        staged_apply_amo(transport, target, msg, &old_value, s->cuda.server_stream);
                }

                staged_ctrl_msg_t resp{};
                resp.op = staged_ctrl_op_t::RESPONSE;
                resp.request_id = msg.request_id;
                resp.value = (msg.flags & STAGED_CTRL_AMO_FETCH) ? old_value : 0;
                resp.status = status;
                if (send_ctrl(s, qp, &resp)) {
                    return;
                }
            } else {
                NVSHMEMI_ERROR_PRINT("[STAGED server] invalid control message opcode %u",
                                     static_cast<unsigned int>(msg.op));
                staged_fail_control(s);
                return;
            }

            /* Re-post recv on the QP that delivered the message. */
            if (qp.post_receive(static_cast<size_t>(buf_idx)) != 0) {
                NVSHMEMI_ERROR_PRINT("[STAGED server] failed to repost control receive");
                staged_fail_control(s);
                return;
            }
        }
        if (!made_progress) {
            std::this_thread::yield();
        }
    }
}

static int staged_execute_self_rma(nvshmem_transport_t transport, const staged_client_op_t& op) {
    auto* s = static_cast<transport_staged_state_t*>(transport->state);
    const auto& rma = std::get<staged_rma_op_t>(op.operation);
    uint64_t total = static_cast<uint64_t>(rma.bytesdesc.elembytes) * rma.bytesdesc.nelems;
    char* dst = nullptr;
    const char* src = nullptr;

    if (rma.verb.desc == NVSHMEMI_OP_P || rma.verb.desc == NVSHMEMI_OP_PUT) {
        dst = static_cast<char*>(rma.remote.ptr);
        src = static_cast<const char*>(rma.local.ptr);
    } else if (rma.verb.desc == NVSHMEMI_OP_G || rma.verb.desc == NVSHMEMI_OP_GET) {
        dst = static_cast<char*>(rma.local.ptr);
        src = static_cast<const char*>(rma.remote.ptr);
    } else {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return staged_copy(transport, dst, src, static_cast<size_t>(total), s->cuda.client_stream,
                       "self RMA copy");
}

static int staged_execute_put(nvshmem_transport_t transport, staged_client_op_t& op) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    const auto& rma = std::get<staged_rma_op_t>(op.operation);
    staged_qp_t& qp = staged_get_qp(s, 0, op.pe);
    const staged_ep_handle_t& rep = qp.remote_endpoint();
    const char* src = static_cast<const char*>(rma.local.ptr);
    uint64_t total = static_cast<uint64_t>(rma.bytesdesc.elembytes) * rma.bytesdesc.nelems;
    uint64_t remote_addr = reinterpret_cast<uint64_t>(rma.remote.ptr);
    size_t slot_bytes = s->bounce_bytes;
    int depth = std::max(1, s->pipeline_depth);
    uint64_t total_chunks = (total + slot_bytes - 1) / slot_bytes;

    /* Copies on the GPU are asynchronous so one chunk's copy overlaps the previous chunk's RDMA. */
    const bool cuda_copy = s->cuda.copy_policy == staged_copy_policy_t::STREAM &&
                           (staged_is_gpu_memory(src) || s->rdma.bounce_in_vidmem());
    std::array<staged_response_state_t::ticket, STAGED_MAX_PIPELINE_DEPTH> response_tickets{};

    auto post_copy = [&](uint64_t chunk_idx) -> int {
        int slot = static_cast<int>(chunk_idx % static_cast<uint64_t>(depth));
        size_t slot_offset = staged_slot_offset(s, slot);
        uint64_t offset = chunk_idx * slot_bytes;
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(slot_bytes, total - offset));
        char* bounce = static_cast<char*>(s->rdma.client_bounce()) + slot_offset;

        TRACE(s->log_level,
              "[STAGED PE%d client] put post_copy pe=%d chunk=%lu slot=%d offset=%lu "
              "bytes=%zu src=%p remote_addr=0x%lx",
              transport->my_pe, op.pe, static_cast<unsigned long>(chunk_idx), slot,
              static_cast<unsigned long>(offset), chunk, src + offset,
              static_cast<unsigned long>(remote_addr + offset));

        if (!cuda_copy) {
            return staged_copy(transport, bounce, src + offset, chunk, s->cuda.client_stream,
                               "PUT D2H");
        }

        cudaError_t cerr =
            cudaMemcpyAsync(bounce, src + offset, chunk, cudaMemcpyDefault, s->cuda.client_stream);
        if (cerr != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] cudaMemcpyAsync D2H failed: %s",
                                 cudaGetErrorString(cerr));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        cerr = cudaEventRecord(s->cuda.client_events[slot], s->cuda.client_stream);
        if (cerr != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] cudaEventRecord failed: %s", cudaGetErrorString(cerr));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return 0;
    };

    auto wait_copy = [&](uint64_t chunk_idx) -> int {
        int slot = static_cast<int>(chunk_idx % static_cast<uint64_t>(depth));
        if (!cuda_copy) {
            return 0;
        }
        cudaError_t cerr = cudaEventSynchronize(s->cuda.client_events[slot]);
        if (cerr != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] cudaEventSynchronize failed: %s",
                                 cudaGetErrorString(cerr));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return 0;
    };

    auto send_chunk = [&](uint64_t chunk_idx) -> int {
        int slot = static_cast<int>(chunk_idx % static_cast<uint64_t>(depth));
        size_t slot_offset = staged_slot_offset(s, slot);
        uint64_t offset = chunk_idx * slot_bytes;
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(slot_bytes, total - offset));

        int status = wait_copy(chunk_idx);
        if (status) {
            return status;
        }
        TRACE(s->log_level,
              "[STAGED PE%d client] d2h_done pe=%d chunk=%lu slot=%d offset=%lu bytes=%zu",
              transport->my_pe, op.pe, static_cast<unsigned long>(chunk_idx), slot,
              static_cast<unsigned long>(offset), chunk);

        status =
            post_rdma_write_wait(s, qp, static_cast<char*>(s->rdma.client_bounce()) + slot_offset,
                                 chunk, rep.bounce_addr + slot_offset, rep.rkey);
        if (status) {
            staged_fail_control(s);
            return status;
        }
        TRACE(s->log_level,
              "[STAGED PE%d client] rdma_done pe=%d chunk=%lu slot=%d offset=%lu bytes=%zu",
              transport->my_pe, op.pe, static_cast<unsigned long>(chunk_idx), slot,
              static_cast<unsigned long>(offset), chunk);

        status = s->responses.reserve(&response_tickets[slot]);
        if (status != 0) {
            return status;
        }

        staged_ctrl_msg_t ctrl{};
        ctrl.op = staged_ctrl_op_t::CHUNK_DONE;
        ctrl.addr = remote_addr + offset;
        ctrl.bytes = chunk;
        ctrl.value = slot_offset;
        ctrl.request_id = response_tickets[slot].request_id;
        status = send_ctrl(s, qp, &ctrl);
        if (status) {
            return status;
        }
        TRACE(s->log_level,
              "[STAGED PE%d client] sent_chunk req=%lu pe=%d chunk=%lu slot=%d bytes=%zu",
              transport->my_pe, static_cast<unsigned long>(response_tickets[slot].request_id),
              op.pe, static_cast<unsigned long>(chunk_idx), slot, chunk);
        return 0;
    };

    auto wait_ack = [&](uint64_t chunk_idx) -> int {
        int slot = static_cast<int>(chunk_idx % static_cast<uint64_t>(depth));
        TRACE(s->log_level, "[STAGED PE%d client] wait_ack req=%lu pe=%d chunk=%lu slot=%d",
              transport->my_pe, static_cast<unsigned long>(response_tickets[slot].request_id),
              op.pe, static_cast<unsigned long>(chunk_idx), slot);

        int status = staged_wait_response(s, response_tickets[slot], nullptr);
        if (status) {
            return status;
        }
        TRACE(s->log_level, "[STAGED PE%d client] ack_done req=%lu pe=%d chunk=%lu slot=%d",
              transport->my_pe, static_cast<unsigned long>(response_tickets[slot].request_id),
              op.pe, static_cast<unsigned long>(chunk_idx), slot);
        return 0;
    };

    uint64_t next_post = 0;
    uint64_t next_send = 0;
    uint64_t next_ack = 0;

    while (next_ack < total_chunks) {
        while (next_post < total_chunks && next_post - next_ack < static_cast<uint64_t>(depth)) {
            int status = post_copy(next_post);
            if (status) {
                return status;
            }
            next_post++;
        }

        if (next_send < next_post) {
            int status = send_chunk(next_send);
            if (status) {
                return status;
            }
            next_send++;
            continue;
        }

        int status = wait_ack(next_ack);
        if (status) {
            return status;
        }
        next_ack++;
    }

    return 0;
}

static int staged_execute_get(nvshmem_transport_t transport, staged_client_op_t& op) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    const auto& rma = std::get<staged_rma_op_t>(op.operation);
    staged_qp_t& qp = staged_get_qp(s, 0, op.pe);
    char* dst = static_cast<char*>(rma.local.ptr);
    uint64_t remaining = static_cast<uint64_t>(rma.bytesdesc.elembytes) * rma.bytesdesc.nelems;
    uint64_t offset = 0;
    uint64_t remote_addr = reinterpret_cast<uint64_t>(rma.remote.ptr);

    while (remaining > 0) {
        size_t chunk =
            remaining > s->bounce_bytes ? s->bounce_bytes : static_cast<size_t>(remaining);
        staged_response_state_t::ticket response_ticket;
        int status = s->responses.reserve(&response_ticket);
        if (status != 0) {
            return status;
        }

        staged_ctrl_msg_t req{};
        req.op = staged_ctrl_op_t::GET_REQ;
        req.addr = remote_addr + offset;
        req.bytes = chunk;
        req.request_id = response_ticket.request_id;
        req.reply_addr = reinterpret_cast<uint64_t>(s->rdma.client_bounce());
        req.reply_rkey = s->rdma.bounce_mr()->rkey;

        status = send_ctrl(s, qp, &req);
        if (status) {
            return status;
        }

        status = staged_wait_response(s, response_ticket, nullptr);
        if (status) {
            return status;
        }

        status = staged_copy(transport, dst + offset, s->rdma.client_bounce(), chunk,
                             s->cuda.client_stream, "GET H2D");
        if (status) {
            return status;
        }

        remaining -= chunk;
        offset += chunk;
    }

    return 0;
}

static int staged_store_amo_return(nvshmem_transport_t transport, const amo_memdesc_t& target,
                                   const amo_bytesdesc_t& bytesdesc, uint64_t value) {
    auto* s = static_cast<transport_staged_state_t*>(transport->state);
    if (!target.retptr) {
        return 0;
    }

    if (target.retflag) {
        g_elem_t ret{};
        ret.data = value;
        ret.flag = target.retflag;
        return staged_copy(transport, target.retptr, &ret, sizeof(ret), s->cuda.client_stream,
                           "AMO return store");
    } else if (bytesdesc.elembytes == sizeof(uint16_t)) {
        uint16_t ret = static_cast<uint16_t>(value);
        return staged_copy(transport, target.retptr, &ret, sizeof(ret), s->cuda.client_stream,
                           "AMO return store");
    } else if (bytesdesc.elembytes == sizeof(uint32_t)) {
        uint32_t ret = static_cast<uint32_t>(value);
        return staged_copy(transport, target.retptr, &ret, sizeof(ret), s->cuda.client_stream,
                           "AMO return store");
    } else {
        uint64_t ret = value;
        return staged_copy(transport, target.retptr, &ret, sizeof(ret), s->cuda.client_stream,
                           "AMO return store");
    }
}

static int staged_execute_amo(nvshmem_transport_t transport, staged_client_op_t& op) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    const auto& amo = std::get<staged_amo_op_t>(op.operation);
    staged_qp_t& qp = staged_get_qp(s, 0, op.pe);
    bool is_fetch = amo.verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH;

    if (op.pe == transport->my_pe) {
        uint64_t old_value = 0;
        staged_ctrl_msg_t local_msg{};
        local_msg.flags =
            static_cast<uint32_t>(amo.verb.desc) | (amo.verb.is_float ? NVSHMEMI_AMO_FLOAT_BIT : 0);
        local_msg.bytes = static_cast<uint64_t>(amo.bytesdesc.elembytes);
        local_msg.value = amo.target.val;
        local_msg.cmp = amo.target.cmp;

        int status = 0;
        {
            std::lock_guard<std::mutex> lk(s->memory.amo_mutex);
            status = staged_apply_amo(transport, amo.target.remote_memdesc.ptr, local_msg,
                                      &old_value, s->cuda.client_stream);
        }
        if (status) {
            return status;
        }
        if (is_fetch) {
            return staged_store_amo_return(transport, amo.target, amo.bytesdesc, old_value);
        }
        return 0;
    }

    staged_response_state_t::ticket response_ticket;
    int status = s->responses.reserve(&response_ticket);
    if (status != 0) {
        return status;
    }
    staged_ctrl_msg_t ctrl{};
    ctrl.op = staged_ctrl_op_t::AMO;
    ctrl.flags =
        static_cast<uint32_t>(amo.verb.desc) | (amo.verb.is_float ? NVSHMEMI_AMO_FLOAT_BIT : 0);
    if (is_fetch) {
        ctrl.flags |= STAGED_CTRL_AMO_FETCH;
    }
    ctrl.addr = reinterpret_cast<uint64_t>(amo.target.remote_memdesc.ptr);
    ctrl.bytes = static_cast<uint64_t>(amo.bytesdesc.elembytes);
    ctrl.value = amo.target.val;
    ctrl.cmp = amo.target.cmp;
    ctrl.request_id = response_ticket.request_id;

    status = send_ctrl(s, qp, &ctrl);
    if (status) {
        return status;
    }

    staged_response_t response{};
    status = staged_wait_response(s, response_ticket, &response);
    if (status) {
        return status;
    }
    if (is_fetch) {
        return staged_store_amo_return(transport, amo.target, amo.bytesdesc, response.value);
    }
    return 0;
}

static int staged_execute_client_op(nvshmem_transport_t transport, staged_client_op_t& op) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    if (s->workers.stopping()) {
        return NVSHMEMX_ERROR_INTERNAL;
    }
    if (op.pe < 0 || op.pe >= s->rdma.num_pes()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (std::holds_alternative<staged_amo_op_t>(op.operation)) {
        return staged_execute_amo(transport, op);
    }

    if (op.pe == transport->my_pe) {
        return staged_execute_self_rma(transport, op);
    }

    const auto& rma = std::get<staged_rma_op_t>(op.operation);
    if (rma.verb.desc == NVSHMEMI_OP_P || rma.verb.desc == NVSHMEMI_OP_PUT) {
        return staged_execute_put(transport, op);
    } else if (rma.verb.desc == NVSHMEMI_OP_G || rma.verb.desc == NVSHMEMI_OP_GET) {
        return staged_execute_get(transport, op);
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

static void staged_client_loop(nvshmem_transport_t transport, staged_startup_latch_t& startup) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    staged_worker_cuda_scope_t cuda_scope(s->cuda, "client");
    int status = cuda_scope.status();
    startup.report(status);
    if (status) {
        staged_fail_control(s);
        return;
    }

    staged_operation_state_t::work_item item;
    while (s->operations.take(&item)) {
        staged_client_op_t& op = *item.operation;
        if (const auto* rma = std::get_if<staged_rma_op_t>(&op.operation)) {
            TRACE(s->log_level, "[STAGED PE%d client] begin seq=%lu %s pe=%d bytes=%lu",
                  transport->my_pe, static_cast<unsigned long>(item.sequence),
                  staged_rma_name(rma->verb), op.pe,
                  static_cast<unsigned long>(rma->bytesdesc.elembytes) *
                      static_cast<unsigned long>(rma->bytesdesc.nelems));
        } else {
            const auto& amo = std::get<staged_amo_op_t>(op.operation);
            TRACE(s->log_level, "[STAGED PE%d client] begin seq=%lu amo pe=%d bytes=%d",
                  transport->my_pe, static_cast<unsigned long>(item.sequence), op.pe,
                  amo.bytesdesc.elembytes);
        }

        status = staged_execute_client_op(transport, op);
        TRACE(s->log_level, "[STAGED PE%d client] end seq=%lu status=%d", transport->my_pe,
              static_cast<unsigned long>(item.sequence), status);

        if (status != 0) {
            s->workers.fail(status);
        }
        int completion_status = s->operations.complete(item.token, status);
        if (completion_status != 0) {
            if (status == 0) {
                s->workers.fail(completion_status);
            }
            return;
        }
        if (status != 0) {
            return;
        }
    }
}

static int staged_submit_client_op(transport_staged_state_t* s, staged_client_op_t op, bool wait) {
    const bool is_rma = std::holds_alternative<staged_rma_op_t>(op.operation);
    staged_operation_state_t::ticket ticket;
    int status = s->operations.submit(std::move(op), wait, &ticket);
    if (status != 0) {
        return status;
    }
    TRACE(s->log_level, "[STAGED client] enqueue slot=%zu generation=%lu kind=%s wait=%d",
          ticket.index, static_cast<unsigned long>(ticket.generation), is_rma ? "rma" : "amo",
          wait ? 1 : 0);
    return wait ? s->operations.wait(ticket) : 0;
}

static int staged_quiet_all(transport_staged_state_t* s) {
    int status = s->operations.quiet();
    TRACE(s->log_level, "[STAGED client] quiet status=%d", status);
    return status;
}

}  // namespace

/* ═══════════════════════  transport interface  ═══════════════════ */

static int nvshmemt_staged_can_reach_peer(int* access, nvshmem_transport_pe_info_t*,
                                          nvshmem_transport_t) {
    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ;
    return 0;
}

static int nvshmemt_staged_connect_endpoints(nvshmem_transport_t t, int*, int, int* out_qp_indices,
                                             int num_qps) {
    if (out_qp_indices != nullptr && num_qps > 0) {
        /* Host requests use the transport's PE-indexed internal QPs. */
        std::fill_n(out_qp_indices, num_qps, NVSHMEMX_QP_DEFAULT);
        return 0;
    }

    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(t->state);
    const int n = t->n_pes;
    const int me = t->my_pe;

    int local_status = 0;
    if (n <= 0 || me < 0 || me >= n || !s->rdma.qps_empty() || s->rdma.bounce_region() ||
        s->rdma.bounce_mr()) {
        local_status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    struct ibv_port_attr port_attr{};
    int verbs_status = 0;
    if (!local_status) {
        verbs_status = ftable.query_port(s->rdma.context(), s->rdma.ib_port(), &port_attr);
    }
    if (verbs_status != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] ibv_query_port failed for port %d: %s", me,
                             s->rdma.ib_port(), strerror(verbs_status));
        local_status = NVSHMEMX_ERROR_INTERNAL;
    }
    if (!local_status) {
        s->rdma.set_local_lid(port_attr.lid);
        s->rdma.set_path_mtu(port_attr.active_mtu ? port_attr.active_mtu : IBV_MTU_4096);
        if (s->rdma.path_mtu() > IBV_MTU_4096) {
            s->rdma.set_path_mtu(IBV_MTU_4096);
        }
    }

    if (!local_status && port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        if (s->rdma.local_lid() == 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED PE%d] InfiniBand port %d has no LID", me,
                                 s->rdma.ib_port());
            local_status = NVSHMEMX_ERROR_INTERNAL;
        }
    } else if (!local_status) {
        if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
            NVSHMEMI_ERROR_PRINT("[STAGED PE%d] unsupported link layer %u on port %d", me,
                                 port_attr.link_layer, s->rdma.ib_port());
            local_status = NVSHMEMX_ERROR_INTERNAL;
        }
    }
    if (!local_status &&
        (s->options.IB_PKEY_INDEX < 0 || s->options.IB_PKEY_INDEX >= port_attr.pkey_tbl_len)) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED PE%d] invalid NVSHMEM_IB_PKEY_INDEX %d for port %d: expected 0 <= "
            "NVSHMEM_IB_PKEY_INDEX < pkey_tbl_len (%hu)",
            me, s->options.IB_PKEY_INDEX, s->rdma.ib_port(), port_attr.pkey_tbl_len);
        local_status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (!local_status) {
        INFO(s->log_level, "[STAGED PE%d] using GID index %u MTU enum %d", me,
             static_cast<unsigned int>(s->rdma.local_gid_index()),
             static_cast<int>(s->rdma.path_mtu()));
        if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            INFO(s->log_level, "[STAGED PE%d] using InfiniBand LID %u", me, s->rdma.local_lid());
        }

        cudaError_t cerr = cudaGetDevice(&s->cuda.device);
        if (cerr != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED PE%d] cudaGetDevice failed during endpoint setup: %s", me,
                                 cudaGetErrorString(cerr));
            local_status = NVSHMEMX_ERROR_INTERNAL;
        }
    }
    if (!local_status && s->cuda.copy_policy == staged_copy_policy_t::STREAM) {
        local_status = staged_init_green_context(s);
    }

    int status = staged_collective_setup_status(t, local_status, "local endpoint setup");
    if (status) {
        return status;
    }

    status = staged_create_qp_resources(s, n, 1);
    status = staged_collective_setup_status(t, status, "QP and bounce-buffer setup");
    if (status) {
        return status;
    }

    /* Bootstrap all-to-all requires a contiguous block of QP handles per destination PE. */
    const size_t total_qps = staged_total_qps(s);
    std::vector<staged_ep_handle_t> local_eps(total_qps);
    for (int qp_slot = 0; qp_slot < s->rdma.qps_per_pe(); ++qp_slot) {
        for (int pe = 0; pe < n; ++pe) {
            staged_qp_t& qp = staged_get_qp(s, qp_slot, pe);
            staged_ep_handle_t& ep =
                local_eps[static_cast<size_t>(pe) * s->rdma.qps_per_pe() + qp_slot];
            ep.qpn = qp.qp()->qp_num;
            ep.rkey = s->rdma.bounce_mr()->rkey;
            ep.bounce_addr = reinterpret_cast<uint64_t>(qp.server_bounce());
            ep.gid = s->rdma.local_gid();
            ep.lid = s->rdma.local_lid();

            TRACE(s->log_level, "[STAGED PE%d] QP slot %d for PE%d: qpn=%u rkey=0x%x bounce=0x%lx",
                  me, qp_slot, pe, ep.qpn, ep.rkey, static_cast<unsigned long>(ep.bounce_addr));
        }
    }

    if (!t->boot_handle->alltoall) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] bootstrap alltoall is unavailable", me);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    std::vector<staged_ep_handle_t> remote_eps(total_qps);
    int rc =
        t->boot_handle->alltoall(local_eps.data(), remote_eps.data(),
                                 sizeof(staged_ep_handle_t) * s->rdma.qps_per_pe(), t->boot_handle);
    if (rc != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] endpoint exchange failed", me);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* ── Connect QPs: INIT → RTR → RTS ── */
    int qp_connect_status = 0;
    for (size_t q = 0; q < total_qps; ++q) {
        const int pe = static_cast<int>(q % static_cast<size_t>(n));
        const int qp_slot = static_cast<int>(q / static_cast<size_t>(n));
        if (pe == me) {
            continue;
        }
        staged_qp_t& qp = staged_get_qp(s, qp_slot, pe);
        qp.remote_endpoint() = remote_eps[static_cast<size_t>(pe) * s->rdma.qps_per_pe() + qp_slot];
        const staged_ep_handle_t& remote_ep = qp.remote_endpoint();

        char gid_str[INET6_ADDRSTRLEN] = {};
        inet_ntop(AF_INET6, remote_ep.gid.raw, gid_str, sizeof(gid_str));
        TRACE(s->log_level, "[STAGED PE%d] remote PE%d QP slot %d qpn=%u lid=%u gid=%s", me, pe,
              qp_slot, remote_ep.qpn, remote_ep.lid, gid_str);

        /* RTR */
        struct ibv_qp_attr rtr_attr{};
        rtr_attr.qp_state = IBV_QPS_RTR;
        rtr_attr.path_mtu = s->rdma.path_mtu();
        rtr_attr.dest_qp_num = remote_ep.qpn;
        rtr_attr.rq_psn = 0;
        rtr_attr.max_dest_rd_atomic = 1;
        rtr_attr.min_rnr_timer = 12;
        rtr_attr.ah_attr.dlid = remote_ep.lid;
        rtr_attr.ah_attr.sl = static_cast<uint8_t>(s->options.IB_SL);
        rtr_attr.ah_attr.src_path_bits = 0;
        rtr_attr.ah_attr.static_rate = 0;
        rtr_attr.ah_attr.port_num = static_cast<uint8_t>(s->rdma.ib_port());

        bool use_grh = port_attr.link_layer == IBV_LINK_LAYER_ETHERNET;
        if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            const nvshmemt_ib_qp_path path = nvshmemt_ib_select_qp_path(
                &s->rdma.local_gid(), s->rdma.local_lid(), remote_ep.lid,
                remote_ep.gid.global.subnet_prefix, remote_ep.gid.global.interface_id);
            rtr_attr.ah_attr.dlid = path.dlid;
            use_grh = s->options.IB_FORCE_GRH || nvshmemt_ib_common_port_requires_grh(&port_attr) ||
                      path.grh_required;
        }
        if (use_grh) {
            rtr_attr.ah_attr.is_global = 1;
            rtr_attr.ah_attr.grh.dgid = remote_ep.gid;
            rtr_attr.ah_attr.grh.sgid_index = s->rdma.local_gid_index();
            rtr_attr.ah_attr.grh.hop_limit = STAGED_GRH_HOP_LIMIT;
            rtr_attr.ah_attr.grh.traffic_class = static_cast<uint8_t>(s->options.IB_TRAFFIC_CLASS);
            rtr_attr.ah_attr.grh.flow_label = 0;
        } else {
            rtr_attr.ah_attr.is_global = 0;
        }

        int verbs_status =
            ftable.modify_qp(qp.qp(), &rtr_attr,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                 IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED PE%d] QP slot %d RTR failed for PE%d: %s", me, qp_slot,
                                 pe, strerror(verbs_status));
            qp_connect_status = NVSHMEMX_ERROR_INTERNAL;
            continue;
        }

        /* RTS */
        struct ibv_qp_attr rts_attr{};
        rts_attr.qp_state = IBV_QPS_RTS;
        rts_attr.sq_psn = 0;
        rts_attr.timeout = static_cast<uint8_t>(s->options.IB_TIMEOUT);
        rts_attr.retry_cnt = static_cast<uint8_t>(s->options.IB_RETRY_CNT);
        rts_attr.rnr_retry = 7;
        rts_attr.max_rd_atomic = 1;
        verbs_status =
            ftable.modify_qp(qp.qp(), &rts_attr,
                             IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                 IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED PE%d] QP slot %d RTS failed for PE%d: %s", me, qp_slot,
                                 pe, strerror(verbs_status));
            qp_connect_status = NVSHMEMX_ERROR_INTERNAL;
            continue;
        }

        INFO(s->log_level, "[STAGED PE%d] connected QP slot %d to PE%d", me, qp_slot, pe);
    }

    status = staged_collective_setup_status(t, qp_connect_status, "QP connection");
    if (status) {
        return status;
    }

    /* ── Launch workers ── */
    int worker_start_status = 0;
    try {
        s->worker_threads = std::make_unique<staged_worker_threads_t>(
            s->workers, [t](staged_startup_latch_t& startup) { staged_server_loop(t, startup); },
            [t](staged_startup_latch_t& startup) { staged_client_loop(t, startup); });
    } catch (const std::exception& err) {
        NVSHMEMI_ERROR_PRINT("[STAGED PE%d] failed to start worker thread: %s", me, err.what());
        worker_start_status = NVSHMEMX_ERROR_INTERNAL;
    }
    if (!worker_start_status) {
        worker_start_status = s->worker_threads->wait_for_start();
    }
    if (worker_start_status) {
        staged_stop_workers(s);
    }

    status = staged_collective_setup_status(t, worker_start_status, "worker startup");
    if (status) {
        staged_stop_workers(s);
        return status;
    }

    INFO(s->log_level, "[STAGED PE%d] transport ready", me);
    return 0;
}

static_assert(sizeof(staged_mem_handle_info_t*) <= NVSHMEM_MEM_HANDLE_SIZE,
              "staged memory-handle metadata pointer must fit in the opaque handle");

static staged_mem_handle_info_t* staged_load_mem_handle_info(
    const nvshmem_mem_handle_t* mem_handle) {
    if (!mem_handle) {
        return nullptr;
    }
    staged_mem_handle_info_t* info = nullptr;
    memcpy(&info, mem_handle->reserved, sizeof(info));
    return info;
}

static void staged_store_mem_handle_info(nvshmem_mem_handle_t* mem_handle,
                                         staged_mem_handle_info_t* info) {
    memcpy(mem_handle->reserved, &info, sizeof(info));
}

static int nvshmemt_staged_get_mem_handle(nvshmem_mem_handle_t* mem_handle, void* buf,
                                          size_t length, nvshmem_transport_t t,
                                          bool /* local_only */) {
    if (!mem_handle) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    memset(mem_handle->reserved, 0, sizeof(mem_handle->reserved));

    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(t->state);
    if (s->cuda.copy_policy != staged_copy_policy_t::GDRCOPY || !staged_is_gpu_memory(buf)) {
        return 0;
    }

    if (check_egm(buf, t->egm_map)) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] CPU-mapping policy cannot map EGM allocation %p len=%zu; "
            "memory registration failed",
            buf, length);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /*
     * Same-physical multi-VA case (mmap into symmetric heap): pin with VA1
     * (alias), keep buf (VA2) as the lookup key. See nvbug 5072809 / IBRC.
     */
    void* alias_va_ptr = nullptr;
    if (t->alias_va_map != nullptr) {
        auto alias_it = t->alias_va_map->find(buf);
        if (alias_it != t->alias_va_map->end()) {
            alias_va_ptr = alias_it->second;
        }
    }

    staged_mem_handle_info_t* stored_info = nullptr;
    int status =
        staged_gdrcopy_map(s, buf, length, alias_va_ptr ? alias_va_ptr : buf, &stored_info);
    if (status != 0) {
        return status;
    }
    staged_store_mem_handle_info(mem_handle, stored_info);
    return 0;
}

static int nvshmemt_staged_release_mem_handle(nvshmem_mem_handle_t* mem_handle,
                                              nvshmem_transport_t t) {
    staged_mem_handle_info_t* handle_info = staged_load_mem_handle_info(mem_handle);
    if (!handle_info) {
        return 0;
    }

    int status =
        staged_gdrcopy_unmap(static_cast<transport_staged_state_t*>(t->state), handle_info);
    if (status != 0) {
        return status;
    }
    staged_store_mem_handle_info(mem_handle, nullptr);
    return 0;
}

/* ═══  RMA  ═══════════════════════════════════════════════════════ */

static int nvshmemt_staged_rma(nvshmem_transport_t tcurr, int pe, rma_verb_t verb,
                               rma_memdesc_t* remote, rma_memdesc_t* local,
                               rma_bytesdesc_t bytesdesc, int /* qp_index */) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(tcurr->state);
    if (verb.desc != NVSHMEMI_OP_P && verb.desc != NVSHMEMI_OP_PUT && verb.desc != NVSHMEMI_OP_G &&
        verb.desc != NVSHMEMI_OP_GET) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return staged_submit_client_op(
        s, staged_client_op_t(pe, staged_rma_op_t{verb, *remote, *local, bytesdesc}), !verb.is_nbi);
}

static int nvshmemt_staged_fence(nvshmem_transport_t, int, int, int) { return 0; }
static int nvshmemt_staged_quiet(nvshmem_transport_t tcurr, int, int) {
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(tcurr->state);
    return staged_quiet_all(s);
}

/* ═══  AMO (control channel)  ════════════════════════════════════ */

static int nvshmemt_staged_amo(nvshmem_transport_t tcurr, int pe, void*, amo_verb_t verb,
                               amo_memdesc_t* target, amo_bytesdesc_t bytesdesc, int) {
    if (bytesdesc.elembytes != sizeof(uint16_t) && bytesdesc.elembytes != sizeof(uint32_t) &&
        bytesdesc.elembytes != sizeof(uint64_t)) {
        NVSHMEMI_ERROR_PRINT("[STAGED] AMO verb %d with size %d not implemented yet", verb.desc,
                             bytesdesc.elembytes);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(tcurr->state);

    return staged_submit_client_op(
        s, staged_client_op_t(pe, staged_amo_op_t{verb, *target, bytesdesc}), true);
}

/* ═══  finalize  ═════════════════════════════════════════════════ */

static int nvshmemt_staged_finalize(nvshmem_transport_t transport) {
    std::unique_ptr<transport_staged_state_t> state(
        static_cast<transport_staged_state_t*>(transport->state));
    if (!state) {
        return 0;
    }
    transport->state = nullptr;
    transport_staged_state_t* s = state.get();

    int final_status = staged_quiet_all(s);

    staged_stop_workers(s);

    int gdrcopy_status = staged_gdrcopy_fini(s);
    if (final_status == 0 && gdrcopy_status != 0) {
        final_status = gdrcopy_status;
    }

    int rdma_status = s->rdma.close();
    if (final_status == 0 && rdma_status != 0) {
        final_status = rdma_status;
    }
    nvshmemt_ibv_ftable_fini(&ibv_handle);

    return final_status;
}

static int nvshmemt_staged_show_info(nvshmem_transport_t, int) { return 0; }

/* ═══  init  ═════════════════════════════════════════════════════ */

extern "C" int nvshmemt_init(nvshmem_transport_t* t, struct nvshmemi_cuda_fn_table* cuda_syms,
                             int api_version) {
    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT("[STAGED] incompatible transport API version %d (plugin major %d)",
                             NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version),
                             NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    nvshmem_transport_t transport =
        static_cast<nvshmem_transport_t>(calloc(1, sizeof(struct nvshmem_transport)));
    if (!transport) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    transport_staged_state_t* state = new transport_staged_state_t();
    state->cuda.syms = cuda_syms;
    int fork_init_status = 0;

    auto& options = state->options;
    int status = nvshmemi_env_options_init(&options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize transport options.");
    nvshmemt_ib_common_sanitize_timeout(&options);
    status =
        nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, nvshmemt_common_get_log_level(&options));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to load libibverbs.");
    nvshmemt_ib_common_sanitize_retry_cnt(&options);
    state->log_level = nvshmemt_common_get_log_level(&options);
    INFO(state->log_level, "[STAGED] API version=%d plugin major=%d",
         NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version), NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);

    /* ── tunables ── */
    state->bounce_bytes = options.STAGED_BOUNCE_SIZE;
    if (state->bounce_bytes == 0 || state->bounce_bytes > STAGED_MAX_BOUNCE_BYTES) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] NVSHMEM_STAGED_BOUNCE_SIZE=%zu is outside the supported range 1-%zu",
            state->bounce_bytes, STAGED_MAX_BOUNCE_BYTES);
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }
    state->pipeline_depth = options.STAGED_PIPELINE_DEPTH;
    if (state->pipeline_depth < 1 || state->pipeline_depth > STAGED_MAX_PIPELINE_DEPTH) {
        NVSHMEMI_ERROR_PRINT("[STAGED] NVSHMEM_STAGED_PIPELINE_DEPTH=%d is outside 1-%d",
                             state->pipeline_depth, STAGED_MAX_PIPELINE_DEPTH);
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }
    if (strcasecmp(options.STAGED_COPY_POLICY, "STREAM") == 0) {
        state->cuda.copy_policy = staged_copy_policy_t::STREAM;
        if (options.STAGED_GREEN_CTX_SMS <= 0) {
            NVSHMEMI_ERROR_PRINT(
                "[STAGED] NVSHMEM_STAGED_GREEN_CTX_SMS must be greater than zero for STREAM");
            status = NVSHMEMX_ERROR_INVALID_VALUE;
            goto out;
        }
        INFO(state->log_level, "[STAGED] copy policy STREAM selected");
    } else if (strcasecmp(options.STAGED_COPY_POLICY, "GDRCOPY") == 0) {
        state->cuda.copy_policy = staged_copy_policy_t::GDRCOPY;
        status = staged_gdrcopy_init(state, cuda_syms);
        if (status) {
            goto out;
        }
    } else {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] invalid NVSHMEM_STAGED_COPY_POLICY='%s'; expected STREAM or GDRCOPY",
            options.STAGED_COPY_POLICY);
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }

    if (strcasecmp(options.STAGED_BOUNCE_LOCATION, "VIDMEM") == 0) {
        state->rdma.set_bounce_in_vidmem(true);
    } else if (strcasecmp(options.STAGED_BOUNCE_LOCATION, "SYSMEM") != 0) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] invalid NVSHMEM_STAGED_BOUNCE_LOCATION='%s'; expected SYSMEM or VIDMEM",
            options.STAGED_BOUNCE_LOCATION);
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }
    INFO(state->log_level, "[STAGED] bounce location %s selected",
         state->rdma.bounce_in_vidmem() ? "VIDMEM" : "SYSMEM");

    /* ── Open IB device ── */
    if (options.ENABLE_NIC_PE_MAPPING && options.HCA_PE_MAPPING_provided &&
        options.HCA_PE_MAPPING[0] && !(options.HCA_LIST_provided && options.HCA_LIST[0])) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] NVSHMEM_HCA_PE_MAPPING is not supported; select the single staged "
            "HCA/port with NVSHMEM_HCA_LIST");
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }

    fork_init_status = ftable.fork_init();
    if (fork_init_status != 0) {
        NVSHMEMI_WARN_PRINT("[STAGED] ibv_fork_init returned %d; continuing without fork safety",
                            fork_init_status);
    }

    status = staged_open_hca(state);
    if (status) {
        goto out;
    }

    INFO(state->log_level,
         "[STAGED] slot=%zu pipeline_depth=%d; one RC QP per PE will be allocated during "
         "endpoint connection",
         state->bounce_bytes, state->pipeline_depth);

    transport->state = state;
    transport->host_ops.can_reach_peer = nvshmemt_staged_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_staged_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_staged_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_staged_release_mem_handle;
    transport->host_ops.rma = nvshmemt_staged_rma;
    transport->host_ops.amo = nvshmemt_staged_amo;
    transport->host_ops.fence = nvshmemt_staged_fence;
    transport->host_ops.quiet = nvshmemt_staged_quiet;
    transport->host_ops.finalize = nvshmemt_staged_finalize;
    transport->host_ops.show_info = nvshmemt_staged_show_info;
    transport->host_ops.put_signal = nvshmemt_put_signal;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->atomics_complete_on_quiet = true;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;
    transport->n_devices = 0;
    transport->max_op_len = SIZE_MAX;
    /* Control-channel AMO responses are written by the host in host endian for every size. */
    transport->atomic_host_endian_min_size = 0;
    *t = transport;
    return 0;

out:
    staged_gdrcopy_fini(state);
    delete state;
    nvshmemt_ibv_ftable_fini(&ibv_handle);
    free(transport);
    return status ? status : NVSHMEMX_ERROR_INTERNAL;
}

}  // namespace nvshmemi
