/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Staged RDMA transport for NVSHMEM.
 * Uses ibverbs RC QPs with host-staged bounce buffers.
 * Data path: GPU --(D2H)--> pinned bounce --(RDMA Write)--> remote bounce --(H2D)--> GPU
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
#include <shared_mutex>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host_transport/transport_constants.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmemi_error_macros.h"
#include "staged_config.h"
#include "transport_common.h"
#include "transport_ib_common.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

namespace nvshmemi {

staged_qp_t::staged_qp_t(struct ibv_context* context, struct ibv_pd* pd, void* server_bounce)
    : context_(context), pd_(pd), server_bounce_(server_bounce) {}

staged_qp_t::~staged_qp_t() { close(); }

int staged_qp_t::close() noexcept {
    int status = 0;
    if (qp_) {
        int verbs_status = ibv_destroy_qp(qp_);
        if (verbs_status != 0) {
            /* A live QP may still reference its CQs and control receive MR. */
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_destroy_qp failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        qp_ = nullptr;
    }
    if (send_cq_) {
        int verbs_status = ibv_destroy_cq(send_cq_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_destroy_cq(send) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            send_cq_ = nullptr;
        }
    }
    if (recv_cq_) {
        int verbs_status = ibv_destroy_cq(recv_cq_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_destroy_cq(recv) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            recv_cq_ = nullptr;
        }
    }
    if (ctrl_recv_mr_) {
        int verbs_status = ibv_dereg_mr(ctrl_recv_mr_);
        if (verbs_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_dereg_mr(control) failed during cleanup: %s",
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

    if (ib_.bounce_mr) {
        int verbs_status = ibv_dereg_mr(ib_.bounce_mr);
        if (verbs_status != 0) {
            /* A registered MR still references its allocation and PD. */
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_dereg_mr(bounce) failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        ib_.bounce_mr = nullptr;
    }
    if (ib_.bounce_region) {
        cudaError_t cuda_status = cudaFreeHost(ib_.bounce_region);
        if (cuda_status != cudaSuccess) {
            NVSHMEMI_ERROR_PRINT("[STAGED] cudaFreeHost(bounce) failed during cleanup: %s",
                                 cudaGetErrorString(cuda_status));
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            ib_.bounce_region = nullptr;
            ib_.client_bounce = nullptr;
            ib_.bounce_region_size = 0;
        }
    }
    if (ib_.protection_domain) {
        int verbs_status = ibv_dealloc_pd(ib_.protection_domain);
        if (verbs_status != 0) {
            /* The device context must outlive a PD that could not be released. */
            NVSHMEMI_ERROR_PRINT("[STAGED] ibv_dealloc_pd failed during cleanup: %s",
                                 nvshmemt_strerror_from_status(verbs_status));
            return status == 0 ? NVSHMEMX_ERROR_INTERNAL : status;
        }
        ib_.protection_domain = nullptr;
    }
    if (ib_.context) {
        int verbs_status = ibv_close_device(ib_.context);
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
        ibv_reg_mr(pd_, ctrl_recv_bufs_.data(), ctrl_recv_bufs_.size() * sizeof(staged_ctrl_msg_t),
                   IBV_ACCESS_LOCAL_WRITE);
    if (!ctrl_recv_mr_) {
        NVSHMEMI_ERROR_PRINT("[STAGED] failed to register control receive buffers for QP %zu: %s",
                             index, strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    send_cq_ = ibv_create_cq(context_, STAGED_CQ_DEPTH, nullptr, nullptr, 0);
    recv_cq_ = ibv_create_cq(context_, STAGED_CQ_DEPTH, nullptr, nullptr, 0);
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
    qp_ = ibv_create_qp(pd_, &qp_attr);
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
    int status = ibv_modify_qp(
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

static int staged_query_usable_gid(transport_staged_state_t* s, struct ibv_context* ctx, int port,
                                   const struct ibv_port_attr* port_attr, int* gid_index,
                                   union ibv_gid* gid) {
    nvshmemt_ibv_function_table verbs{};
    verbs.get_device_name = ibv_get_device_name;
    verbs.query_gid = ibv_query_gid;

    int status = ib_get_gid_index(&verbs, ctx, static_cast<uint8_t>(port), port_attr, gid_index,
                                  s->log_level, &s->options);
    if (status != NVSHMEMX_SUCCESS) {
        return status;
    }
    if (ibv_query_gid(ctx, port, *gid_index, gid) != 0) {
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return NVSHMEMX_SUCCESS;
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

static int staged_cuda_copy_sync(transport_staged_state_t* s, void* dst, const void* src,
                                 size_t bytes, cudaStream_t stream, const char* label) {
    std::lock_guard<std::mutex> lk(s->cuda.copy_mutex);
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

    const size_t bounce_region_size = ring_count * ring_bytes;
    void* bounce_region = nullptr;
    cudaError_t cerr = cudaHostAlloc(&bounce_region, bounce_region_size, 0);
    if (cerr != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("[STAGED] cudaHostAlloc failed for %zu bytes: %s", bounce_region_size,
                             cudaGetErrorString(cerr));
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    struct ibv_mr* bounce_mr =
        ibv_reg_mr(rdma.protection_domain(), bounce_region, bounce_region_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!bounce_mr) {
        NVSHMEMI_ERROR_PRINT("[STAGED] ibv_reg_mr failed: %s", strerror(errno));
        cudaFreeHost(bounce_region);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    rdma.set_bounce_storage(bounce_region, bounce_region_size, bounce_region, bounce_mr);
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

    INFO(s->log_level,
         "[STAGED] allocating %zu QPs (%d per PE) and %zu bytes of pinned bounce "
         "storage",
         num_qps, rdma.qps_per_pe(), rdma.bounce_region_size());

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
                                 ibv_wc_status_str(wc.status));
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
            NVSHMEMI_ERROR_PRINT("[STAGED] RDMA Write failed: %s", ibv_wc_status_str(wc.status));
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return 0;
    });
}

static size_t staged_slot_offset(transport_staged_state_t* s, int slot) {
    return static_cast<size_t>(slot) * s->bounce_bytes;
}

static int staged_pointer_needs_cuda_copy(const void* ptr, bool* needs_cuda_copy) {
    if (!needs_cuda_copy) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    *needs_cuda_copy = false;

    cudaPointerAttributes attrs{};
    cudaError_t error = cudaPointerGetAttributes(&attrs, ptr);
    if (error == cudaErrorInvalidValue) {
        /* Unregistered host memory is not known to the CUDA runtime. */
        (void)cudaGetLastError();
        return 0;
    }
    if (error != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("[STAGED] cudaPointerGetAttributes(%p) failed: %s", ptr,
                             cudaGetErrorString(error));
        (void)cudaGetLastError();
        return NVSHMEMX_ERROR_INTERNAL;
    }

    *needs_cuda_copy = attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
    return 0;
}

#ifdef NVSHMEM_USE_GDRCOPY
/* Caller holds memory.mem_handle_mutex in shared mode for the lifetime of the returned pointer. */
static staged_mem_handle_info_t* staged_mem_handle_info_for_ptr(transport_staged_state_t* s,
                                                                const void* ptr, size_t bytes) {
    if (!ptr) {
        return nullptr;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    auto it = std::find_if(s->memory.mem_handle_infos.begin(), s->memory.mem_handle_infos.end(),
                           [addr, bytes](const auto& entry) {
                               const auto& info = entry.second;
                               if (!info || !info->ptr) {
                                   return false;
                               }
                               const uintptr_t begin = reinterpret_cast<uintptr_t>(info->ptr);
                               if (addr < begin) {
                                   return false;
                               }
                               const size_t offset = static_cast<size_t>(addr - begin);
                               return offset <= info->size && bytes <= info->size - offset;
                           });
    return it == s->memory.mem_handle_infos.end() ? nullptr : it->second.get();
}

#endif

enum class staged_copy_direction_t {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
};

static int staged_copy_host_device(nvshmem_transport_t transport, void* dst, const void* src,
                                   size_t bytes, cudaStream_t stream, const char* label,
                                   staged_copy_direction_t direction) {
    auto* s = static_cast<transport_staged_state_t*>(transport->state);
    if (bytes == 0) {
        return 0;
    }

    const void* gpu_ptr = direction == staged_copy_direction_t::DEVICE_TO_HOST ? src : dst;
    bool needs_cuda_copy = false;
    int status = staged_pointer_needs_cuda_copy(gpu_ptr, &needs_cuda_copy);
    if (status != 0) {
        return status;
    }
    if (!needs_cuda_copy) {
        memcpy(dst, src, bytes);
        return 0;
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (s->cuda.copy_policy == staged_copy_policy_t::GDRCOPY) {
        std::shared_lock<std::shared_mutex> handle_lock(s->memory.mem_handle_mutex);
        auto* info = staged_mem_handle_info_for_ptr(s, gpu_ptr, bytes);
        if (!info || !info->cpu_mapping.mapped || !info->cpu_mapping.cpu_ptr) {
            NVSHMEMI_ERROR_PRINT(
                "[STAGED] %s requires a GPU CPU mapping for GPU address %p (bytes=%zu)", label,
                gpu_ptr, bytes);
            return NVSHMEMX_ERROR_INTERNAL;
        }

        const uintptr_t addr = reinterpret_cast<uintptr_t>(gpu_ptr);
        const uintptr_t begin = reinterpret_cast<uintptr_t>(info->ptr);
        void* mapped_gpu_ptr = static_cast<char*>(info->cpu_mapping.cpu_ptr) + (addr - begin);
        std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
        int status =
            direction == staged_copy_direction_t::DEVICE_TO_HOST
                ? nvshmemt_gpu_cpu_copy_from(&s->memory.gpu_cpu_mapping_state, &info->cpu_mapping,
                                             dst, mapped_gpu_ptr, bytes)
                : nvshmemt_gpu_cpu_copy_to(&s->memory.gpu_cpu_mapping_state, &info->cpu_mapping,
                                           mapped_gpu_ptr, src, bytes);
        if (status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] %s GPU CPU mapping %s copy failed: %d", label,
                                 direction == staged_copy_direction_t::DEVICE_TO_HOST
                                     ? "device-to-host"
                                     : "host-to-device",
                                 status);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        if (direction == staged_copy_direction_t::HOST_TO_DEVICE) {
            STORE_BARRIER();
        }
        return 0;
    }
#endif

    if (s->cuda.copy_policy == staged_copy_policy_t::GDRCOPY) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s has no GPU CPU mapping; refusing CUDA fallback", label);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return staged_cuda_copy_sync(s, dst, src, bytes, stream, label);
}

static int staged_copy_to_host(nvshmem_transport_t transport, void* host_dst, const void* src,
                               size_t bytes, cudaStream_t stream, const char* label) {
    return staged_copy_host_device(transport, host_dst, src, bytes, stream, label,
                                   staged_copy_direction_t::DEVICE_TO_HOST);
}

static int staged_copy_from_host(nvshmem_transport_t transport, void* dst, const void* host_src,
                                 size_t bytes, cudaStream_t stream, const char* label) {
    return staged_copy_host_device(transport, dst, host_src, bytes, stream, label,
                                   staged_copy_direction_t::HOST_TO_DEVICE);
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
    int status = staged_copy_to_host(transport, &old_value, target, sizeof(old_value), stream,
                                     "AMO load target");
    if (status) {
        return status;
    }

    status = staged_compute_amo_value(old_value, msg, &new_value);
    if (status) {
        return status;
    }

    status = staged_copy_from_host(transport, target, &new_value, sizeof(new_value), stream,
                                   "AMO store target");
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
                                     ibv_wc_status_str(wc.status));
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
                int status = staged_copy_from_host(
                    transport, dst, static_cast<char*>(qp.server_bounce()) + slot_offset, len,
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

                int status = staged_copy_to_host(transport, qp.server_bounce(), src, len,
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

    if (s->cuda.copy_policy == staged_copy_policy_t::GDRCOPY) {
        uint64_t remaining = total;
        uint64_t offset = 0;
        while (remaining > 0) {
            size_t chunk =
                remaining > s->bounce_bytes ? s->bounce_bytes : static_cast<size_t>(remaining);
            int status = staged_copy_to_host(transport, s->rdma.client_bounce(), src + offset,
                                             chunk, nullptr, "self RMA D2H");
            if (status) {
                return status;
            }
            status = staged_copy_from_host(transport, dst + offset, s->rdma.client_bounce(), chunk,
                                           nullptr, "self RMA H2D");
            if (status) {
                return status;
            }
            remaining -= chunk;
            offset += chunk;
        }
        return 0;
    }

    if (!s->cuda.client_stream) {
        NVSHMEMI_ERROR_PRINT("[STAGED] self RMA requires a Green client stream");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return staged_cuda_copy_sync(s, dst, src, static_cast<size_t>(total), s->cuda.client_stream,
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

    bool src_needs_cuda_copy = false;
    int pointer_status = staged_pointer_needs_cuda_copy(src, &src_needs_cuda_copy);
    if (pointer_status != 0) {
        return pointer_status;
    }
    const bool cuda_copy =
        src_needs_cuda_copy && s->cuda.copy_policy == staged_copy_policy_t::STREAM;
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

        if (!src_needs_cuda_copy) {
            memcpy(bounce, src + offset, chunk);
            return 0;
        }

        if (s->cuda.copy_policy == staged_copy_policy_t::GDRCOPY) {
            return staged_copy_to_host(transport, bounce, src + offset, chunk,
                                       s->cuda.client_stream, "PUT D2H");
        }

        {
            std::lock_guard<std::mutex> lk(s->cuda.copy_mutex);
            cudaError_t cerr = cudaMemcpyAsync(bounce, src + offset, chunk, cudaMemcpyDefault,
                                               s->cuda.client_stream);
            if (cerr != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED] cudaMemcpyAsync D2H failed: %s",
                                     cudaGetErrorString(cerr));
                return NVSHMEMX_ERROR_INTERNAL;
            }
            cerr = cudaEventRecord(s->cuda.client_events[slot], s->cuda.client_stream);
            if (cerr != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED] cudaEventRecord failed: %s",
                                     cudaGetErrorString(cerr));
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
        return 0;
    };

    auto wait_copy = [&](uint64_t chunk_idx) -> int {
        int slot = static_cast<int>(chunk_idx % static_cast<uint64_t>(depth));
        if (!cuda_copy) {
            return 0;
        }
        {
            std::lock_guard<std::mutex> lk(s->cuda.copy_mutex);
            cudaError_t cerr = cudaEventSynchronize(s->cuda.client_events[slot]);
            if (cerr != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("[STAGED] cudaEventSynchronize failed: %s",
                                     cudaGetErrorString(cerr));
                return NVSHMEMX_ERROR_INTERNAL;
            }
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

        status = staged_copy_from_host(transport, dst + offset, s->rdma.client_bounce(), chunk,
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
        return staged_copy_from_host(transport, target.retptr, &ret, sizeof(ret),
                                     s->cuda.client_stream, "AMO return store");
    } else if (bytesdesc.elembytes == sizeof(uint16_t)) {
        uint16_t ret = static_cast<uint16_t>(value);
        return staged_copy_from_host(transport, target.retptr, &ret, sizeof(ret),
                                     s->cuda.client_stream, "AMO return store");
    } else if (bytesdesc.elembytes == sizeof(uint32_t)) {
        uint32_t ret = static_cast<uint32_t>(value);
        return staged_copy_from_host(transport, target.retptr, &ret, sizeof(ret),
                                     s->cuda.client_stream, "AMO return store");
    } else {
        uint64_t ret = value;
        return staged_copy_from_host(transport, target.retptr, &ret, sizeof(ret),
                                     s->cuda.client_stream, "AMO return store");
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
        verbs_status = ibv_query_port(s->rdma.context(), s->rdma.ib_port(), &port_attr);
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
            ibv_modify_qp(qp.qp(), &rtr_attr,
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
            ibv_modify_qp(qp.qp(), &rts_attr,
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

#ifdef NVSHMEM_USE_GDRCOPY
static int staged_release_gpu_cpu_mapping(transport_staged_state_t* s,
                                          staged_mem_handle_info_t& info) {
    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    return nvshmemt_gpu_cpu_unmap(&s->memory.gpu_cpu_mapping_state, &info.cpu_mapping);
}
#endif
static_assert(sizeof(staged_mem_handle_info_t*) <= NVSHMEM_MEM_HANDLE_SIZE,
              "staged memory-handle metadata pointer must fit in the opaque handle");

static staged_mem_handle_info_t* staged_load_mem_handle_info(
    const nvshmem_mem_handle_t* mem_handle) {
    if (!mem_handle) {
        return nullptr;
    }
    staged_mem_handle_info_t* info = nullptr;
    memcpy(&info, mem_handle, sizeof(info));
    return info;
}

static void staged_store_mem_handle_info(nvshmem_mem_handle_t* mem_handle,
                                         staged_mem_handle_info_t* info) {
    memcpy(mem_handle, &info, sizeof(info));
}

static int nvshmemt_staged_get_mem_handle(nvshmem_mem_handle_t* mem_handle, void* buf,
                                          size_t length, nvshmem_transport_t t,
                                          bool /* local_only */) {
    if (!mem_handle) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    memset(mem_handle, 0, sizeof(*mem_handle));

#ifdef NVSHMEM_USE_GDRCOPY
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(t->state);
    if (s->cuda.copy_policy != staged_copy_policy_t::GDRCOPY) {
        return 0;
    }
    bool needs_cuda_copy = false;
    int status = staged_pointer_needs_cuda_copy(buf, &needs_cuda_copy);
    if (status != 0) {
        return status;
    }
    if (!needs_cuda_copy) {
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
    if (t->alias_va_map != nullptr && t->alias_va_map->count(buf)) {
        alias_va_ptr = t->alias_va_map->operator[](buf);
    }

    auto handle_info = std::make_unique<staged_mem_handle_info_t>();
    handle_info->ptr = buf;
    handle_info->size = length;
    void* mapping_buf = alias_va_ptr ? alias_va_ptr : buf;
    int mapping_status = 0;
    {
        /* Application code can register buffers concurrently. Serialize all GPU CPU mapping
         * operations with worker copies and teardown. */
        std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
        mapping_status =
            nvshmemt_gpu_cpu_map(&s->memory.gpu_cpu_mapping_state, mapping_buf, length,
                                 NVSHMEMT_GPU_CPU_MAPPING_FLAG_NONE, &handle_info->cpu_mapping);
    }
    if (mapping_status != 0) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] CPU-mapping policy (%s) failed for %p len=%zu status=%d; "
            "memory registration failed",
            nvshmemt_gpu_cpu_mapping_backend_name(&s->memory.gpu_cpu_mapping_state), mapping_buf,
            length, mapping_status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    staged_mem_handle_info_t* stored_info = handle_info.get();
    {
        std::unique_lock<std::shared_mutex> lk(s->memory.mem_handle_mutex);
        s->memory.mem_handle_infos.insert(stored_info);
    }
    staged_store_mem_handle_info(mem_handle, handle_info.release());
    return 0;
#else
    (void)buf;
    (void)length;
    (void)t;
    return 0;
#endif
}

static int nvshmemt_staged_release_mem_handle(nvshmem_mem_handle_t* mem_handle,
                                              nvshmem_transport_t t) {
    staged_mem_handle_info_t* handle_info = staged_load_mem_handle_info(mem_handle);
    if (!handle_info) {
        return 0;
    }

#ifdef NVSHMEM_USE_GDRCOPY
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(t->state);
    bool retrying_retired_cleanup = false;
    {
        std::unique_lock<std::shared_mutex> lk(s->memory.mem_handle_mutex);
        auto it = s->memory.mem_handle_infos.find(handle_info);
        if (it != s->memory.mem_handle_infos.end()) {
            s->memory.mem_handle_infos.erase(it);
        } else {
            auto retired_it = s->memory.retired_mem_handle_infos.find(handle_info);
            if (retired_it == s->memory.retired_mem_handle_infos.end()) {
                return 0;
            }
            retrying_retired_cleanup = true;
        }
    }

    int status = staged_release_gpu_cpu_mapping(s, *handle_info);
    if (status) {
        NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping cleanup failed status=%d", status);
        std::unique_lock<std::shared_mutex> lk(s->memory.mem_handle_mutex);
        s->memory.retired_mem_handle_infos.insert(handle_info);
        return NVSHMEMX_ERROR_INTERNAL;
    }
    if (retrying_retired_cleanup) {
        std::unique_lock<std::shared_mutex> lk(s->memory.mem_handle_mutex);
        s->memory.retired_mem_handle_infos.erase(handle_info);
    }
    delete handle_info;
    staged_store_mem_handle_info(mem_handle, nullptr);
#else
    (void)t;
    staged_store_mem_handle_info(mem_handle, nullptr);
#endif
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
    transport_staged_state_t* s = static_cast<transport_staged_state_t*>(transport->state);
    if (!s) {
        return 0;
    }

    staged_quiet_all(s);

    staged_stop_workers(s);

    std::unordered_set<staged_mem_handle_info_t*> mem_handle_infos;
    {
        std::unique_lock<std::shared_mutex> lk(s->memory.mem_handle_mutex);
        mem_handle_infos.swap(s->memory.mem_handle_infos);
        mem_handle_infos.insert(s->memory.retired_mem_handle_infos.begin(),
                                s->memory.retired_mem_handle_infos.end());
        s->memory.retired_mem_handle_infos.clear();
    }
    for (auto* info : mem_handle_infos) {
#ifdef NVSHMEM_USE_GDRCOPY
        int status = staged_release_gpu_cpu_mapping(s, *info);
        if (status) {
            NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping final cleanup failed status=%d", status);
        }
#endif
        delete info;
    }
#ifdef NVSHMEM_USE_GDRCOPY
    nvshmemt_gpu_cpu_mapping_fini(&s->memory.gpu_cpu_mapping_state);
#endif

    delete s;
    transport->state = nullptr;
    return 0;
}

static int nvshmemt_staged_show_info(nvshmem_transport_t, int) { return 0; }

/* ═══  init  ═════════════════════════════════════════════════════ */

}  // namespace nvshmemi
