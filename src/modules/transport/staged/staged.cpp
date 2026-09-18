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
#include <new>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
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

staged_qp_t::~staged_qp_t() {
    if (qp_) {
        ibv_destroy_qp(qp_);
    }
    if (send_cq_) {
        ibv_destroy_cq(send_cq_);
    }
    if (recv_cq_) {
        ibv_destroy_cq(recv_cq_);
    }
    if (ctrl_recv_mr_) {
        ibv_dereg_mr(ctrl_recv_mr_);
    }
}

staged_rdma_state_t::~staged_rdma_state_t() {
    qps_.qps.clear();
    if (ib_.bounce_mr) {
        ibv_dereg_mr(ib_.bounce_mr);
    }
    if (ib_.bounce_region) {
        cudaFreeHost(ib_.bounce_region);
    }
    if (ib_.protection_domain) {
        ibv_dealloc_pd(ib_.protection_domain);
    }
    if (ib_.context) {
        ibv_close_device(ib_.context);
    }
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

}  // namespace

}  // namespace nvshmemi
