/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <array>
#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <algorithm>
#include <vector>

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host/scope_guard.h"
#include "transport_common.h"
#include "transport_ib_common.h"  // for nvshmemt_ib_common_mem_handle
#include "transport_mlx5_common.h"
#include "device_host_transport/nvshmem_common_gpunetio.h"
#include "gpunetio/doca_gpunetio_host.h"

#define DOCA_CHECK(call)                                                \
    do {                                                                \
        doca_error_t RES = call;                                        \
        if (RES != DOCA_SUCCESS) {                                      \
            INFO(gpunetio_state->log_level, "gpunetio error: %d", RES); \
            return NVSHMEMX_ERROR_INTERNAL;                             \
        }                                                               \
    } while (0)

#define CUDA_RUNTIME_CHECK_RET(stmt, err)                                         \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            return (err);                                                         \
        }                                                                         \
    } while (0)

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

constexpr int MAX_GPU_PCI_ADDRESS_LEN = 32U;

// GPUNetIO-specific constants
constexpr int GPUNETIO_GPAGE_BITS = 16;
constexpr size_t GPUNETIO_GPAGE_SIZE = (1ULL << GPUNETIO_GPAGE_BITS);

// First slot is reserved for non-fetch operations.
constexpr int GPUNETIO_IBUF_RESERVED_SLOTS = 1;

// QP connection parameters
constexpr uint32_t GPUNETIO_QP_PSN = 0;
constexpr uint32_t GPUNETIO_QP_PKEY_INDEX = 0;
constexpr int GPUNETIO_QP_ACK_TIMEOUT = 20;
constexpr int GPUNETIO_QP_RETRY_CNT = 7;
constexpr int GPUNETIO_QP_RNR_RETRY = 7;
constexpr int GPUNETIO_QP_MIN_RNR_TIMER = 12;
constexpr int GPUNETIO_QP_HOP_LIMIT = 255;
constexpr bool GPUNETIO_QP_ALLOW_REMOTE_WRITE = true;
constexpr bool GPUNETIO_QP_ALLOW_REMOTE_READ = true;

// Memory objects and internal buffers
struct gpunetio_mem_object {
    enum doca_gpu_mem_type mem_type;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } aligned;
};

struct gpunetio_internal_buffer {
    gpunetio_mem_object *mem_object;
    nvshmemt_ib_common_mem_handle *mem_handle;
};

// Memory handles (registration / remote access)
struct gpunetio_mem_handle {
    nvshmemt_ib_common_mem_handle dev_mem_handles[NVSHMEMI_GPUNETIO_MAX_DEVICES_PER_PE];
    int num_devs;
};

struct gpunetio_device_local_only_mhandle_cache {
    nvshmemi_gpunetio_device_local_only_mhandle_t mhandle;
    void *dev_ptr;
};

// Endpoint and exchange info
struct gpunetio_ep {
    doca_gpu_verbs_qp_hl *qp;
    uint32_t qpn;
    int portid;
    uint32_t user_index;
    gpunetio_internal_buffer internal_buf;
};

struct gpunetio_exch_info {
    int lid;
    int qpn;
    union ibv_gid gid;
    doca_verbs_gid vgid;
};

// Device (single IB NIC)
struct gpunetio_device {
    // Flags
    int num_eps_per_pe = 0;
    // Common device information
    nvshmemt_ib_common_device common_device = {};
    // GPUNetIO-specific device information
    doca_dev_t *net_dev = nullptr;
    // Local RC endpoints for this device
    std::vector<gpunetio_ep *> rc_eps;
    // This mutex is required to avoid a race with the progress thread and the QP-specific API
    // reallocating and modifying rc_eps.
    std::unique_ptr<std::mutex> rc_eps_mtx{new std::mutex()};
    doca_verbs_ah_attr_t *ah = nullptr;
    doca_gpu_verbs_qp_hl *qp_local_backup = nullptr;
    doca_gpu_dev_verbs_nic_handler nic_handler_request = {};
};

// Transport instance state
struct nvshmemt_gpunetio_state_t {
    // Per-instance transport state
    std::vector<gpunetio_device> devices;
    std::vector<int> dev_ids;
    std::vector<int> port_ids;
    std::vector<int> selected_dev_ids;
    // Global array of all QPs from all devices (to be transferred to GPU later on)
    std::vector<nvshmemi_gpunetio_device_qp_t> qp_h;
    // Global array of lkeys / rkeys
    std::vector<nvshmemi_gpunetio_device_key_t> device_lkeys;
    std::vector<nvshmemi_gpunetio_device_key_t> device_rkeys;
    std::vector<gpunetio_device_local_only_mhandle_cache> device_local_only_mhandles;
    // Pointer to GPU buffer lkeys / rkeys.
    void *device_lkeys_d = nullptr;
    void *device_rkeys_d = nullptr;
    // GPU handlers
    doca_gpu_t *gpu_device = nullptr;
    cudaStream_t my_stream = nullptr;
    CUdevice cached_gpu_device_id = 0;
    // Global flags
    std::unique_ptr<nvshmemi_options_s> options;
    int log_level = 0;
    bool skip_cst = false;
    bool dmabuf_support_for_data_buffers = false;
    bool connect_endpoints_first_call = false;
    int last_device_index = 0;
    int cur_qp_index = 0;
    int last_num_rcs = 0;
    int qp_depth = 0;
    int num_fetch_slots_per_rc = 0;
    int num_requests_in_batch = 0;
    // Function tables
    nvshmemt_ibv_function_table ftable = {};
    void *ibv_handle = nullptr;
#ifdef NVSHMEM_USE_MLX5DV
    nvshmemt_mlx5dv_function_table mlx5dv_ftable = {};
    void *mlx5dv_handle = nullptr;
#endif
    nvshmemi_cuda_fn_table *cuda_syms = nullptr;
};

// Forward declarations
static bool gpunetio_qp_requires_cpu_proxy(doca_gpu_verbs_qp_hl *qp);
static void gpunetio_activate_progress_function(nvshmem_transport_t t);
int nvshmemt_gpunetio_progress(nvshmem_transport_t t);

// *** Utility functions ***
static inline int gpunetio_round_up_pow2(int n) {
    int pow2 = 0;
    for (pow2 = 1; pow2 < n; pow2 <<= 1)
        ;
    return pow2;
}

static constexpr int gpunetio_round_up_pow2_or_0(int n) {
    return (n == 0) ? 0 : gpunetio_round_up_pow2(n);
}

// Internal buffer
static int gpunetio_destroy_internal_buffer(gpunetio_internal_buffer *internal_buf,
                                            nvshmemt_gpunetio_state_t *gpunetio_state) {
    int status = 0;

    if (internal_buf->mem_handle) {
        nvshmemt_ib_common_release_mem_handle(
            &gpunetio_state->ftable,
            reinterpret_cast<nvshmem_mem_handle_t *>(internal_buf->mem_handle),
            gpunetio_state->log_level);
        delete internal_buf->mem_handle;
        internal_buf->mem_handle = nullptr;
    }

    if (internal_buf->mem_object) {
        if (internal_buf->mem_object->aligned.gpu_ptr) {
            doca_gpu_mem_free(gpunetio_state->gpu_device,
                              internal_buf->mem_object->aligned.gpu_ptr);
        }
        delete internal_buf->mem_object;
        internal_buf->mem_object = nullptr;
    }

    return status;
}

static int gpunetio_create_internal_buffer(gpunetio_internal_buffer *internal_buf,
                                           nvshmemt_gpunetio_state_t *gpunetio_state,
                                           gpunetio_device *device, size_t size) {
    int status = 0;
    auto guard =
        make_scope_guard([&]() { gpunetio_destroy_internal_buffer(internal_buf, gpunetio_state); });

    internal_buf->mem_object = new gpunetio_mem_object();
    internal_buf->mem_object->mem_type = DOCA_GPU_MEM_TYPE_GPU;
    internal_buf->mem_object->aligned.cpu_ptr = nullptr;
    internal_buf->mem_object->aligned.size = size;

    status = doca_gpu_mem_alloc(gpunetio_state->gpu_device, internal_buf->mem_object->aligned.size,
                                GPUNETIO_GPAGE_SIZE, DOCA_GPU_MEM_TYPE_GPU,
                                &internal_buf->mem_object->aligned.gpu_ptr, nullptr);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "cannot allocate internal buffer.\n");

    internal_buf->mem_handle = new nvshmemt_ib_common_mem_handle();

    status = nvshmemt_ib_common_reg_mem_handle(
        &gpunetio_state->ftable, &gpunetio_state->mlx5dv_ftable, device->common_device.pd,
        reinterpret_cast<nvshmem_mem_handle_t *>(internal_buf->mem_handle),
        internal_buf->mem_object->aligned.gpu_ptr, internal_buf->mem_object->aligned.size, false,
        gpunetio_state->dmabuf_support_for_data_buffers, gpunetio_state->cuda_syms,
        gpunetio_state->log_level, gpunetio_state->options->IB_ENABLE_RELAXED_ORDERING,
        device->common_device.data_direct);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "Unable to register memory for DOCA transport.\n");

    guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

// Endpoint lifecycle
static int gpunetio_destroy_ep(gpunetio_ep *ep, nvshmemt_gpunetio_state_t *gpunetio_state) {
    int status = 0;

    if (!ep) {
        return status;
    }

    if (ep->qp) {
        status = doca_gpu_verbs_destroy_qp_hl(ep->qp);
        if (status) {
            NVSHMEMI_WARN_PRINT("doca_gpu_verbs_destroy_qp_hl failed for ep %p \n", ep);
        }
    }

    if (ep->internal_buf.mem_handle) {
        status = gpunetio_destroy_internal_buffer(&ep->internal_buf, gpunetio_state);
        if (status) {
            NVSHMEMI_WARN_PRINT("gpunetio_destroy_internal_buffer failed for ep %p \n", ep);
        }
    }

    delete ep;

    return status;
}

// Device and exchange info
static int gpunetio_fill_exch_info(gpunetio_exch_info *exch_info, const gpunetio_ep *ep,
                                   const gpunetio_device *device) {
    const ibv_port_attr *port_attr = &device->common_device.port_attr[ep->portid - 1];
    const union ibv_gid *gid = &device->common_device.gid_info[ep->portid - 1].local_gid;

    exch_info->lid = port_attr->lid;
    exch_info->qpn = ep->qpn;

    memcpy(exch_info->gid.raw, gid->raw, sizeof(union ibv_gid));
    memcpy(exch_info->vgid.raw, gid->raw, sizeof(union ibv_gid));
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_get_cuda_device_id(nvshmemt_gpunetio_state_t *gpunetio_state,
                                       CUdevice *gpu_device_id) {
    if (CUPFN(gpunetio_state->cuda_syms, cuCtxGetDevice(gpu_device_id))) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetDevice failed.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return NVSHMEMX_SUCCESS;
}

static bool gpunetio_cst_is_required(nvshmemt_gpunetio_state_t *gpunetio_state,
                                     gpunetio_device *device, CUdevice dev_id) {
    bool rval = true;

    int order = 0;
    if (CUPFN(gpunetio_state->cuda_syms,
              cuDeviceGetAttribute(
                  &order, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                  dev_id))) {
        NVSHMEMI_WARN_PRINT("Cannot query dev attr. Assuming no GDR write ordering\n");
    } else {
        // GPU guarantees incoming PCIe write ordering. No need to do CST.
        if (order >= CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) rval = false;
    }
    rval = rval || device->common_device.data_direct;
    return rval;
}

// QP connection logic
static int gpunetio_create_ah(nvshmemt_gpunetio_state_t *gpunetio_state, gpunetio_device *device,
                              int portid) {
    DOCA_CHECK(doca_verbs_ah_attr_create(device->net_dev, &device->ah));
    auto ah_deleter = [](doca_verbs_ah_attr_t *p) { doca_verbs_ah_attr_destroy(p); };
    std::unique_ptr<doca_verbs_ah_attr_t, decltype(ah_deleter)> ah_guard(device->ah, ah_deleter);

    DOCA_CHECK(doca_verbs_ah_attr_set_sl(device->ah, gpunetio_state->options->IB_SL));
    DOCA_CHECK(doca_verbs_ah_attr_set_traffic_class(device->ah,
                                                    gpunetio_state->options->IB_TRAFFIC_CLASS));

    if (device->common_device.port_attr[portid - 1].link_layer == 1) {
        DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(device->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
    } else {
        DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(device->ah, DOCA_VERBS_ADDR_TYPE_IPv4));
        DOCA_CHECK(doca_verbs_ah_attr_set_hop_limit(device->ah, GPUNETIO_QP_HOP_LIMIT));
    }

    DOCA_CHECK(doca_verbs_ah_attr_set_sgid_index(
        device->ah, device->common_device.gid_info[portid - 1].local_gid_index));

    ah_guard.release();
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_create_qp_attr(nvshmemt_gpunetio_state_t *gpunetio_state, ibv_context *context,
                                   doca_verbs_qp_attr_t **out_verbs_qp_attr,
                                   doca_verbs_ah_attr_t *ah, int portid, uint32_t dest_qp_num) {
    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    DOCA_CHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
    auto deleter = [](doca_verbs_qp_attr_t *p) { doca_verbs_qp_attr_destroy(p); };
    std::unique_ptr<doca_verbs_qp_attr_t, decltype(deleter)> attr_uptr(verbs_qp_attr, deleter);

    doca_verbs_device_attr *verbs_device_attr;
    DOCA_CHECK(doca_verbs_query_device(context, &verbs_device_attr));
    uint8_t max_rd_atomic = doca_verbs_device_attr_get_max_qp_rd_atom(verbs_device_attr);
    uint8_t max_dest_rd_atomic = doca_verbs_device_attr_get_max_qp_init_rd_atom(verbs_device_attr);

    DOCA_CHECK(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, GPUNETIO_QP_PSN));
    DOCA_CHECK(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, GPUNETIO_QP_PSN));
    DOCA_CHECK(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, GPUNETIO_QP_PKEY_INDEX));
    DOCA_CHECK(doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_VERBS_MTU_SIZE_4K_BYTES));
    DOCA_CHECK(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, portid));
    DOCA_CHECK(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, GPUNETIO_QP_ACK_TIMEOUT));
    DOCA_CHECK(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, GPUNETIO_QP_RETRY_CNT));
    DOCA_CHECK(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, GPUNETIO_QP_RNR_RETRY));
    DOCA_CHECK(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, GPUNETIO_QP_MIN_RNR_TIMER));
    DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT));
    DOCA_CHECK(
        doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, GPUNETIO_QP_ALLOW_REMOTE_WRITE));
    DOCA_CHECK(
        doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, GPUNETIO_QP_ALLOW_REMOTE_READ));
    DOCA_CHECK(
        doca_verbs_qp_attr_set_atomic_mode(verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_8BYTES));
    DOCA_CHECK(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ah));
    DOCA_CHECK(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, dest_qp_num));
    DOCA_CHECK(doca_verbs_qp_attr_set_max_rd_atomic(verbs_qp_attr, max_rd_atomic));
    DOCA_CHECK(doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, max_dest_rd_atomic));

    *out_verbs_qp_attr = attr_uptr.release();
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_transition_qp_to_rts(nvshmemt_gpunetio_state_t *gpunetio_state,
                                         doca_verbs_qp_t *qp, doca_verbs_qp_attr_t *verbs_qp_attr) {
    DOCA_CHECK(
        doca_verbs_qp_modify(qp, verbs_qp_attr,
                             DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                 DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
                                 DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));

    DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR));

    DOCA_CHECK(doca_verbs_qp_modify(
        qp, verbs_qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | DOCA_VERBS_QP_ATTR_DEST_QP_NUM |
            DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_AH_ATTR |
            DOCA_VERBS_QP_ATTR_ATOMIC_MODE | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
            DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC));

    DOCA_CHECK(doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS));

    DOCA_CHECK(doca_verbs_qp_modify(
        qp, verbs_qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN | DOCA_VERBS_QP_ATTR_ACK_TIMEOUT |
            DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY |
            DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_connect_qps(nvshmemt_gpunetio_state_t *gpunetio_state, gpunetio_ep *ep,
                                gpunetio_device *device, int portid,
                                gpunetio_exch_info *remote_exch_info) {
    DOCA_CHECK(doca_verbs_ah_attr_set_gid(device->ah, remote_exch_info->vgid));
    DOCA_CHECK(doca_verbs_ah_attr_set_dlid(device->ah, remote_exch_info->lid));

    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    int rc = gpunetio_create_qp_attr(gpunetio_state, device->common_device.context, &verbs_qp_attr,
                                     device->ah, portid, remote_exch_info->qpn);
    if (rc) return rc;

    rc = gpunetio_transition_qp_to_rts(gpunetio_state, ep->qp->qp, verbs_qp_attr);
    doca_verbs_qp_attr_destroy(verbs_qp_attr);
    return rc;
}

static int gpunetio_connect_self_loop_qp(nvshmemt_gpunetio_state_t *gpunetio_state,
                                         gpunetio_device *device, int portid,
                                         doca_gpu_verbs_qp_hl *qp_local,
                                         doca_gpu_verbs_qp_hl *qp_local_backup) {
    ibv_port_attr port_attr;
    const union ibv_gid *gid = &device->common_device.gid_info[portid - 1].local_gid;
    doca_verbs_gid vgid;
    uint32_t dest_qp_num;
    ibv_query_port(device->common_device.context, portid, &port_attr);

    memcpy(vgid.raw, gid->raw, sizeof(union ibv_gid));

    DOCA_CHECK(doca_verbs_ah_attr_set_gid(device->ah, vgid));
    if (port_attr.link_layer == 1)
        DOCA_CHECK(doca_verbs_ah_attr_set_dlid(device->ah, port_attr.lid));

    DOCA_CHECK(doca_verbs_qp_get_qpn(qp_local_backup->qp, &dest_qp_num));
    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    int rc = gpunetio_create_qp_attr(gpunetio_state, device->common_device.context, &verbs_qp_attr,
                                     device->ah, portid, dest_qp_num);
    if (rc) return rc;
    auto qp_attr_guard = make_scope_guard([&]() { doca_verbs_qp_attr_destroy(verbs_qp_attr); });

    DOCA_CHECK(doca_verbs_qp_get_qpn(qp_local->qp, &dest_qp_num));
    doca_verbs_qp_attr_t *verbs_qp_attr_backup = nullptr;
    rc = gpunetio_create_qp_attr(gpunetio_state, device->common_device.context,
                                 &verbs_qp_attr_backup, device->ah, portid, dest_qp_num);
    if (rc) return rc;
    auto qp_attr_backup_guard =
        make_scope_guard([&]() { doca_verbs_qp_attr_destroy(verbs_qp_attr_backup); });

    rc = gpunetio_transition_qp_to_rts(gpunetio_state, qp_local->qp, verbs_qp_attr);
    if (rc) return rc;

    rc = gpunetio_transition_qp_to_rts(gpunetio_state, qp_local_backup->qp, verbs_qp_attr_backup);
    return rc;
}

static int gpunetio_create_qp(nvshmem_transport_t t, nvshmemt_gpunetio_state_t *gpunetio_state,
                              doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, gpunetio_ep **ep_ptr,
                              gpunetio_device *device, int portid, uint32_t qp_idx) {
    int status = 0;
    gpunetio_ep *ep = new gpunetio_ep();
    auto guard = make_scope_guard([&]() { gpunetio_destroy_ep(ep, gpunetio_state); });

    status = doca_gpu_verbs_create_qp_hl(qp_init_attr, &ep->qp);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "doca_gpu_verbs_create_qp_hl failed.\n");

    ep->portid = portid;
    ep->user_index = qp_idx;
    DOCA_CHECK(doca_verbs_qp_get_qpn(ep->qp->qp, &ep->qpn));

    status = gpunetio_create_internal_buffer(
        &ep->internal_buf, gpunetio_state, device,
        NVSHMEMI_GPUNETIO_IBUF_SLOT_SIZE *
            (gpunetio_state->num_fetch_slots_per_rc + GPUNETIO_IBUF_RESERVED_SLOTS));
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "gpunetio_create_internal_buffer failed.\n");

    *ep_ptr = ep;
    TRACE(gpunetio_state->log_level, "Created QP: qp_idx=%d, qpn=%d", qp_idx, ep->qpn);

    if (gpunetio_qp_requires_cpu_proxy(ep->qp)) {
        gpunetio_activate_progress_function(t);
    }

    guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

// Phase 1: One-time global setup
static int gpunetio_connect_global_setup(nvshmemt_gpunetio_state_t *gpunetio_state,
                                         int num_selected_devs, int *selected_dev_ids) {
    int status = 0;

    status = gpunetio_get_cuda_device_id(gpunetio_state, &gpunetio_state->cached_gpu_device_id);
    if (status) {
        return status;
    }

    // Populate selected device IDs if not already done
    if (gpunetio_state->selected_dev_ids.empty()) {
        gpunetio_state->selected_dev_ids.resize(num_selected_devs);
    }

    // Validate device IDs
    for (int i = 0; i < num_selected_devs; i++) {
        if (selected_dev_ids[i] < 0 ||
            selected_dev_ids[i] >= static_cast<int>(gpunetio_state->dev_ids.size())) {
            NVSHMEMI_ERROR_PRINT("Invalid device ID %d.\n", selected_dev_ids[i]);
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        gpunetio_state->selected_dev_ids[i] = gpunetio_state->dev_ids[selected_dev_ids[i]];
    }

    return NVSHMEMX_SUCCESS;
}

// Progress
static bool gpunetio_qp_requires_cpu_proxy(doca_gpu_verbs_qp_hl *qp) {
    return qp->qp_gverbs->qp_cpu->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
}

static void gpunetio_activate_progress_function(nvshmem_transport_t t) {
    t->host_ops.progress = nvshmemt_gpunetio_progress;
    t->no_proxy = false;
}

int nvshmemt_gpunetio_progress(nvshmem_transport_t t) {
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    int n_pes = t->n_pes;

    // Iterate over all devices and their EPs and progress QPs
    for (int dev_idx : gpunetio_state->selected_dev_ids) {
        gpunetio_device *device = &gpunetio_state->devices[dev_idx];
        {
            std::lock_guard<std::mutex> lk(*device->rc_eps_mtx);
            int num_eps = device->num_eps_per_pe * n_pes;
            for (int j = 0; j < num_eps; j++) {
                gpunetio_ep *ep = device->rc_eps[j];
                if (ep == nullptr) continue;

                int status = doca_gpu_verbs_cpu_proxy_progress(ep->qp->qp_gverbs, nullptr);
                if (status) {
                    NVSHMEMI_WARN_PRINT("doca_gpu_verbs_cpu_proxy_progress failed for ep %p \n",
                                        ep);
                }
            }
        }

        // Progress local backup QP
        if (device->qp_local_backup && device->qp_local_backup->qp_gverbs) {
            int status =
                doca_gpu_verbs_cpu_proxy_progress(device->qp_local_backup->qp_gverbs, nullptr);
            if (status) {
                NVSHMEMI_WARN_PRINT(
                    "doca_gpu_verbs_cpu_proxy_progress failed for qp_local_backup\n");
            }
        }
    }

    return NVSHMEMX_SUCCESS;
}

// Populate and copy over state to GPU
static int gpunetio_setup_gpu_state(nvshmem_transport_t t) {
    int status = 0;

    nvshmemt_gpunetio_state_t *gpunetio_state;
    gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    nvshmemi_gpunetio_device_state_t *gpunetio_device_state_h;
    gpunetio_device_state_h =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);
    nvshmemi_gpunetio_device_qp_t *qp_d = gpunetio_device_state_h->globalmem.qps;
    nvshmemi_gpunetio_device_qp_t *qp_d_temp = nullptr;

    auto qp_d_guard = make_scope_guard([&]() {
        if (qp_d && qp_d != gpunetio_device_state_h->globalmem.qps) {
            cudaError_t err = cudaFree(qp_d);
            CUDA_RUNTIME_ERROR_STRING(err);
        }
    });

    doca_gpu_dev_verbs_qp *qp_tmp;
    int num_rc_handles = 0;
    int n_devs_selected = static_cast<int>(gpunetio_state->selected_dev_ids.size());

    assert(gpunetio_device_state_h != nullptr);

    // Calculate total RC handle count across all devices
    for (int dev_idx : gpunetio_state->selected_dev_ids) {
        gpunetio_device *device = &gpunetio_state->devices[dev_idx];
        num_rc_handles += device->num_eps_per_pe * t->n_pes;
    }
    INFO(gpunetio_state->log_level, "num_rc_handles: %d (last_num_rcs: %d)", num_rc_handles,
         gpunetio_state->last_num_rcs);

    if (num_rc_handles <= 0) {
        NVSHMEMI_WARN_PRINT("num_rc_handles is 0\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    // Resize host-side QP array
    gpunetio_state->qp_h.resize(num_rc_handles);

    // Reallocate device-side QP array if it already exists
    if (qp_d != nullptr) {
        CUDA_RUNTIME_CHECK_RET(
            cudaMalloc(&qp_d_temp, num_rc_handles * sizeof(nvshmemi_gpunetio_device_qp_t)),
            NVSHMEMX_ERROR_OUT_OF_MEMORY);
        auto qp_d_temp_guard = make_scope_guard([&]() {
            if (qp_d_temp) {
                cudaError_t err = cudaFree(qp_d_temp);
                CUDA_RUNTIME_ERROR_STRING(err);
            }
        });
        CUDA_RUNTIME_CHECK_RET(
            cudaMemcpyAsync(qp_d_temp, qp_d,
                            gpunetio_state->last_num_rcs * sizeof(nvshmemi_gpunetio_device_qp_t),
                            cudaMemcpyDeviceToDevice, gpunetio_state->my_stream),
            NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(
            cudaMemcpyAsync(gpunetio_state->qp_h.data(), qp_d,
                            gpunetio_state->last_num_rcs * sizeof(nvshmemi_gpunetio_device_qp_t),
                            cudaMemcpyDeviceToHost, gpunetio_state->my_stream),
            NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                               NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(cudaFree(qp_d), NVSHMEMX_ERROR_INTERNAL);
        qp_d = qp_d_temp;
        qp_d_temp = nullptr;
    }

    // Populate QP host array with data from all devices and all eps
    for (size_t i = 0; i < gpunetio_state->selected_dev_ids.size(); ++i) {
        int dev_idx = gpunetio_state->selected_dev_ids[i];
        gpunetio_device *device = &gpunetio_state->devices[dev_idx];
        int device_num_eps_per_pe = device->num_eps_per_pe;

        for (int j = 0; j < device_num_eps_per_pe; ++j) {
            for (int k = 0; k < t->n_pes; ++k) {
                int dst_pe = (j * t->n_pes + 1 + t->my_pe + k) % t->n_pes;
                int device_ep_index = j * t->n_pes + dst_pe;
                assert(device->rc_eps[device_ep_index] != nullptr);

                // Loopback has only one ep per device
                if (dst_pe == t->my_pe && j > 0) {
                    continue;
                }

                // Copy to the right index in the qp_h array holding *all* QPs for all devices
                int global_ep_index =
                    static_cast<int>(i) * device_num_eps_per_pe * t->n_pes + device_ep_index;
                assert(global_ep_index < num_rc_handles);

                if (global_ep_index < gpunetio_state->last_num_rcs) {
                    continue;
                }

                if (doca_gpu_verbs_get_qp_dev(device->rc_eps[device_ep_index]->qp->qp_gverbs,
                                              &qp_tmp) != DOCA_SUCCESS) {
                    status = NVSHMEMX_ERROR_INTERNAL;
                    return status;
                }
                CUDA_RUNTIME_CHECK_RET(
                    cudaMemcpyAsync(&(gpunetio_state->qp_h[global_ep_index].qp), qp_tmp,
                                    sizeof(doca_gpu_dev_verbs_qp), cudaMemcpyDefault,
                                    gpunetio_state->my_stream),
                    NVSHMEMX_ERROR_OUT_OF_MEMORY);
                CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                                       NVSHMEMX_ERROR_OUT_OF_MEMORY);

                TRACE(
                    gpunetio_state->log_level,
                    "Exported handle %d for PE %d for device %d at global index %d, pointer is %p",
                    device_ep_index, dst_pe, dev_idx, global_ep_index,
                    &(gpunetio_state->qp_h[global_ep_index].qp));

                gpunetio_state->qp_h[global_ep_index].ibuf.buf =
                    device->rc_eps[device_ep_index]->internal_buf.mem_object->aligned.gpu_ptr;
                gpunetio_state->qp_h[global_ep_index].ibuf.nslots =
                    gpunetio_state->num_fetch_slots_per_rc;
                gpunetio_state->qp_h[global_ep_index].ibuf.lkey =
                    htobe32(device->rc_eps[device_ep_index]->internal_buf.mem_handle->lkey);
                gpunetio_state->qp_h[global_ep_index].ibuf.rkey =
                    htobe32(device->rc_eps[device_ep_index]->internal_buf.mem_handle->rkey);
                gpunetio_state->qp_h[global_ep_index].dev_idx = static_cast<uint32_t>(i);
            }
        }
    }

    // Allocate device QP array if first time
    if (qp_d == nullptr) {
        CUDA_RUNTIME_CHECK_RET(
            cudaMalloc(&qp_d, num_rc_handles * sizeof(nvshmemi_gpunetio_device_qp_t)),
            NVSHMEMX_ERROR_OUT_OF_MEMORY);
    }
    // Copy full QP array to GPU
    CUDA_RUNTIME_CHECK_RET(cudaMemcpyAsync(qp_d, gpunetio_state->qp_h.data(),
                                           num_rc_handles * sizeof(nvshmemi_gpunetio_device_qp_t),
                                           cudaMemcpyDefault, gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_OUT_OF_MEMORY);
    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_OUT_OF_MEMORY);

    gpunetio_device_state_h->globalmem.qps = qp_d;
    gpunetio_device_state_h->may_skip_cst = gpunetio_state->skip_cst;
    gpunetio_device_state_h->num_devices_initialized = n_devs_selected;
    gpunetio_device_state_h->num_rc_per_pe = num_rc_handles / n_devs_selected / t->n_pes;
    gpunetio_device_state_h->num_default_rc_per_pe =
        gpunetio_state->options->GPUNETIO_NUM_RC_PER_PE;
    gpunetio_device_state_h->log2_cumem_granularity = t->log2_cumem_granularity;
    gpunetio_device_state_h->num_requests_in_batch = gpunetio_state->num_requests_in_batch;

    INFO(gpunetio_state->log_level,
         "num_rc_per_pe %d num_rc_handles %d n_devs_selected %d n_pes %d",
         gpunetio_device_state_h->num_rc_per_pe, num_rc_handles, n_devs_selected, t->n_pes);

    // QP group switches for load balancing (allocate only once)
    if (gpunetio_device_state_h->globalmem.qp_group_switches == nullptr) {
        int default_num_rc_handles =
            gpunetio_state->options->GPUNETIO_NUM_RC_PER_PE * n_devs_selected * t->n_pes;
        if (num_rc_handles == default_num_rc_handles) {
            int num_qp_groups = std::max(num_rc_handles / n_devs_selected / t->n_pes, 2);
            uint8_t *qp_group_switches_d;
            CUDA_RUNTIME_CHECK_RET(cudaMalloc(reinterpret_cast<void **>(&qp_group_switches_d),
                                              num_qp_groups * sizeof(uint8_t)),
                                   NVSHMEMX_ERROR_OUT_OF_MEMORY);
            CUDA_RUNTIME_CHECK_RET(
                cudaMemsetAsync(qp_group_switches_d, 0, num_qp_groups * sizeof(uint8_t),
                                gpunetio_state->my_stream),
                NVSHMEMX_ERROR_INTERNAL);
            gpunetio_device_state_h->globalmem.qp_group_switches = qp_group_switches_d;
        }
    }

    gpunetio_state->last_num_rcs = num_rc_handles;

    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_INTERNAL);

    qp_d_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

// Phase 3: Per-device endpoint setup
static int gpunetio_connect_device_endpoints(nvshmemt_gpunetio_state_t *gpunetio_state,
                                             gpunetio_device *device, int portid,
                                             nvshmem_transport_t t, int num_rc_eps_per_pe) {
    int status = 0;
    int mype = t->my_pe;
    int n_pes = t->n_pes;
    int new_num_rc_eps = num_rc_eps_per_pe * n_pes;
    doca_gpu_verbs_qp_init_attr_hl qp_init_attr;

    // exch_info structures
    std::vector<gpunetio_exch_info> local_exch_info(new_num_rc_eps);
    std::vector<gpunetio_exch_info> peer_exch_info(new_num_rc_eps);

    // get first index of additional RC endpoints
    int rc_first_index = device->num_eps_per_pe * n_pes;

    if (new_num_rc_eps <= 0) {
        return NVSHMEMX_SUCCESS;
    }

    {
        std::lock_guard<std::mutex> lk(*device->rc_eps_mtx);
        try {
            device->rc_eps.resize(device->rc_eps.size() + new_num_rc_eps, nullptr);
        } catch (const std::bad_alloc &) {
            status = NVSHMEMX_ERROR_OUT_OF_MEMORY;
            NVSHMEMI_ERROR_PRINT("allocation of rc_eps failed.\n");
            return status;
        }
    }

    auto ep_cleanup_guard = make_scope_guard([&]() {
        // Reset EP vector to original size on failure
        std::lock_guard<std::mutex> lk(*device->rc_eps_mtx);
        for (int i = rc_first_index; i < rc_first_index + new_num_rc_eps; ++i) {
            if (device->rc_eps[i]) {
                gpunetio_destroy_ep(device->rc_eps[i], gpunetio_state);
                device->rc_eps[i] = nullptr;
            }
        }
        device->rc_eps.resize(rc_first_index);

        // If we created the local backup QP, destroy it
        if (rc_first_index == 0 && device->qp_local_backup) {
            doca_gpu_verbs_destroy_qp_hl(device->qp_local_backup);
            device->qp_local_backup = nullptr;
        }
    });

    if (!device->common_device.pd) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "device->common_device.pd is NULL for device\n");
    }
    if (!device->common_device.context) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "device->common_device.context is NULL for device\n");
    }

    if (!gpunetio_state->gpu_device) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "gpunetio_state->gpu_device is NULL\n");
    }

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.gpu_dev = gpunetio_state->gpu_device;
    qp_init_attr.ibpd = device->common_device.pd;
    qp_init_attr.sq_nwqe = gpunetio_state->qp_depth;
    qp_init_attr.nic_handler = device->nic_handler_request;
    qp_init_attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
    qp_init_attr.cq_collapsed = true;
    qp_init_attr.net_dev = device->net_dev;

    if (gpunetio_state->options->GPUNETIO_ENABLE_ORDERING_SEMANTIC) {
        INFO(gpunetio_state->log_level, "Ordering semantic for DDP will be enabled via GPUNetIO\n");
        qp_init_attr.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL;
    }

    INFO(gpunetio_state->log_level, "Creating %d RC QPs", num_rc_eps_per_pe);
    for (int i = 0; i < num_rc_eps_per_pe; i++) {
        for (int j = 0; j < n_pes; j++) {
            int dst_pe = (i * n_pes + 1 + mype + j) % n_pes;
            int mapped_i = rc_first_index + i * n_pes + dst_pe;
            int local_mapped_i = i + num_rc_eps_per_pe * dst_pe;

            // Skip self-loop QP
            if (dst_pe == mype) continue;

            TRACE(gpunetio_state->log_level, "dst_pe: %d, mapped_i: %d, local_mapped_i: %d", dst_pe,
                  mapped_i, local_mapped_i);

            status = gpunetio_create_qp(t, gpunetio_state, &qp_init_attr, &device->rc_eps[mapped_i],
                                        device, portid, mapped_i);

            if (status != NVSHMEMX_SUCCESS) {
                if (gpunetio_state->options->GPUNETIO_ENABLE_ORDERING_SEMANTIC) {
                    NVSHMEMI_ERROR_PRINT(
                        "gpunetio_create_qp with ordering semantic enabled failed, please retry "
                        "with "
                        "NVSHMEM_GPUNETIO_ENABLE_ORDERING_SEMANTIC=0\n");
                }
                NVSHMEMI_ERROR_PRINT("gpunetio_create_qp failed on RC #%d.", mapped_i);
                return NVSHMEMX_ERROR_INTERNAL;
            }

            status = gpunetio_fill_exch_info(&local_exch_info[local_mapped_i],
                                             device->rc_eps[mapped_i], device);
            NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                                  "gpunetio_fill_exch_info failed on RC #%d.", mapped_i);
        }
    }

    status =
        t->boot_handle->alltoall(local_exch_info.data(), peer_exch_info.data(),
                                 sizeof(gpunetio_exch_info) * num_rc_eps_per_pe, t->boot_handle);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "alltoall of exch_info failed.\n");

    for (int i = 0; i < num_rc_eps_per_pe; ++i) {
        for (int j = 0; j < n_pes; ++j) {
            int ep_index = rc_first_index + i * n_pes + j;
            int peer_handle_index = num_rc_eps_per_pe * j + i;
            // No loopback to self
            if (j == mype) {
                continue;
            }
            TRACE(gpunetio_state->log_level,
                  "Resetting and initializing RC #%d with qp_idx #%d QPN: %d", ep_index,
                  device->rc_eps[ep_index]->user_index, device->rc_eps[ep_index]->qpn);
            TRACE(gpunetio_state->log_level, "local QPN: %d, remote handle QPN: %d",
                  device->rc_eps[ep_index]->qpn, peer_exch_info[peer_handle_index].qpn);

            status = gpunetio_connect_qps(gpunetio_state, device->rc_eps[ep_index], device, portid,
                                          &peer_exch_info[peer_handle_index]);
            NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                                  "gpunetio_connect_qps failed on RC #%d.", ep_index);

            TRACE(gpunetio_state->log_level, "DONE RC #%d", ep_index);
        }
    }

    // We have a single loopback QP for each device (at the same slot the regular loop skips)
    int mype_ep_index = rc_first_index + mype;
    if (device->rc_eps[mype_ep_index] == nullptr) {
        status = gpunetio_create_qp(t, gpunetio_state, &qp_init_attr,
                                    &device->rc_eps[mype_ep_index], device, portid, mype_ep_index);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "gpunetio_create_qp failed on loopback QP.\n");

        // Dummy backup QP to have matching local QP
        status = doca_gpu_verbs_create_qp_hl(&qp_init_attr, &device->qp_local_backup);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "doca_gpu_verbs_create_qp_hl failed.");
        if (gpunetio_qp_requires_cpu_proxy(device->qp_local_backup)) {
            gpunetio_activate_progress_function(t);
        }

        // Connect self-loop QP RC to backup local QP
        status = gpunetio_connect_self_loop_qp(gpunetio_state, device, portid,
                                               device->rc_eps[mype_ep_index]->qp,
                                               device->qp_local_backup);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "gpunetio_connect_self_loop_qp failed on loopback QP.\n");
    }

    {
        std::lock_guard<std::mutex> lk(*device->rc_eps_mtx);
        device->num_eps_per_pe += num_rc_eps_per_pe;
    }
    gpunetio_state->cur_qp_index += new_num_rc_eps;
    // Set global state to skip_cst as soon as one device requires CST
    gpunetio_state->skip_cst &=
        (!gpunetio_cst_is_required(gpunetio_state, device, gpunetio_state->cached_gpu_device_id));

    ep_cleanup_guard.dismiss();
    return status;
}

// Parse and cache
static int gpunetio_parse_nic_handler_request(doca_gpu_dev_verbs_nic_handler *out_loc,
                                              const char *str) {
    std::string req = str;
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());
    std::for_each(req.begin(), req.end(), [](char &c) { c = ::tolower(c); });

    if (req == "auto") {
        *out_loc = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    } else if (req == "gpu") {
        *out_loc = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
    } else if (req == "gpu_sm_bf") {
        *out_loc = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF;
    } else if (req == "cpu") {
        *out_loc = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    } else {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    return NVSHMEMX_SUCCESS;
}

static void gpunetio_initialize_cache_state(nvshmemt_gpunetio_state_t *gpunetio_state) {
    gpunetio_state->connect_endpoints_first_call = true;
    gpunetio_state->last_device_index = 0;
    gpunetio_state->cur_qp_index = 0;
    gpunetio_state->last_num_rcs = 0;
    gpunetio_state->qp_h.clear();
}

static int gpunetio_connect_qps_only(nvshmemt_gpunetio_state_t *gpunetio_state,
                                     nvshmem_transport_t t, int *out_qp_indices, int num_qps) {
    assert(out_qp_indices != nullptr);

    for (int i = 0; i < num_qps; i++) {
        out_qp_indices[i] = gpunetio_state->cur_qp_index;
        int n_devs_selected = static_cast<int>(gpunetio_state->selected_dev_ids.size());
        int selected_dev_idx = gpunetio_state->last_device_index % n_devs_selected;
        int dev_idx = gpunetio_state->selected_dev_ids[selected_dev_idx];
        int portid = gpunetio_state->port_ids[selected_dev_idx];
        gpunetio_device *device = &gpunetio_state->devices[dev_idx];

        int status = gpunetio_connect_device_endpoints(gpunetio_state, device, portid, t, 1);
        if (status) {
            NVSHMEMI_ERROR_PRINT("gpunetio_connect_device_endpoints failed on QP #%d.\n", i);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        gpunetio_state->last_device_index++;
    }

    return gpunetio_setup_gpu_state(t);
}

// Public transport API functions
int nvshmemt_gpunetio_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                        int num_selected_devs, int *out_qp_indices, int num_qps) {
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    int status = 0;
    int init_dev_cnt = 0;

    if (!gpunetio_state->connect_endpoints_first_call) {
        // Short path for subsequent calls (QP-specific API)
        return gpunetio_connect_qps_only(gpunetio_state, t, out_qp_indices, num_qps);
    }

    // Phase 1: Global setup (only on first call)
    status = gpunetio_connect_global_setup(gpunetio_state, num_selected_devs, selected_dev_ids);
    if (status) {
        NVSHMEMI_ERROR_PRINT("gpunetio_connect_global_setup failed.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    // Phase 2-3: Per-device processing (cached per device)
    for (int i = 0; i < num_selected_devs; i++) {
        int dev_idx = gpunetio_state->dev_ids[selected_dev_ids[i]];
        gpunetio_device *device = &gpunetio_state->devices[dev_idx];
        int portid = gpunetio_state->port_ids[selected_dev_ids[i]];

        status = gpunetio_create_ah(gpunetio_state, device, portid);
        if (status) {
            NVSHMEMI_ERROR_PRINT("gpunetio_create_ah failed.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }

        status = gpunetio_connect_device_endpoints(gpunetio_state, device, portid, t,
                                                   gpunetio_state->options->GPUNETIO_NUM_RC_PER_PE);
        if (status) return status;

        init_dev_cnt++;
    }

    // Multiple devices break our CST optimizations
    if (init_dev_cnt > 1) {
        gpunetio_state->skip_cst = false;
    }

    // Phase 4: GPU setup (only once)
    gpunetio_setup_gpu_state(t);

    // Set device flags
    if (init_dev_cnt < num_selected_devs) {
        NVSHMEMI_WARN_PRINT("Failed to initialize all selected devices. Perf may be limited.\n");
    }

    gpunetio_state->connect_endpoints_first_call = false;

    return status;
}

int nvshmemt_gpunetio_can_reach_peer(int *access, nvshmem_transport_pe_info *peer_info,
                                     nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
              NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;

    return status;
}

int nvshmemt_gpunetio_show_info(nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("gpunetio show info not implemented\n");
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_finalize(nvshmem_transport_t transport) {
    assert(transport != nullptr);
    auto transport_guard = make_scope_guard([&]() { free(transport); });
    nvshmemt_gpunetio_state_t *gpunetio_state =
        static_cast<nvshmemt_gpunetio_state_t *>(transport->state);
    nvshmemi_gpunetio_device_state_t *gpunetio_device_state_h;

    int status = 0;
    int ret = 0;

    if (!gpunetio_state) {
        return status;
    }

    gpunetio_state->device_lkeys.clear();
    gpunetio_state->device_rkeys.clear();

    if (gpunetio_state->device_lkeys_d) {
        cudaError_t err = cudaFree(gpunetio_state->device_lkeys_d);
        CUDA_RUNTIME_ERROR_STRING(err);
        gpunetio_state->device_lkeys_d = nullptr;
    }

    if (gpunetio_state->device_rkeys_d) {
        cudaError_t err = cudaFree(gpunetio_state->device_rkeys_d);
        CUDA_RUNTIME_ERROR_STRING(err);
        gpunetio_state->device_rkeys_d = nullptr;
    }

    gpunetio_device_state_h =
        static_cast<nvshmemi_gpunetio_device_state_t *>(transport->type_specific_shared_state);
    if (gpunetio_device_state_h) {
        if (gpunetio_device_state_h->globalmem.qps) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.qps);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.qps = nullptr;
        }
        if (gpunetio_device_state_h->globalmem.lkeys) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.lkeys);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.lkeys = nullptr;
        }
        if (gpunetio_device_state_h->globalmem.rkeys) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.rkeys);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.rkeys = nullptr;
        }
        if (gpunetio_device_state_h->globalmem.qp_group_switches) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.qp_group_switches);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.qp_group_switches = nullptr;
        }
    }

    gpunetio_state->qp_h.clear();

    for (int dev_id : gpunetio_state->selected_dev_ids) {
        gpunetio_device *device = &gpunetio_state->devices[dev_id];

        if (device->ah) {
            ret = doca_verbs_ah_attr_destroy(device->ah);
            if (ret) {
                NVSHMEMI_WARN_PRINT("doca_verbs_ah_attr_destroy failed for device %d Err: %d:%s.\n",
                                    dev_id, errno, strerror(errno));
                if (!status) status = ret;
            }
            device->ah = nullptr;
        }

        {
            std::lock_guard<std::mutex> lk(*device->rc_eps_mtx);
            for (auto *ep : device->rc_eps) {
                ret = gpunetio_destroy_ep(ep, gpunetio_state);
                if (ret) {
                    NVSHMEMI_WARN_PRINT("gpunetio_destroy_ep failed for device %d\n", dev_id);
                    if (!status) status = ret;
                }
            }
            device->rc_eps.clear();
        }

        // Clean up backup QP
        ret = doca_gpu_verbs_destroy_qp_hl(device->qp_local_backup);
        if (ret) {
            NVSHMEMI_WARN_PRINT(
                "doca_gpu_verbs_destroy_qp_hl failed for device %d qp_local_backup \n", dev_id);
            if (!status) status = ret;
        }
    }

    // Free all devices, not just ones we used.
    for (size_t i = 0; i < gpunetio_state->dev_ids.size(); ++i) {
        gpunetio_device *device = &gpunetio_state->devices[gpunetio_state->dev_ids[i]];
        if (device->common_device.pd) {
            ret = gpunetio_state->ftable.dealloc_pd(device->common_device.pd);
            if (ret) {
                INFO(gpunetio_state->log_level,
                     "ibv_dealloc_pd failed for device %zu Err: %d:%s.\n", i, errno,
                     strerror(errno));
                if (!status) status = ret;
            }
        }
        if (device->common_device.context) {
            ret = gpunetio_state->ftable.close_device(device->common_device.context);
            if (ret) {
                NVSHMEMI_WARN_PRINT("ibv_close_device failed for device %zu Err: %d:%s.\n", i,
                                    errno, strerror(errno));
                if (!status) status = ret;
            }
        }
    }

    nvshmemt_ibv_ftable_fini(&gpunetio_state->ibv_handle);

#ifdef NVSHMEM_USE_MLX5DV
    if (gpunetio_state->mlx5dv_handle) {
        nvshmemt_mlx5dv_ftable_fini(&gpunetio_state->mlx5dv_handle);
    }
#endif

    if (gpunetio_state->my_stream) {
        cudaError_t err = cudaStreamDestroy(gpunetio_state->my_stream);
        CUDA_RUNTIME_ERROR_STRING(err);
        gpunetio_state->my_stream = nullptr;
    }

    ret = doca_gpu_destroy(gpunetio_state->gpu_device);
    if (ret) {
        NVSHMEMI_WARN_PRINT("doca_gpu_destroy failed for device %p \n", gpunetio_state->gpu_device);
        if (!status) status = ret;
    }

    delete gpunetio_state;

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

    return status;
}

int nvshmemt_gpunetio_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_stride,
                                                    nvshmem_mem_handle_t *mem_handles,
                                                    uint64_t heap_offset, size_t size) {
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    int n_pes = t->n_pes;

    size_t num_rkeys;

    nvshmemi_gpunetio_device_state_t *gpunetio_device_state;

    gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);
    assert(gpunetio_device_state != nullptr);

    auto error_guard = make_scope_guard([&]() {
        if (gpunetio_state->device_rkeys_d) {
            cudaError_t err = cudaFree(gpunetio_state->device_rkeys_d);
            CUDA_RUNTIME_ERROR_STRING(err);
        }
        gpunetio_state->device_rkeys.clear();
    });

    static_assert(sizeof(nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE,
                  "static_assert(sizeof(T) <= NVSHMEM_MEM_HANDLE_SIZE) failed");

    size_t num_elements;
    // size must be divisible by cumem_granularity, which is a power of 2.
    assert((size & ((1ULL << t->log2_cumem_granularity) - 1)) == 0);

    num_elements = size >> t->log2_cumem_granularity;

    // With user buffers being mmaped at the end of the heap, incrementally adding to lkeys vector
    // won't be sufficient as buffers from end of heap could be registered. So we resize
    // gpunetio_device_rkeys vector to number of chunks  * n_pes * n_devs_selected and populate
    // entries as they are added.
    size_t chunk_idx = heap_offset >> t->log2_cumem_granularity;
    size_t num_chunks =
        (heap_offset + size + t->log2_cumem_granularity - 1) >> t->log2_cumem_granularity;
    // assuming all PEs have same num_devs
    int num_devs = reinterpret_cast<gpunetio_mem_handle *>(&mem_handles[t->index])->num_devs;
    if (gpunetio_state->device_rkeys.size() < num_chunks * n_pes * num_devs) {
        gpunetio_state->device_rkeys.resize(num_chunks * n_pes * num_devs);
    }

    for (; num_elements > 0; --num_elements) {
        for (int i = 0; i < n_pes; ++i) {
            // sizeof(gpunetio_mem_handle) <= sizeof(nvshmem_mem_handle_t)
            // So, we calculate the pointer with nvshmem_mem_handle_t and convert to
            // gpunetio_mem_handle later.
            auto *gmhandle = reinterpret_cast<gpunetio_mem_handle *>(
                &mem_handles[i * transport_stride + t->index]);
            assert((num_devs == gmhandle->num_devs) &&
                   "Currently, we only support same number of "
                   "devices per PE");
            for (int j = 0; j < gmhandle->num_devs; j++) {
                nvshmemt_ib_common_mem_handle *handle = &gmhandle->dev_mem_handles[j];
                nvshmemi_gpunetio_device_key_t device_key;
                device_key.key = htobe32(handle->rkey);
                device_key.next_addr = heap_offset + size;

                gpunetio_state->device_rkeys.at(
                    ((chunk_idx + num_elements - 1) * n_pes * num_devs) + (i * num_devs) + j) =
                    device_key;
            }
        }
    }

    if (gpunetio_state->device_rkeys_d) {
        CUDA_RUNTIME_CHECK_RET(cudaFree(gpunetio_state->device_rkeys_d), NVSHMEMX_ERROR_INTERNAL);
        gpunetio_state->device_rkeys_d = nullptr;
    }

    num_rkeys = gpunetio_state->device_rkeys.size();

    // For cache optimization, put rkeys in constant memory first.
    std::copy_n(gpunetio_state->device_rkeys.begin(),
                std::min(num_rkeys, static_cast<size_t>(NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS)),
                gpunetio_device_state->constmem.rkeys);

    // Put the rest that don't fit in constant memory in global memory
    if (num_rkeys > NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS) {
        size_t rkeys_array_size = sizeof(nvshmemi_gpunetio_device_key_t) *
                                  (num_rkeys - NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS);

        nvshmemi_gpunetio_device_key_t *data_ptr =
            &gpunetio_state->device_rkeys.data()[NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS];

        CUDA_RUNTIME_CHECK_RET(cudaMalloc(&gpunetio_state->device_rkeys_d, rkeys_array_size),
                               NVSHMEMX_ERROR_OUT_OF_MEMORY);

        CUDA_RUNTIME_CHECK_RET(
            cudaMemcpyAsync(gpunetio_state->device_rkeys_d, (const void *)data_ptr,
                            rkeys_array_size, cudaMemcpyHostToDevice, gpunetio_state->my_stream),
            NVSHMEMX_ERROR_INTERNAL);

        CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                               NVSHMEMX_ERROR_INTERNAL);
    }

    gpunetio_device_state->globalmem.rkeys =
        static_cast<nvshmemi_gpunetio_device_key_t *>(gpunetio_state->device_rkeys_d);

    error_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

// Memory handle management start
int nvshmemt_gpunetio_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                     nvshmem_transport_t t, bool local_only) {
    int status = 0;
    nvshmem_transport_t transport = t;
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(transport->state);

    __be32 device_lkey;
    gpunetio_mem_handle *handle;

    nvshmemi_gpunetio_device_local_only_mhandle_t *device_mhandle_d = nullptr;
    bool did_emplace = false;

    nvshmemi_gpunetio_device_state_t *gpunetio_device_state;
    gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);
    if (gpunetio_device_state == nullptr) {
        NVSHMEMI_WARN_PRINT("gpunetio_device_state is NULL\n");
    }

    int n_devs_selected = static_cast<int>(gpunetio_state->selected_dev_ids.size());

    memset(mem_handle, 0, sizeof(*mem_handle));
    handle = reinterpret_cast<gpunetio_mem_handle *>(mem_handle);
    handle->num_devs = n_devs_selected;

    auto error_guard = make_scope_guard([&]() {
        if (device_mhandle_d) {
            cudaError_t err = cudaFree(device_mhandle_d);
            CUDA_RUNTIME_ERROR_STRING(err);
        }
        if (did_emplace) {
            if (local_only) {
                gpunetio_state->device_local_only_mhandles.pop_back();
            } else {
                gpunetio_state->device_lkeys.clear();
            }
        }
        for (int i = 0; i < n_devs_selected; ++i) {
            nvshmemt_ib_common_release_mem_handle(
                &gpunetio_state->ftable,
                reinterpret_cast<nvshmem_mem_handle_t *>(&handle->dev_mem_handles[i]),
                gpunetio_state->log_level);
        }
    });

    // In cases where same physical memory has been mapped to multiple VAs (say VA1 and VA2)
    // e.g., user buffers mmapped into symmetric heap. Using VA2 for buffer registration
    // and gdrcopy.pin_buffer is unsupported in RM/nv-p2p (Ref nvbug 507809).
    // We need to use VA1 (first mapped address mapped) as a work around.
    // For an mmapped buffer, buf is VA2, we track the VA2->VA1 mapping during mmap call
    // alias_va_ptr will hold the VA1 address, if applicable
    void *alias_va_ptr = nullptr;
    if (transport->alias_va_map != nullptr && transport->alias_va_map->count(buf)) {
        INFO(gpunetio_state->log_level, "DOCA: alias va found for buf: %p, alias va: %p", buf,
             transport->alias_va_map->operator[](buf));
        alias_va_ptr = transport->alias_va_map->operator[](buf);
    }

    for (int i = 0; i < n_devs_selected; ++i) {
        gpunetio_device *device = &gpunetio_state->devices[gpunetio_state->selected_dev_ids[i]];
        auto *dev_handle = reinterpret_cast<nvshmem_mem_handle_t *>(&handle->dev_mem_handles[i]);

        INFO(gpunetio_state->log_level, "[%d] DOCA: device used %s, data_direct support: %d",
             transport->my_pe, device->common_device.dev->name, device->common_device.data_direct);

        status = nvshmemt_ib_common_reg_mem_handle(
            &gpunetio_state->ftable, &gpunetio_state->mlx5dv_ftable, device->common_device.pd,
            dev_handle, buf, length, local_only, gpunetio_state->dmabuf_support_for_data_buffers,
            gpunetio_state->cuda_syms, gpunetio_state->log_level,
            gpunetio_state->options->IB_ENABLE_RELAXED_ORDERING, device->common_device.data_direct,
            alias_va_ptr);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "Unable to register memory handle.\n");
    }

    if (local_only) {
        gpunetio_device_local_only_mhandle_cache device_mhandle_cache;
        nvshmemi_gpunetio_device_local_only_mhandle_t *device_mhandle_h =
            &device_mhandle_cache.mhandle;
        nvshmemi_init_gpunetio_device_local_only_memhandle((*device_mhandle_h));

        void *mhandle_gpu_ptr;

        cudaPointerAttributes buf_attributes;

        CUDA_RUNTIME_CHECK_RET(cudaPointerGetAttributes(&buf_attributes, buf),
                               NVSHMEMX_ERROR_INTERNAL);

        CUDA_RUNTIME_CHECK_RET(
            cudaMalloc(reinterpret_cast<void **>(&device_mhandle_d), sizeof(*device_mhandle_d)),
            NVSHMEMX_ERROR_OUT_OF_MEMORY);

        device_mhandle_h->start = reinterpret_cast<uint64_t>(buf);
        device_mhandle_h->end = reinterpret_cast<uint64_t>(buf) + length - 1;
        device_mhandle_h->is_sysmem_scope = (buf_attributes.type != cudaMemoryTypeDevice);
        device_mhandle_h->next = nullptr;
        for (int i = 0; i < n_devs_selected; ++i) {
            device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
            device_mhandle_h->lkeys[i] = device_lkey;
        }

        CUDA_RUNTIME_CHECK_RET(
            cudaMemcpyAsync(device_mhandle_d, device_mhandle_h, sizeof(*device_mhandle_d),
                            cudaMemcpyHostToDevice, gpunetio_state->my_stream),
            NVSHMEMX_ERROR_INTERNAL);

        device_mhandle_cache.dev_ptr = device_mhandle_d;

        if (gpunetio_state->device_local_only_mhandles.empty()) {
            gpunetio_device_state->globalmem.local_only_mhandle_head = device_mhandle_d;
        } else {
            gpunetio_device_local_only_mhandle_cache *last_mhandle_cache =
                &gpunetio_state->device_local_only_mhandles.back();
            mhandle_gpu_ptr = reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(last_mhandle_cache->dev_ptr) +
                offsetof(nvshmemi_gpunetio_device_local_only_mhandle_t, next));
            last_mhandle_cache->mhandle.next = device_mhandle_d;
            CUDA_RUNTIME_CHECK_RET(
                cudaMemcpyAsync(mhandle_gpu_ptr, &device_mhandle_d, sizeof(device_mhandle_d),
                                cudaMemcpyHostToDevice, gpunetio_state->my_stream),
                NVSHMEMX_ERROR_INTERNAL);
        }

        gpunetio_state->device_local_only_mhandles.emplace_back(device_mhandle_cache);
        did_emplace = true;
    } else {
        size_t num_lkeys;
        size_t num_elements;
        // length must be divisible by cumem_granularity, which is a power of 2.
        assert((length & ((1ULL << transport->log2_cumem_granularity) - 1)) == 0);

        num_elements = length >> transport->log2_cumem_granularity;

        // With user buffers being mmaped at the of the heap, incremently adding to lkeys vector
        // won't be sufficient as buffers from end of heap could be registered. So we resize
        // gpunetio_device_lkeys vector to number of chunks  * n_devs_selected and populate entries
        // as they are added.
        size_t chunk_idx =
            (reinterpret_cast<char *>(buf) - reinterpret_cast<char *>(t->heap_base)) >>
            t->log2_cumem_granularity;
        size_t num_chunks =
            (reinterpret_cast<char *>(buf) - reinterpret_cast<char *>(t->heap_base) + length +
             t->log2_cumem_granularity - 1) >>
            t->log2_cumem_granularity;
        if (gpunetio_state->device_lkeys.size() < num_chunks * n_devs_selected) {
            gpunetio_state->device_lkeys.resize(num_chunks * n_devs_selected);
        }

        while (num_elements > 0) {
            for (int i = 0; i < n_devs_selected; i++) {
                device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
                nvshmemi_gpunetio_device_key_t dev_key;
                dev_key.key = device_lkey;
                dev_key.next_addr = reinterpret_cast<uint64_t>(buf) + length;
                gpunetio_state->device_lkeys.at(((chunk_idx + num_elements - 1) * n_devs_selected) +
                                                i) = dev_key;
            }
            --num_elements;
        }

        did_emplace = true;

        if (gpunetio_state->device_lkeys_d) {
            CUDA_RUNTIME_CHECK_RET(cudaFree(gpunetio_state->device_lkeys_d),
                                   NVSHMEMX_ERROR_INTERNAL);
            gpunetio_state->device_lkeys_d = nullptr;
        }

        num_lkeys = gpunetio_state->device_lkeys.size();

        // Put lkeys in constant memory first for cache optimization
        std::copy_n(gpunetio_state->device_lkeys.begin(),
                    std::min(num_lkeys, static_cast<size_t>(NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS)),
                    gpunetio_device_state->constmem.lkeys);

        // If we have overflow, put the rest in global memory
        if (num_lkeys > NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS) {
            size_t lkeys_array_size = sizeof(nvshmemi_gpunetio_device_key_t) *
                                      (num_lkeys - NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS);

            nvshmemi_gpunetio_device_key_t *data_ptr =
                &gpunetio_state->device_lkeys.data()[NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS];

            CUDA_RUNTIME_CHECK_RET(cudaMalloc(&gpunetio_state->device_lkeys_d, lkeys_array_size),
                                   NVSHMEMX_ERROR_OUT_OF_MEMORY);

            CUDA_RUNTIME_CHECK_RET(
                cudaMemcpyAsync(gpunetio_state->device_lkeys_d, (const void *)data_ptr,
                                lkeys_array_size, cudaMemcpyHostToDevice,
                                gpunetio_state->my_stream),
                NVSHMEMX_ERROR_INTERNAL);
        }
        gpunetio_device_state->globalmem.lkeys =
            static_cast<nvshmemi_gpunetio_device_key_t *>(gpunetio_state->device_lkeys_d);
    }

    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_INTERNAL);

    error_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    nvshmemi_gpunetio_device_state_t *gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);

    if (gpunetio_device_state == nullptr) {
        NVSHMEMI_WARN_PRINT("gpunetio_device_state is NULL\n");
    }

    auto *gpunetio_mem_handle = reinterpret_cast<struct gpunetio_mem_handle *>(mem_handle);
    nvshmemt_ib_common_mem_handle *handle = &gpunetio_mem_handle->dev_mem_handles[0];

    if (handle->local_only) {
        uint32_t position = 0;
        gpunetio_device_local_only_mhandle_cache *prev_mhandle_cache = nullptr;
        gpunetio_device_local_only_mhandle_cache *next_mhandle_cache = nullptr;
        gpunetio_device_local_only_mhandle_cache *curr_mhandle_cache = nullptr;
        void *mhandle_gpu_ptr;

        for (auto it = gpunetio_state->device_local_only_mhandles.begin();
             it != gpunetio_state->device_local_only_mhandles.end(); ++it) {
            if (it->mhandle.start == reinterpret_cast<uint64_t>(handle->buf)) {
                curr_mhandle_cache = &gpunetio_state->device_local_only_mhandles.data()[position];
                if (position > 0)
                    prev_mhandle_cache =
                        &gpunetio_state->device_local_only_mhandles.data()[position - 1];
                if (position < gpunetio_state->device_local_only_mhandles.size() - 1)
                    next_mhandle_cache =
                        &gpunetio_state->device_local_only_mhandles.data()[position + 1];
                break;
            }
            ++position;
        }
        if (!curr_mhandle_cache) {
            NVSHMEMI_ERROR_PRINT("mem_handle is not registered.\n");
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        // Remove this element from the linked list on both host and GPU.
        if (prev_mhandle_cache) {
            if (next_mhandle_cache)
                prev_mhandle_cache->mhandle.next =
                    static_cast<nvshmemi_gpunetio_device_local_only_mhandle_t *>(
                        next_mhandle_cache->dev_ptr);
            else
                prev_mhandle_cache->mhandle.next = nullptr;
            mhandle_gpu_ptr = reinterpret_cast<void *>(
                reinterpret_cast<uintptr_t>(prev_mhandle_cache->dev_ptr) +
                offsetof(nvshmemi_gpunetio_device_local_only_mhandle_t, next));
            CUDA_RUNTIME_CHECK_RET(
                cudaMemcpyAsync(mhandle_gpu_ptr, &prev_mhandle_cache->mhandle.next,
                                sizeof(prev_mhandle_cache->mhandle.next), cudaMemcpyHostToDevice,
                                gpunetio_state->my_stream),
                NVSHMEMX_ERROR_INTERNAL);
        } else {
            // The caller will trigger device state update.
            if (next_mhandle_cache)
                gpunetio_device_state->globalmem.local_only_mhandle_head =
                    static_cast<nvshmemi_gpunetio_device_local_only_mhandle_t *>(
                        next_mhandle_cache->dev_ptr);
            else
                gpunetio_device_state->globalmem.local_only_mhandle_head = nullptr;
        }
        // Free the copy of this element on GPU.
        CUDA_RUNTIME_CHECK_RET(cudaFree(curr_mhandle_cache->dev_ptr), NVSHMEMX_ERROR_INTERNAL);

        gpunetio_state->device_local_only_mhandles.erase(
            gpunetio_state->device_local_only_mhandles.begin() + position);
    }

    for (size_t i = 0; i < gpunetio_state->selected_dev_ids.size(); i++) {
        handle = &gpunetio_mem_handle->dev_mem_handles[i];
        status = nvshmemt_ib_common_release_mem_handle(
            &gpunetio_state->ftable, reinterpret_cast<nvshmem_mem_handle_t *>(handle),
            gpunetio_state->log_level);
        if (status) {
            NVSHMEMI_ERROR_PRINT("nvshmemt_ib_common_release_mem_handle failed.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }

    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_INTERNAL);

    return NVSHMEMX_SUCCESS;
}

static int gpunetio_init_ftables(std::unique_ptr<nvshmemt_gpunetio_state_t> &gpunetio_state,
                                 std::unique_ptr<nvshmemi_options_s> &options,
                                 nvshmemi_cuda_fn_table *table) {
    int status = 0;

    gpunetio_state->cuda_syms = table;
    if (nvshmemt_ibv_ftable_init(&gpunetio_state->ibv_handle, &gpunetio_state->ftable,
                                 gpunetio_state->log_level)) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "Unable to dlopen libibverbs. Skipping DOCA transport.\n");
    }

#ifdef NVSHMEM_USE_MLX5DV
    if (!options->DISABLE_DATA_DIRECT) {
        if (nvshmemt_mlx5dv_ftable_init(&gpunetio_state->mlx5dv_handle,
                                        &gpunetio_state->mlx5dv_ftable,
                                        gpunetio_state->log_level)) {
            NVSHMEMI_WARN_PRINT("Unable to dlopen libmlx5dv. Disabling directNIC features.\n");
            gpunetio_state->mlx5dv_ftable.mlx5dv_internal_is_supported = nullptr;
            gpunetio_state->mlx5dv_ftable.mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
            gpunetio_state->mlx5dv_ftable.mlx5dv_internal_reg_dmabuf_mr = nullptr;
        }
    } else {
        gpunetio_state->mlx5dv_ftable.mlx5dv_internal_is_supported = nullptr;
        gpunetio_state->mlx5dv_ftable.mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
        gpunetio_state->mlx5dv_ftable.mlx5dv_internal_reg_dmabuf_mr = nullptr;
        INFO(gpunetio_state->log_level,
             "directNIC features are disabled by NVSHMEM_DISABLE_DATA_DIRECT=1");
    }
#else
    INFO(gpunetio_state->log_level, "directNIC features are disabled\n");
#endif

    return NVSHMEMX_SUCCESS;
}

static int gpunetio_init_populate_state(std::unique_ptr<nvshmemt_gpunetio_state_t> &gpunetio_state,
                                        std::unique_ptr<nvshmemi_options_s> &options) {
    int status = 0;
    gpunetio_state->log_level = nvshmemt_common_get_log_level(options.get());
    gpunetio_state->skip_cst = true;  // will be set to false if multiple devices are selected or if
                                      // CST is required for a device
    gpunetio_initialize_cache_state(gpunetio_state.get());

    gpunetio_state->qp_depth = options->QP_DEPTH;
    if (gpunetio_state->qp_depth > 0) {
        gpunetio_state->qp_depth = gpunetio_round_up_pow2_or_0(gpunetio_state->qp_depth);
    }
    if (gpunetio_state->qp_depth <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH must be a positive number.\n");
    } else if (gpunetio_state->qp_depth < NVSHMEMI_GPUNETIO_MIN_QP_DEPTH) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH must be at least %d.\n",
                           NVSHMEMI_GPUNETIO_MIN_QP_DEPTH);
    } else if (gpunetio_state->qp_depth > NVSHMEMI_GPUNETIO_MAX_QP_DEPTH) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH can be at most %d.\n", NVSHMEMI_GPUNETIO_MAX_QP_DEPTH);
    }

    gpunetio_state->num_requests_in_batch = options->GPUNETIO_NUM_REQUESTS_IN_BATCH;
    if (gpunetio_state->num_requests_in_batch > 0) {
        gpunetio_state->num_requests_in_batch =
            gpunetio_round_up_pow2_or_0(gpunetio_state->num_requests_in_batch);
    }
    if (gpunetio_state->num_requests_in_batch <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH must be a positive number.\n");
    } else if (gpunetio_state->num_requests_in_batch > gpunetio_state->qp_depth) {
        NVSHMEMI_ERROR_RET(
            status, NVSHMEMX_ERROR_INVALID_VALUE,
            "NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH must not be larger than QP depth.\n");
    }

    gpunetio_state->num_fetch_slots_per_rc = options->GPUNETIO_NUM_FETCH_SLOTS_PER_RC;
    if (gpunetio_state->num_fetch_slots_per_rc > 0)
        gpunetio_state->num_fetch_slots_per_rc =
            gpunetio_round_up_pow2(gpunetio_state->num_fetch_slots_per_rc);
    if (gpunetio_state->num_fetch_slots_per_rc <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_GPUNETIO_NUM_FETCH_SLOTS_PER_RC must be a positive number.\n");
    }
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_init_gpu(std::unique_ptr<nvshmemt_gpunetio_state_t> &gpunetio_state,
                             std::unique_ptr<nvshmemi_options_s> &options) {
    CUdevice gpu_device_id;
    int status = 0;
    int lowest_stream_priority;
    int highest_stream_priority;

    status = CUPFN(gpunetio_state->cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        return status;
    }

    char pci_bus_id[MAX_GPU_PCI_ADDRESS_LEN];
    CUDA_RUNTIME_CHECK_RET(
        cudaDeviceGetPCIBusId(pci_bus_id, MAX_GPU_PCI_ADDRESS_LEN, gpu_device_id),
        NVSHMEMX_ERROR_INTERNAL);
    INFO(gpunetio_state->log_level, "Creating DOCA GPU device handler for GPU with bus ID: %s\n",
         pci_bus_id);
    status = doca_gpu_create(pci_bus_id, &gpunetio_state->gpu_device);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "doca_gpu_create failed to create handler for GPU with bus ID: %s\n",
                          pci_bus_id);

    status = nvshmemt_ib_common_check_dmabuf_support(
        gpunetio_state->dmabuf_support_for_data_buffers, gpunetio_state->cuda_syms,
        options->IB_DISABLE_DMABUF);
    if (status) return status;

    CUDA_RUNTIME_CHECK_RET(
        cudaDeviceGetStreamPriorityRange(&lowest_stream_priority, &highest_stream_priority),
        NVSHMEMX_ERROR_INTERNAL);
    CUDA_RUNTIME_CHECK_RET(
        cudaStreamCreateWithPriority(&gpunetio_state->my_stream, cudaStreamNonBlocking,
                                     highest_stream_priority),
        NVSHMEMX_ERROR_INTERNAL);

    return NVSHMEMX_SUCCESS;
}

static int gpunetio_init_nic_devices(nvshmem_transport *transport,
                                     std::unique_ptr<nvshmemt_gpunetio_state_t> &gpunetio_state,
                                     std::unique_ptr<nvshmemi_options_s> &options) {
    int num_devices = 0;
    ibv_device **dev_list = nullptr;
    int status = 0;
    doca_gpu_dev_verbs_nic_handler nic_handler_request = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    uint32_t atomic_host_endian_size = 0;

    status =
        gpunetio_parse_nic_handler_request(&nic_handler_request, options->GPUNETIO_NIC_HANDLER);
    NVSHMEMI_NZ_ERROR_RET(status, status, "NVSHMEM_GPUNETIO_NIC_HANDLER is not valid.\n");
    INFO(gpunetio_state->log_level, "NVSHMEM_GPUNETIO_NIC_HANDLER requested: %d\n",
         nic_handler_request);

    dev_list = gpunetio_state->ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_RET(dev_list, status, NVSHMEMX_ERROR_INTERNAL, "get_device_list failed \n");
    INFO(gpunetio_state->log_level, "Found %d devices\n", num_devices);

    struct nvshmemt_ib_hca_filter hca_filter = {};
    struct nvshmemt_ib_common_state temp_state = {};
    temp_state.options = options.get();
    temp_state.log_level = gpunetio_state->log_level;
    nvshmemt_ib_common_parse_hca_filter(hca_filter, temp_state);

    struct nvshmemt_ib_common_device common_devs[MAX_NUM_HCAS] = {};
    int dev_ids[MAX_NUM_PES_PER_NODE];
    int port_ids[MAX_NUM_PES_PER_NODE];
    temp_state.devices = common_devs;
    temp_state.dev_ids = dev_ids;
    temp_state.port_ids = port_ids;

    status = nvshmemt_ib_common_enumerate_devices(&gpunetio_state->ftable, temp_state,
                                                  sizeof(nvshmemt_ib_common_device), hca_filter,
                                                  dev_list, num_devices);
    if (status) return status;

    {
        bool device_checked[MAX_NUM_HCAS] = {};
        int write_idx = 0;
        for (int i = 0; i < temp_state.n_dev_ids; i++) {
            int dev_idx = temp_state.dev_ids[i];
            struct nvshmemt_ib_common_device *dev = &common_devs[dev_idx];

            if (!device_checked[dev_idx]) {
                device_checked[dev_idx] = true;
                const char *name = gpunetio_state->ftable.get_device_name(dev->dev);

                if (!nvshmemt_ib_common_query_mlx5_caps(dev->context)) {
                    NVSHMEMI_WARN_PRINT(
                        "device %s is not enumerated as an mlx5 device. Skipping...", name);
                    gpunetio_state->ftable.close_device(dev->context);
                    if (dev->pd) gpunetio_state->ftable.dealloc_pd(dev->pd);
                    dev->context = nullptr;
                    dev->pd = nullptr;
                    continue;
                }

                status = nvshmemt_ib_common_check_nic_ext_atomic_support(dev->context);
                if (status) {
                    NVSHMEMI_WARN_PRINT(
                        "device %s does not support all necessary atomic operations. You may want "
                        "to check the PCI_ATOMIC_MODE value in the NIC firmware. Skipping...\n",
                        name);
                    gpunetio_state->ftable.close_device(dev->context);
                    if (dev->pd) gpunetio_state->ftable.dealloc_pd(dev->pd);
                    dev->context = nullptr;
                    dev->pd = nullptr;
                    continue;
                }
            }

            if (!dev->context) continue;

            temp_state.dev_ids[write_idx] = temp_state.dev_ids[i];
            temp_state.port_ids[write_idx] = temp_state.port_ids[i];
            write_idx++;
        }
        temp_state.n_dev_ids = write_idx;
    }

    nvshmemt_ib_common_warn_missing_hcas(hca_filter);
    nvshmemt_ib_common_log_device_assignment(temp_state);

    if (!temp_state.n_dev_ids) {
        INFO(
            gpunetio_state->log_level,
            "no active IB device that supports GPU-initiated communication is found, exiting...\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = nvshmemt_ib_common_discover_pci_paths(
        transport, temp_state, sizeof(nvshmemt_ib_common_device), &gpunetio_state->ftable,
        &gpunetio_state->mlx5dv_ftable);
    if (status) return status;

    // We need to copy the detected devices into the vector-based structures of gpunetio_state.
    std::array<int, MAX_NUM_HCAS> dev_remap;
    dev_remap.fill(-1);
    for (int i = 0; i < temp_state.n_dev_ids; i++) {
        int dev_id = temp_state.dev_ids[i];
        if (dev_remap[dev_id] == -1) {
            gpunetio_device gpudev;
            gpudev.common_device = common_devs[dev_id];
            gpunetio_state->devices.push_back(std::move(gpudev));
            dev_remap[dev_id] = static_cast<int>(gpunetio_state->devices.size() - 1);
        }
        gpunetio_state->dev_ids.push_back(dev_remap[dev_id]);
        gpunetio_state->port_ids.push_back(temp_state.port_ids[i]);
    }

    for (int dev_id : gpunetio_state->dev_ids) {
        auto *device = &gpunetio_state->devices[dev_id];
        if (device->common_device.data_direct && !options->GPUNETIO_NUM_RC_PER_PE_provided) {
            // Need 8 QPs for achieving bandwidth in data direct device
            options->GPUNETIO_NUM_RC_PER_PE = 8;
            INFO(gpunetio_state->log_level,
                 "Setting GPUNETIO_NUM_RC_PER_PE = 8 as data direct device is detected");
        }

        // Report whether we need to do atomic endianness conversions on 8 byte operands.
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device->common_device.context);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");

        device->nic_handler_request = nic_handler_request;
    }

    if (options->GPUNETIO_NUM_RC_PER_PE <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "GPUNETIO_NUM_RC_PER_PE must be greater than 0");
    }

    transport->atomic_host_endian_min_size = atomic_host_endian_size;

    // Open NIC in GPUNetIO
    for (auto &device : gpunetio_state->devices) {
        status = doca_verbs_dev_open(device.common_device.pd, &device.net_dev);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed to open DOCA net device\n");
    }

    return NVSHMEMX_SUCCESS;
}

int nvshmemt_init(nvshmem_transport_t *t, nvshmemi_cuda_fn_table *table, int api_version) {
    int status = 0;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<nvshmemi_options_s> options(new nvshmemi_options_s());
    status = nvshmemi_env_options_init(options.get());
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "Unable to initialize NVSHMEM options.\n");

    // Allocate generic transport
    auto transport_del = [](nvshmem_transport *p) {
        free(p->device_pci_paths);
        free(p);
    };
    std::unique_ptr<nvshmem_transport, decltype(transport_del)> transport_owner(
        static_cast<nvshmem_transport *>(calloc(1, sizeof(nvshmem_transport))), transport_del);
    auto *transport = transport_owner.get();
    NVSHMEMI_NULL_ERROR_RET(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            "Unable to allocate transport stuct for doca transport.\n");

    // Global state for GPUNetIO transport
    std::unique_ptr<nvshmemt_gpunetio_state_t> gpunetio_state(new nvshmemt_gpunetio_state_t());
    transport->state = gpunetio_state.get();

    status = gpunetio_init_populate_state(gpunetio_state, options);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed while parsing options.\n");

    status = gpunetio_init_ftables(gpunetio_state, options, table);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed to initialize ftables.\n");

    status = gpunetio_init_gpu(gpunetio_state, options);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed to get and initialize GPU.\n");

    status = gpunetio_init_nic_devices(transport_owner.get(), gpunetio_state, options);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed to parse and select NICs.\n");

    transport->host_ops.can_reach_peer = nvshmemt_gpunetio_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_gpunetio_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_gpunetio_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_gpunetio_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_gpunetio_show_info;
    transport->host_ops.finalize = nvshmemt_gpunetio_finalize;
    transport->host_ops.rma = nullptr;
    transport->host_ops.amo = nullptr;
    transport->host_ops.fence = nullptr;
    transport->host_ops.quiet = nullptr;
    transport->host_ops.enforce_cst = nullptr;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_gpunetio_add_device_remote_mem_handles;
    transport->host_ops.put_signal = nullptr;
    // We update the progress function in connect_endpoints when we get the NIC handlers from
    // GPUNetIO for the different QPs.
    transport->host_ops.progress = nullptr;
    transport->no_proxy = true;
    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->type = NVSHMEM_TRANSPORT_LIB_CODE_GPUNETIO;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    gpunetio_state->options = std::move(options);
    gpunetio_state.release();
    *t = transport_owner.release();

    return NVSHMEMX_SUCCESS;
}
