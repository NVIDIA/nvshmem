/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <endian.h>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <algorithm>
#include <vector>

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host/scope_guard.h"
#include "transport_common.h"
#include "transport_ib_common.h"  // for nvshmemt_ib_common_mem_handle
#include "transport_mlx5_common.h"
#include "device_host_transport/nvshmem_constants.h"
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

// Forward declarations
static void gpunetio_activate_progress_function(nvshmem_transport_t t);

// GPUNetIO-specific constants
constexpr int GPUNETIO_GPAGE_BITS = 16;
constexpr size_t GPUNETIO_GPAGE_SIZE = (1ULL << GPUNETIO_GPAGE_BITS);

// First slot is reserved for non-fetch operations.
constexpr int GPUNETIO_IBUF_RESERVED_SLOTS = 1;

// QP connection parameters
constexpr uint32_t GPUNETIO_QP_PSN = 0;
constexpr uint32_t GPUNETIO_QP_PKEY_INDEX = 0;
constexpr int GPUNETIO_QP_RNR_RETRY = 7;
constexpr int GPUNETIO_QP_MIN_RNR_TIMER = 12;
constexpr int GPUNETIO_QP_HOP_LIMIT = 255;
constexpr bool GPUNETIO_QP_ALLOW_REMOTE_WRITE = true;
constexpr bool GPUNETIO_QP_ALLOW_REMOTE_READ = true;

// CPU data path WQE / CQE constants
constexpr size_t GPUNETIO_WQE_BB = sizeof(doca_gpu_dev_verbs_wqe);  // 64 bytes
constexpr size_t GPUNETIO_WQE_DS = 16;                              // mlx5 data-segment granularity
constexpr uint8_t GPUNETIO_CQE_OPCODE_REQ = 0x00;
constexpr uint8_t GPUNETIO_CQE_OPCODE_REQ_ERR = 0x0d;
constexpr uint8_t GPUNETIO_CQE_OPCODE_INVALID = 0x0f;
constexpr uint8_t GPUNETIO_WQE_CTRL_CQ_UPDATE = 0x08;  // MLX5_WQE_CTRL_CQ_UPDATE
// Extended (masked) atomic segments and WQEs
constexpr uint32_t GPUNETIO_4_BYTE_EXT_AMO_OPMOD = 0x08000000;
constexpr uint32_t GPUNETIO_8_BYTE_EXT_AMO_OPMOD = 0x09000000;

// CPU data path RW WQE / CQE structs
struct __attribute__((__packed__)) gpunetio_rw_inline_data_seg {
    uint32_t byte_count;
    union {
        uint64_t data_64;
        uint32_t data_32;
        uint16_t data_16;
        uint8_t data_8;
    } data;
    uint32_t reserved;
};
static_assert(sizeof(gpunetio_rw_inline_data_seg) == 16, "inline data seg must be 16 bytes");

struct __attribute__((__packed__, __aligned__(4))) gpunetio_rw_wqe {
    doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl;
    doca_gpunetio_ib_mlx5_wqe_raddr_seg raddr;
    union {
        doca_gpunetio_ib_mlx5_wqe_data_seg data_seg;
        gpunetio_rw_inline_data_seg data_inl;
    } data;
};
static_assert(sizeof(gpunetio_rw_wqe) == 48, "rw wqe must be 48 bytes");

struct __attribute__((__packed__, __aligned__(4))) gpunetio_atomic_64_wqe {
    doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl;
    doca_gpunetio_ib_mlx5_wqe_raddr_seg raddr;
    doca_gpunetio_ib_mlx5_wqe_atomic_seg atomic;
    doca_gpunetio_ib_mlx5_wqe_data_seg data;
};
static_assert(sizeof(gpunetio_atomic_64_wqe) == 64, "atomic wqe must be 64 bytes");

struct __attribute__((__packed__)) gpunetio_atomic_32_masked_fetch_add_seg {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
};
static_assert(sizeof(gpunetio_atomic_32_masked_fetch_add_seg) == 16,
              "32-bit masked FA seg must be 16 bytes");

struct __attribute__((__packed__)) gpunetio_atomic_32_masked_compare_swap_seg {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
};
static_assert(sizeof(gpunetio_atomic_32_masked_compare_swap_seg) == 16,
              "32-bit masked CS seg must be 16 bytes");

struct __attribute__((__packed__, __aligned__(4))) gpunetio_atomic_32_wqe {
    doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl;
    doca_gpunetio_ib_mlx5_wqe_raddr_seg raddr;
    union {
        gpunetio_atomic_32_masked_fetch_add_seg fa_seg;
        gpunetio_atomic_32_masked_compare_swap_seg cs_seg;
    };
    doca_gpunetio_ib_mlx5_wqe_data_seg data;
};
static_assert(sizeof(gpunetio_atomic_32_wqe) == 64, "32-bit atomic wqe must be 64 bytes");

struct __attribute__((__packed__)) gpunetio_atomic_64_masked_fetch_add_seg {
    uint64_t add_data;
    uint64_t field_boundary;
};
static_assert(sizeof(gpunetio_atomic_64_masked_fetch_add_seg) == 16,
              "64-bit masked FA seg must be 16 bytes");

struct __attribute__((__packed__)) gpunetio_atomic_64_masked_compare_swap_seg {
    uint64_t swap;
    uint64_t compare;
};
static_assert(sizeof(gpunetio_atomic_64_masked_compare_swap_seg) == 16,
              "64-bit masked CS seg must be 16 bytes");

struct __attribute__((__packed__, __aligned__(4))) gpunetio_atomic_64_masked_fa_wqe {
    doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl;
    doca_gpunetio_ib_mlx5_wqe_raddr_seg raddr;
    gpunetio_atomic_64_masked_fetch_add_seg fa_seg;
    doca_gpunetio_ib_mlx5_wqe_data_seg data;
};
static_assert(sizeof(gpunetio_atomic_64_masked_fa_wqe) == 64,
              "64-bit masked FA wqe must be 64 bytes");

// QP data-path kind: which path drives WQE posting / doorbell / CQ polling.
// GPU: WQEs posted by GPU SMs, rings in GPU memory (existing GPUNetIO mode).
// CPU: WQEs posted by CPU, rings in host memory (enable_umem_cpu + CPU_PROXY).
enum class gpunetio_qp_kind { GPU, CPU };

// Memory objects and internal buffers
struct gpunetio_device;
struct nvshmemt_gpunetio_state_t;

// DOCA-allocated GPU buffer + IB memory handle registered
struct gpunetio_internal_buffer {
    static std::pair<std::unique_ptr<gpunetio_internal_buffer>, nvshmemx_status> make(
        nvshmemt_gpunetio_state_t *state, gpunetio_device *device, size_t size);
    ~gpunetio_internal_buffer();

    enum doca_gpu_mem_type mem_type = DOCA_GPU_MEM_TYPE_GPU;
    struct {
        void *cpu_ptr = nullptr;
        void *gpu_ptr = nullptr;
        size_t size = 0;
    } aligned;
    nvshmemt_ib_common_mem_handle mem_handle{};

   private:
    gpunetio_internal_buffer() = default;
    nvshmemt_gpunetio_state_t *state_ = nullptr;
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
struct gpunetio_exch_info {
    int lid;
    int qpn;
    union ibv_gid gid;
    doca_verbs_gid vgid;
};

// Endpoint (= QP + metadata)
struct gpunetio_ep {
    static std::pair<std::unique_ptr<gpunetio_ep>, nvshmemx_status> make(
        nvshmem_transport_t t, nvshmemt_gpunetio_state_t *gpunetio_state,
        doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, gpunetio_device *device, int portid,
        uint32_t qp_idx, gpunetio_qp_kind kind);
    ~gpunetio_ep();

    int connect(nvshmemt_gpunetio_state_t *gpunetio_state, gpunetio_exch_info *remote_exch_info);
    gpunetio_exch_info create_exch_info() const;
    bool requires_cpu_proxy() const;
    int progress();

    // CPU data path polling
    int check_poll_avail(bool wait_all, uint32_t min_free_slots = 1);

    gpunetio_device *device_ = nullptr;
    doca_gpu_verbs_qp_hl *qp = nullptr;
    uint32_t qpn = 0;
    int portid = 0;
    uint32_t user_index = 0;
    gpunetio_qp_kind kind = gpunetio_qp_kind::GPU;
    int log_level = 0;
    std::unique_ptr<gpunetio_internal_buffer> internal_buf;

    // CPU data path tracking
    uint64_t head_op_id = 0;
    uint64_t tail_op_id = 0;
    uint16_t wqe_bb_idx = 0;
    uint32_t cqe_ci = 0;

   private:
    gpunetio_ep() = default;
};

// Device (single IB NIC)
struct gpunetio_device {
    static std::unique_ptr<gpunetio_device> make(nvshmemt_gpunetio_state_t *state);
    ~gpunetio_device();

    int open_net_dev();
    int create_ah(int portid);
    int create_qp_attr(doca_verbs_qp_attr_t **out_verbs_qp_attr, uint32_t dest_qp_num,
                       int portid) const;
    int transition_qp_to_rts(doca_verbs_qp_t *qp, doca_verbs_qp_attr_t *verbs_qp_attr) const;
    int add_endpoints(nvshmem_transport_t t, int portid, int num_rc_eps_per_pe,
                      gpunetio_qp_kind kind);
    int connect_self_loop(int portid, doca_gpu_verbs_qp_hl *qp_local,
                          doca_gpu_verbs_qp_hl *qp_backup);
    int progress(int n_pes);
    int cpu_progress_locked();
    gpunetio_ep *get_cpu_ep_from_qp_index(int pe, int qp_index, int n_pes, int mype);
    bool cst_is_required() const;

    // Flags
    int num_gpu_eps_per_pe = 0;
    int num_cpu_eps_per_pe = 0;
    int num_default_cpu_eps_per_pe = 0;
    // Common device information
    nvshmemt_ib_common_device common_device = {};
    // GPUNetIO-specific device information
    doca_dev_t *net_dev = nullptr;
    // Local RC endpoints for this device, split by data-path kind.
    std::vector<std::unique_ptr<gpunetio_ep>> rc_eps_gpu;
    std::vector<std::unique_ptr<gpunetio_ep>> rc_eps_cpu;
    // Stable global base for each per-device GPU QP slot, this is required when adding QPs to the
    // end of the flattened GPU QP array.
    std::vector<int> gpu_qp_handle_bases;
    // This mutex is required to avoid a race with the progress thread and the QP-specific API
    // reallocating and modifying the rc_eps vectors.
    std::unique_ptr<std::mutex> rc_eps_mtx{new std::mutex()};
    doca_verbs_ah_attr_t *ah = nullptr;
    // Per-kind self-loopback backup QP.
    doca_gpu_verbs_qp_hl *qp_local_backup_gpu = nullptr;
    doca_gpu_verbs_qp_hl *qp_local_backup_cpu = nullptr;
    doca_gpu_dev_verbs_nic_handler nic_handler_request = {};

    // CPU data path dummy MRs
    struct ibv_mr *dummy_mr = nullptr;
    void *dummy_mr_buf = nullptr;
    uint32_t dummy_mr_lkey = 0;
    // Round-robin counters for CPU EP selection (QP_DEFAULT + QP_ANY)
    uint32_t cpu_rr_default = 0;
    uint32_t cpu_rr_any = 0;
    // Serializing CPU QP polling between calling and progress threads
    std::mutex cpu_progress_mtx;

   private:
    gpunetio_device() = default;
    nvshmemt_gpunetio_state_t *state_ = nullptr;
};

// Transport instance state
struct nvshmemt_gpunetio_state_t {
    static std::pair<std::unique_ptr<nvshmemt_gpunetio_state_t>, nvshmemx_status> make(
        nvshmem_transport *transport, nvshmemi_options_s *options, nvshmemi_cuda_fn_table *table);
    ~nvshmemt_gpunetio_state_t();

    int connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids, int num_selected_devs,
                          int *out_qp_indices, int num_qps);
    gpunetio_ep *get_next_cpu_ep(int pe, int qp_index, int n_pes, int mype);

    // Per-instance transport state
    std::vector<std::unique_ptr<gpunetio_device>> devices;
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
    bool has_multiple_cpu_qps = false;
    int cur_gpu_qp_index = 0;
    int cur_cpu_qp_index = 0;
    // Index of the first user CPU QP (after the default QPs)
    int first_user_cpu_qp = -1;
    // QP-specific user QPs mapped to device and slot
    std::vector<std::pair<int, int>> user_cpu_qps;
    int last_num_rcs = 0;
    int qp_depth = 0;
    int num_fetch_slots_per_rc = 0;
    int num_requests_in_batch = 0;
    uint32_t cpu_dev_rr = 0;
    // Function tables
    nvshmemt_ibv_function_table ftable = {};
    void *ibv_handle = nullptr;
    nvshmemt_mlx5dv_function_table mlx5dv_ftable = {};
    void *mlx5dv_handle = nullptr;
    nvshmemi_cuda_fn_table *cuda_syms = nullptr;

   private:
    nvshmemt_gpunetio_state_t() = default;

    int init_populate_state(nvshmemi_options_s *options);
    int init_ftables(nvshmemi_options_s *options, nvshmemi_cuda_fn_table *table);
    int init_gpu(nvshmemi_options_s *options);
    int init_nic_devices(nvshmem_transport *transport, nvshmemi_options_s *options);

    void initialize_cache_state();
    int get_cuda_device_id(CUdevice *out);
    int connect_global_setup(int num_selected_devs, int *selected_dev_ids);
    int setup_gpu_state(nvshmem_transport_t t);
    int connect_qps_only(nvshmem_transport_t t, int *out_qp_indices, int num_qps);
};

// Utility functions
static inline int gpunetio_round_up_pow2(int n) {
    int pow2 = 0;
    for (pow2 = 1; pow2 < n; pow2 <<= 1)
        ;
    return pow2;
}

static constexpr int gpunetio_round_up_pow2_or_0(int n) {
    return (n == 0) ? 0 : gpunetio_round_up_pow2(n);
}

gpunetio_ep *gpunetio_device::get_cpu_ep_from_qp_index(int pe, int qp_index, int n_pes, int mype) {
    int num_total = num_cpu_eps_per_pe;
    int num_default = num_default_cpu_eps_per_pe;

    int slot;
    if (qp_index == NVSHMEMX_QP_HOST) {
        slot = 0;
    } else if (qp_index == NVSHMEMX_QP_DEFAULT) {
        if (num_default <= 0) {
            // No QPs in default pool
            return nullptr;
        }
        if (pe == mype) {
            slot = 1;
        } else {
            slot = 1 + static_cast<int>(cpu_rr_default++ % static_cast<uint32_t>(num_default));
        }
    } else if (qp_index == NVSHMEMX_QP_ANY) {
        if (num_total <= 1) {
            // No QPs in any pool
            return nullptr;
        }
        slot = 1 + static_cast<int>(cpu_rr_any % static_cast<uint32_t>(num_total - 1));
        cpu_rr_any = (cpu_rr_any + 1) % static_cast<uint32_t>(num_total - 1);
    } else if (qp_index >= 0 && qp_index < num_total) {
        slot = qp_index;
    } else {
        return nullptr;
    }
    return rc_eps_cpu[slot * n_pes + pe].get();
}

// Host-side SQ post
static inline void gpunetio_cpu_post_send(gpunetio_ep *ep, uint32_t num_wqe_consumed) {
    auto *qp = ep->qp->qp_gverbs;
    uint32_t idx = static_cast<uint32_t>(ep->wqe_bb_idx);

    STORE_BARRIER();

    reinterpret_cast<std::atomic<uint64_t> *>(qp->cpu_db)
        ->store(static_cast<uint64_t>(idx), std::memory_order_release);

    ep->head_op_id += num_wqe_consumed;
}

// Parse and cache
static int gpunetio_parse_nic_handler_request(doca_gpu_dev_verbs_nic_handler *out_loc,
                                              std::string_view str) {
    auto start = str.find_first_not_of(' ');
    if (start == std::string_view::npos) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    auto req = str.substr(start, str.find_last_not_of(' ') - start + 1);

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

static bool gpunetio_qp_requires_cpu_proxy(doca_gpu_verbs_qp_hl *qp) {
    return qp->qp_gverbs->qp_cpu->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
}

// gpunetio_internal_buffer implementation
std::pair<std::unique_ptr<gpunetio_internal_buffer>, nvshmemx_status>
gpunetio_internal_buffer::make(nvshmemt_gpunetio_state_t *state, gpunetio_device *device,
                               size_t size) {
    std::unique_ptr<gpunetio_internal_buffer> buf(new gpunetio_internal_buffer());
    buf->state_ = state;
    buf->mem_type = DOCA_GPU_MEM_TYPE_GPU;
    buf->aligned.cpu_ptr = nullptr;
    buf->aligned.size = size;

    int status = doca_gpu_mem_alloc(state->gpu_device, buf->aligned.size, GPUNETIO_GPAGE_SIZE,
                                    DOCA_GPU_MEM_TYPE_GPU, &buf->aligned.gpu_ptr, nullptr);
    if (status) {
        NVSHMEMI_ERROR_PRINT("cannot allocate internal buffer.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    status = nvshmemt_ib_common_reg_mem_handle(
        &state->ftable, &state->mlx5dv_ftable, device->common_device.pd,
        reinterpret_cast<nvshmem_mem_handle_t *>(&buf->mem_handle), buf->aligned.gpu_ptr,
        buf->aligned.size, false, state->dmabuf_support_for_data_buffers, state->cuda_syms,
        state->log_level, state->options->IB_ENABLE_RELAXED_ORDERING,
        device->common_device.data_direct);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to register memory for DOCA transport.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    return {std::move(buf), NVSHMEMX_SUCCESS};
}

gpunetio_internal_buffer::~gpunetio_internal_buffer() {
    if (!state_) return;
    if (mem_handle.mr) {
        nvshmemt_ib_common_release_mem_handle(&state_->ftable,
                                              reinterpret_cast<nvshmem_mem_handle_t *>(&mem_handle),
                                              state_->log_level);
    }
    if (aligned.gpu_ptr) {
        doca_gpu_mem_free(state_->gpu_device, aligned.gpu_ptr);
    }
}

// gpunetio_ep lifecycle
std::pair<std::unique_ptr<gpunetio_ep>, nvshmemx_status> gpunetio_ep::make(
    nvshmem_transport_t t, nvshmemt_gpunetio_state_t *gpunetio_state,
    doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, gpunetio_device *device, int portid_in,
    uint32_t qp_idx, gpunetio_qp_kind kind) {
    std::unique_ptr<gpunetio_ep> ep(new gpunetio_ep());

    int status = doca_gpu_verbs_create_qp_hl(qp_init_attr, &ep->qp);
    if (status) {
        NVSHMEMI_ERROR_PRINT("doca_gpu_verbs_create_qp_hl failed.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    ep->device_ = device;
    ep->portid = portid_in;
    ep->user_index = qp_idx;
    ep->kind = kind;
    ep->log_level = gpunetio_state->log_level;
    {
        doca_error_t doca_ret = doca_verbs_qp_get_qpn(ep->qp->qp, &ep->qpn);
        if (doca_ret != DOCA_SUCCESS) {
            INFO(gpunetio_state->log_level, "gpunetio error: %d", doca_ret);
            return {nullptr, NVSHMEMX_ERROR_INTERNAL};
        }
    }

    auto [ibuf, ibuf_status] = gpunetio_internal_buffer::make(
        gpunetio_state, device,
        NVSHMEMI_GPUNETIO_IBUF_SLOT_SIZE *
            (gpunetio_state->num_fetch_slots_per_rc + GPUNETIO_IBUF_RESERVED_SLOTS));
    if (ibuf_status) {
        NVSHMEMI_ERROR_PRINT("gpunetio_internal_buffer::make failed.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }
    ep->internal_buf = std::move(ibuf);

    TRACE(gpunetio_state->log_level, "Created QP: qp_idx=%d, qpn=%d", qp_idx, ep->qpn);

    if (ep->requires_cpu_proxy()) {
        gpunetio_activate_progress_function(t);
    }

    return {std::move(ep), NVSHMEMX_SUCCESS};
}

int gpunetio_ep::connect(nvshmemt_gpunetio_state_t *gpunetio_state,
                         gpunetio_exch_info *remote_exch_info) {
    const struct ibv_port_attr *port_attr = device_->common_device.port_attr + (portid - 1);
    uint16_t dlid = remote_exch_info->lid;

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        const struct nvshmemt_ib_qp_path path = nvshmemt_ib_select_qp_path(
            &device_->common_device.gid_info[portid - 1].local_gid, port_attr->lid,
            remote_exch_info->lid, remote_exch_info->gid.global.subnet_prefix,
            remote_exch_info->gid.global.interface_id);
        const bool use_ib_grh = gpunetio_state->options->IB_FORCE_GRH ||
                                nvshmemt_ib_common_port_requires_grh(port_attr) ||
                                path.grh_required;

        /* The AH is shared across endpoints, so set its address type for every peer. */
        DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(
            device_->ah,
            use_ib_grh ? DOCA_VERBS_ADDR_TYPE_IB_GRH : DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
        if (use_ib_grh) {
            DOCA_CHECK(doca_verbs_ah_attr_set_hop_limit(device_->ah, GPUNETIO_QP_HOP_LIMIT));
        }
        dlid = path.dlid;
    }

    DOCA_CHECK(doca_verbs_ah_attr_set_gid(device_->ah, remote_exch_info->vgid));
    DOCA_CHECK(doca_verbs_ah_attr_set_dlid(device_->ah, dlid));

    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    int rc = device_->create_qp_attr(&verbs_qp_attr, remote_exch_info->qpn, portid);
    if (rc) return rc;

    rc = device_->transition_qp_to_rts(qp->qp, verbs_qp_attr);
    doca_verbs_qp_attr_destroy(verbs_qp_attr);
    return rc;
}

gpunetio_exch_info gpunetio_ep::create_exch_info() const {
    const auto &gid = device_->common_device.gid_info[portid - 1].local_gid;
    gpunetio_exch_info info{};
    info.lid = device_->common_device.port_attr[portid - 1].lid;
    info.qpn = qpn;
    info.gid = gid;
    std::copy(std::begin(gid.raw), std::end(gid.raw), info.vgid.raw);
    return info;
}

bool gpunetio_ep::requires_cpu_proxy() const { return gpunetio_qp_requires_cpu_proxy(qp); }

int gpunetio_ep::progress() {
    if (!requires_cpu_proxy()) {
        return NVSHMEMX_SUCCESS;
    }

    if (kind == gpunetio_qp_kind::CPU) {
        // Host-side CQ polling for CPU data-path. Caller must hold
        // device_->cpu_progress_mtx.
        auto *qp_cpu = qp->qp_gverbs->qp_cpu;
        auto *cq = &qp_cpu->cq_sq;

        auto *cqe = reinterpret_cast<volatile doca_gpunetio_ib_mlx5_cqe64 *>(
            cq->cqe_daddr + (cqe_ci & cq->cqe_mask) * sizeof(doca_gpunetio_ib_mlx5_cqe64));
        int ownership = (cqe_ci / cq->cqe_num) & 1;

        while (!((cqe->op_own & 0x01) ^ ownership)) {
            uint8_t opcode = cqe->op_own >> 4;
            if (opcode == GPUNETIO_CQE_OPCODE_INVALID) break;
            if (opcode == GPUNETIO_CQE_OPCODE_REQ_ERR) {
                auto *err = reinterpret_cast<volatile doca_gpunetio_ib_mlx5_err_cqe_ex *>(cqe);
                TRACE(log_level,
                      "CQ error ep=%p qpn=%u syndrome=0x%x vendor=0x%x "
                      "hw_synd=0x%x wqe_counter=%u",
                      static_cast<void *>(this), qpn, err->syndrome, err->vendor_err_synd,
                      err->hw_err_synd, ntohs(err->wqe_counter));
                tail_op_id++;
            } else {
                uint32_t sop_opcode = ntohl(cqe->sop_drop_qpn) & 0xFF000000;
                uint32_t bcnt = ntohl(cqe->byte_cnt);
                if (sop_opcode == (DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS << 24) &&
                    bcnt >= 8) {
                    tail_op_id += 2;
                } else {
                    tail_op_id++;
                }
            }
            cqe_ci++;
            cq->cqe_ci = cqe_ci;
            STORE_BARRIER();
            *cq->dbrec = htobe32(cqe_ci & 0x00ffffff);
            cqe = reinterpret_cast<volatile doca_gpunetio_ib_mlx5_cqe64 *>(
                cq->cqe_daddr + (cqe_ci & cq->cqe_mask) * sizeof(doca_gpunetio_ib_mlx5_cqe64));
            ownership = (cqe_ci / cq->cqe_num) & 1;
        }
        return NVSHMEMX_SUCCESS;
    } else {
        // GPU-kind EPs with CPU_PROXY: use doca_gpu_verbs_cpu_proxy_progress.
        int status = doca_gpu_verbs_cpu_proxy_progress(qp->qp_gverbs, nullptr);
        if (status) {
            NVSHMEMI_WARN_PRINT("doca_gpu_verbs_cpu_proxy_progress failed for ep %p \n",
                                static_cast<void *>(this));
        }
        return status;
    }
}

int gpunetio_ep::check_poll_avail(bool wait_all, uint32_t min_free_slots) {
    auto *qp_cpu = qp->qp_gverbs->qp_cpu;
    uint32_t threshold = wait_all ? 0 : static_cast<uint32_t>(qp_cpu->sq_wqe_num) - min_free_slots;
    while (head_op_id - tail_op_id > threshold) {
        device_->cpu_progress_locked();
    }
    return NVSHMEMX_SUCCESS;
}

gpunetio_ep::~gpunetio_ep() {
    if (qp) {
        int status = doca_gpu_verbs_destroy_qp_hl(qp);
        if (status) {
            NVSHMEMI_WARN_PRINT("doca_gpu_verbs_destroy_qp_hl failed for ep %p \n",
                                static_cast<void *>(this));
        }
    }
}

// gpunetio_device
std::unique_ptr<gpunetio_device> gpunetio_device::make(nvshmemt_gpunetio_state_t *state) {
    std::unique_ptr<gpunetio_device> dev(new gpunetio_device());
    dev->state_ = state;
    return dev;
}

int gpunetio_device::open_net_dev() {
    int status = doca_verbs_dev_open(common_device.pd, &net_dev);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "Failed to open DOCA net device\n");
    return NVSHMEMX_SUCCESS;
}

bool gpunetio_device::cst_is_required() const {
    bool rval = true;

    int order = 0;
    if (CUPFN(state_->cuda_syms,
              cuDeviceGetAttribute(
                  &order, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                  state_->cached_gpu_device_id))) {
        NVSHMEMI_WARN_PRINT("Cannot query dev attr. Assuming no GDR write ordering\n");
    } else {
        // GPU guarantees incoming PCIe write ordering. No need to do CST.
        if (order >= CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) rval = false;
    }
    rval = rval || common_device.data_direct;
    return rval;
}

// QP connection logic
int gpunetio_device::create_ah(int portid) {
    // DOCA_CHECK macro captures `gpunetio_state` from the enclosing scope.
    nvshmemt_gpunetio_state_t *gpunetio_state = state_;

    doca_verbs_ah_attr_t *local_ah = nullptr;
    DOCA_CHECK(doca_verbs_ah_attr_create(net_dev, &local_ah));
    auto ah_deleter = [](doca_verbs_ah_attr_t *p) { doca_verbs_ah_attr_destroy(p); };
    std::unique_ptr<doca_verbs_ah_attr_t, decltype(ah_deleter)> ah_guard(local_ah, ah_deleter);

    DOCA_CHECK(doca_verbs_ah_attr_set_sl(local_ah, state_->options->IB_SL));
    DOCA_CHECK(doca_verbs_ah_attr_set_traffic_class(local_ah, state_->options->IB_TRAFFIC_CLASS));

    const struct ibv_port_attr *port_attr = common_device.port_attr + (portid - 1);
    const bool use_ib_grh =
        state_->options->IB_FORCE_GRH || nvshmemt_ib_common_port_requires_grh(port_attr);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(
            local_ah, use_ib_grh ? DOCA_VERBS_ADDR_TYPE_IB_GRH : DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
        if (use_ib_grh) {
            DOCA_CHECK(doca_verbs_ah_attr_set_hop_limit(local_ah, GPUNETIO_QP_HOP_LIMIT));
        }
    } else {
        DOCA_CHECK(doca_verbs_ah_attr_set_addr_type(local_ah, DOCA_VERBS_ADDR_TYPE_IPv4));
        DOCA_CHECK(doca_verbs_ah_attr_set_hop_limit(local_ah, GPUNETIO_QP_HOP_LIMIT));
    }

    DOCA_CHECK(doca_verbs_ah_attr_set_sgid_index(
        local_ah, common_device.gid_info[portid - 1].local_gid_index));

    ah = ah_guard.release();
    return NVSHMEMX_SUCCESS;
}

int gpunetio_device::create_qp_attr(doca_verbs_qp_attr_t **out_verbs_qp_attr, uint32_t dest_qp_num,
                                    int portid) const {
    nvshmemt_gpunetio_state_t *gpunetio_state = state_;

    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    DOCA_CHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
    auto deleter = [](doca_verbs_qp_attr_t *p) { doca_verbs_qp_attr_destroy(p); };
    std::unique_ptr<doca_verbs_qp_attr_t, decltype(deleter)> attr_uptr(verbs_qp_attr, deleter);

    doca_verbs_device_attr *verbs_device_attr;
    DOCA_CHECK(doca_verbs_query_device(common_device.context, &verbs_device_attr));
    auto dev_attr_deleter = [](doca_verbs_device_attr *p) { doca_verbs_device_attr_free(p); };
    std::unique_ptr<doca_verbs_device_attr, decltype(dev_attr_deleter)> dev_attr_uptr(
        verbs_device_attr, dev_attr_deleter);
    uint8_t max_rd_atomic = doca_verbs_device_attr_get_max_qp_rd_atom(verbs_device_attr);
    uint8_t max_dest_rd_atomic = doca_verbs_device_attr_get_max_qp_init_rd_atom(verbs_device_attr);

    DOCA_CHECK(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, GPUNETIO_QP_PSN));
    DOCA_CHECK(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, GPUNETIO_QP_PSN));
    DOCA_CHECK(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, GPUNETIO_QP_PKEY_INDEX));
    DOCA_CHECK(doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_VERBS_MTU_SIZE_4K_BYTES));
    DOCA_CHECK(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, portid));
    DOCA_CHECK(
        doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, gpunetio_state->options->IB_TIMEOUT));
    DOCA_CHECK(
        doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, gpunetio_state->options->IB_RETRY_CNT));
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

int gpunetio_device::transition_qp_to_rts(doca_verbs_qp_t *qp,
                                          doca_verbs_qp_attr_t *verbs_qp_attr) const {
    nvshmemt_gpunetio_state_t *gpunetio_state = state_;

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

int gpunetio_device::connect_self_loop(int portid, doca_gpu_verbs_qp_hl *qp_local,
                                       doca_gpu_verbs_qp_hl *qp_backup) {
    nvshmemt_gpunetio_state_t *gpunetio_state = state_;
    ibv_port_attr port_attr;
    const union ibv_gid *gid = &common_device.gid_info[portid - 1].local_gid;
    doca_verbs_gid vgid;
    uint32_t dest_qp_num;
    ibv_query_port(common_device.context, portid, &port_attr);

    memcpy(vgid.raw, gid->raw, sizeof(union ibv_gid));

    DOCA_CHECK(doca_verbs_ah_attr_set_gid(ah, vgid));
    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DOCA_CHECK(doca_verbs_ah_attr_set_dlid(ah, port_attr.lid));
    }

    DOCA_CHECK(doca_verbs_qp_get_qpn(qp_backup->qp, &dest_qp_num));
    doca_verbs_qp_attr_t *verbs_qp_attr = nullptr;
    int rc = create_qp_attr(&verbs_qp_attr, dest_qp_num, portid);
    if (rc) return rc;
    auto qp_attr_guard = make_scope_guard([&]() { doca_verbs_qp_attr_destroy(verbs_qp_attr); });

    DOCA_CHECK(doca_verbs_qp_get_qpn(qp_local->qp, &dest_qp_num));
    doca_verbs_qp_attr_t *verbs_qp_attr_backup = nullptr;
    rc = create_qp_attr(&verbs_qp_attr_backup, dest_qp_num, portid);
    if (rc) return rc;
    auto qp_attr_backup_guard =
        make_scope_guard([&]() { doca_verbs_qp_attr_destroy(verbs_qp_attr_backup); });

    rc = transition_qp_to_rts(qp_local->qp, verbs_qp_attr);
    if (rc) return rc;

    rc = transition_qp_to_rts(qp_backup->qp, verbs_qp_attr_backup);
    return rc;
}

// Per-device endpoint setup. The kind selects which data path the new QPs belong to:
//   - GPU: device-driven QPs
//   - CPU: host-driven QPs
int gpunetio_device::add_endpoints(nvshmem_transport_t t, int portid, int num_rc_eps_per_pe,
                                   gpunetio_qp_kind kind) {
    int status = 0;
    int mype = t->my_pe;
    int n_pes = t->n_pes;
    int new_num_rc_eps = num_rc_eps_per_pe * n_pes;
    doca_gpu_verbs_qp_init_attr_hl qp_init_attr = {};
    const char *kind_str = (kind == gpunetio_qp_kind::CPU) ? "CPU" : "GPU";

    // Per-kind state aliases so the rest of the routine is kind-agnostic.
    auto &rc_eps = (kind == gpunetio_qp_kind::CPU) ? rc_eps_cpu : rc_eps_gpu;
    auto &num_eps_per_pe =
        (kind == gpunetio_qp_kind::CPU) ? num_cpu_eps_per_pe : num_gpu_eps_per_pe;
    auto &qp_local_backup =
        (kind == gpunetio_qp_kind::CPU) ? qp_local_backup_cpu : qp_local_backup_gpu;

    // exch_info structures
    std::vector<gpunetio_exch_info> local_exch_info(new_num_rc_eps);
    std::vector<gpunetio_exch_info> peer_exch_info(new_num_rc_eps);

    // get first index of additional RC endpoints (within this kind's vector)
    int rc_first_index = num_eps_per_pe * n_pes;

    // Only the first device has a host QP, subsequent don't have one
    const bool skip_host_slot = (kind == gpunetio_qp_kind::CPU) && (rc_first_index == 0) &&
                                (this != state_->devices[state_->selected_dev_ids[0]].get());

    if (new_num_rc_eps <= 0) {
        return NVSHMEMX_SUCCESS;
    }

    {
        std::lock_guard<std::mutex> lk(*rc_eps_mtx);
        rc_eps.resize(rc_eps.size() + new_num_rc_eps);
    }

    auto ep_cleanup_guard = make_scope_guard([&]() {
        // Reset EP vector to original size on failure, destructors will be called on shrinking
        std::lock_guard<std::mutex> lk(*rc_eps_mtx);
        rc_eps.resize(rc_first_index);

        // If we created the local backup QP for this kind, destroy it
        if (rc_first_index == 0 && qp_local_backup) {
            doca_gpu_verbs_destroy_qp_hl(qp_local_backup);
            qp_local_backup = nullptr;
        }
    });

    if (!common_device.pd) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "common_device.pd is NULL for device\n");
    }
    if (!common_device.context) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "common_device.context is NULL for device\n");
    }

    if (!state_->gpu_device) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "state_->gpu_device is NULL\n");
    }

    qp_init_attr.gpu_dev = state_->gpu_device;
    qp_init_attr.ibpd = common_device.pd;
    qp_init_attr.sq_nwqe = state_->qp_depth;
    qp_init_attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
    qp_init_attr.cq_collapsed = true;
    qp_init_attr.net_dev = net_dev;

    if (kind == gpunetio_qp_kind::CPU) {
        // CPU data path: SQ/CQ/DBR umem in host memory, doorbells driven by the CPU proxy.
        // Use a standard (non-collapsed) CQ so the host can poll with sequential
        // ownership-bit logic; collapsed CQ is a GPU-kernel optimisation.
        qp_init_attr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
        qp_init_attr.enable_umem_cpu = true;
        qp_init_attr.cq_collapsed = false;
    } else {
        qp_init_attr.nic_handler = nic_handler_request;
    }

    if (state_->options->GPUNETIO_ENABLE_ORDERING_SEMANTIC) {
        INFO(state_->log_level, "Ordering semantic for DDP will be enabled via GPUNetIO\n");
        qp_init_attr.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL;
    }

    INFO(state_->log_level, "Creating %d %s-data-path RC QPs", num_rc_eps_per_pe, kind_str);
    for (int i = 0; i < num_rc_eps_per_pe; i++) {
        // Only the first device has a host QP, subsequent don't have one
        if (skip_host_slot && i == 0) continue;
        for (int j = 0; j < n_pes; j++) {
            int dst_pe = (i * n_pes + 1 + mype + j) % n_pes;
            int mapped_i = rc_first_index + i * n_pes + dst_pe;
            int local_mapped_i = i + num_rc_eps_per_pe * dst_pe;

            // Skip self-loop QP
            if (dst_pe == mype) continue;

            TRACE(state_->log_level, "[%s] dst_pe: %d, mapped_i: %d, local_mapped_i: %d", kind_str,
                  dst_pe, mapped_i, local_mapped_i);

            auto [ep, ep_status] =
                gpunetio_ep::make(t, state_, &qp_init_attr, this, portid, mapped_i, kind);

            if (ep_status != NVSHMEMX_SUCCESS) {
                if (state_->options->GPUNETIO_ENABLE_ORDERING_SEMANTIC) {
                    NVSHMEMI_ERROR_PRINT(
                        "gpunetio_ep::make with ordering semantic enabled failed, please retry "
                        "with "
                        "NVSHMEM_GPUNETIO_ENABLE_ORDERING_SEMANTIC=0\n");
                }
                NVSHMEMI_ERROR_PRINT("gpunetio_ep::make failed on %s RC #%d.", kind_str, mapped_i);
                return NVSHMEMX_ERROR_INTERNAL;
            }

            local_exch_info[local_mapped_i] = ep->create_exch_info();
            rc_eps[mapped_i] = std::move(ep);
        }
    }

    status =
        t->boot_handle->alltoall(local_exch_info.data(), peer_exch_info.data(),
                                 sizeof(gpunetio_exch_info) * num_rc_eps_per_pe, t->boot_handle);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "alltoall of exch_info failed.\n");

    for (int i = 0; i < num_rc_eps_per_pe; ++i) {
        // Mirror the make loop: empty host slot on non-host devices has no EPs to connect.
        if (skip_host_slot && i == 0) continue;
        for (int j = 0; j < n_pes; ++j) {
            int ep_index = rc_first_index + i * n_pes + j;
            int peer_handle_index = num_rc_eps_per_pe * j + i;
            // No loopback to self
            if (j == mype) {
                continue;
            }
            TRACE(state_->log_level,
                  "[%s] Resetting and initializing RC #%d with qp_idx #%d QPN: %d", kind_str,
                  ep_index, rc_eps[ep_index]->user_index, rc_eps[ep_index]->qpn);
            TRACE(state_->log_level, "[%s] local QPN: %d, remote handle QPN: %d", kind_str,
                  rc_eps[ep_index]->qpn, peer_exch_info[peer_handle_index].qpn);

            status = rc_eps[ep_index]->connect(state_, &peer_exch_info[peer_handle_index]);
            NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                                  "gpunetio_ep::connect failed on %s RC #%d.", kind_str, ep_index);

            TRACE(state_->log_level, "[%s] DONE RC #%d", kind_str, ep_index);
        }
    }

    // One loopback QP per device per kind, created only on the first batch for that kind

    if (rc_first_index == 0) {
        // For CPU kind, the first slot is the dedicated host EP, so shift by one.
        int slot_within_batch = (kind == gpunetio_qp_kind::CPU) ? 1 : 0;
        int mype_ep_index = rc_first_index + slot_within_batch * n_pes + mype;

        assert(rc_eps[mype_ep_index] == nullptr);
        auto [loopback_ep, lb_status] =
            gpunetio_ep::make(t, state_, &qp_init_attr, this, portid, mype_ep_index, kind);
        NVSHMEMI_NZ_ERROR_RET(lb_status, NVSHMEMX_ERROR_INTERNAL,
                              "gpunetio_ep::make failed on %s loopback QP.\n", kind_str);
        rc_eps[mype_ep_index] = std::move(loopback_ep);

        // Dummy backup QP to have matching local QP for this kind
        status = doca_gpu_verbs_create_qp_hl(&qp_init_attr, &qp_local_backup);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "doca_gpu_verbs_create_qp_hl failed for %s backup.", kind_str);
        if (gpunetio_qp_requires_cpu_proxy(qp_local_backup)) {
            gpunetio_activate_progress_function(t);
        }

        // Connect self-loop QP RC to backup local QP
        status = connect_self_loop(portid, rc_eps[mype_ep_index]->qp, qp_local_backup);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "gpunetio_device::connect_self_loop failed on %s loopback QP.\n",
                              kind_str);
    }

    // Allocate dummy local MR for non-fetch AMO data segments (once per device).
    if (kind == gpunetio_qp_kind::CPU && dummy_mr == nullptr) {
        void *buf = nullptr;
        if (posix_memalign(&buf, 64, 64) != 0 || buf == nullptr) {
            NVSHMEMI_ERROR_PRINT("posix_memalign for dummy MR buffer failed.\n");
            return NVSHMEMX_ERROR_OUT_OF_MEMORY;
        }
        memset(buf, 0, 64);
        struct ibv_mr *mr =
            state_->ftable.reg_mr(common_device.pd, buf, 64,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if (!mr) {
            free(buf);
            NVSHMEMI_ERROR_PRINT("ibv_reg_mr for dummy MR failed.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }
        dummy_mr = mr;
        dummy_mr_buf = buf;
        dummy_mr_lkey = mr->lkey;
    }

    {
        std::lock_guard<std::mutex> lk(*rc_eps_mtx);
        num_eps_per_pe += num_rc_eps_per_pe;
    }
    if (kind == gpunetio_qp_kind::GPU) {
        for (int i = 0; i < num_rc_eps_per_pe; ++i) {
            gpu_qp_handle_bases.push_back(state_->cur_gpu_qp_index + i * n_pes);
        }
        state_->cur_gpu_qp_index += new_num_rc_eps;
    } else {
        state_->cur_cpu_qp_index += new_num_rc_eps;
    }
    // Set global state to skip_cst as soon as one device requires CST
    state_->skip_cst &= (!cst_is_required());

    ep_cleanup_guard.dismiss();
    return status;
}

int gpunetio_device::progress(int n_pes) {
    {
        std::lock_guard<std::mutex> lk(*rc_eps_mtx);
        int num_gpu_eps = num_gpu_eps_per_pe * n_pes;
        for (int j = 0; j < num_gpu_eps; j++) {
            if (!rc_eps_gpu[j]) continue;
            rc_eps_gpu[j]->progress();
        }
    }
    cpu_progress_locked();

    if (qp_local_backup_gpu && qp_local_backup_gpu->qp_gverbs &&
        gpunetio_qp_requires_cpu_proxy(qp_local_backup_gpu)) {
        int status = doca_gpu_verbs_cpu_proxy_progress(qp_local_backup_gpu->qp_gverbs, nullptr);
        if (status) {
            NVSHMEMI_WARN_PRINT(
                "doca_gpu_verbs_cpu_proxy_progress failed for qp_local_backup_gpu\n");
        }
    }
    if (qp_local_backup_cpu && qp_local_backup_cpu->qp_gverbs &&
        gpunetio_qp_requires_cpu_proxy(qp_local_backup_cpu)) {
        int status = doca_gpu_verbs_cpu_proxy_progress(qp_local_backup_cpu->qp_gverbs, nullptr);
        if (status) {
            NVSHMEMI_WARN_PRINT(
                "doca_gpu_verbs_cpu_proxy_progress failed for qp_local_backup_cpu\n");
        }
    }
    return NVSHMEMX_SUCCESS;
}

int gpunetio_device::cpu_progress_locked() {
    std::lock_guard<std::mutex> lk(cpu_progress_mtx);
    for (auto &ep : rc_eps_cpu) {
        if (!ep) continue;
        if (ep->requires_cpu_proxy()) doca_gpu_verbs_cpu_proxy_progress(ep->qp->qp_gverbs, nullptr);
        ep->progress();
    }
    return NVSHMEMX_SUCCESS;
}

gpunetio_device::~gpunetio_device() {
    // Destroy endpoints
    rc_eps_gpu.clear();
    rc_eps_cpu.clear();

    if (dummy_mr) {
        void *buf = dummy_mr->addr;
        state_->ftable.dereg_mr(dummy_mr);
        free(buf);
        dummy_mr = nullptr;
        dummy_mr_buf = nullptr;
        dummy_mr_lkey = 0;
    }

    if (ah) {
        int ret = doca_verbs_ah_attr_destroy(ah);
        if (ret) {
            NVSHMEMI_WARN_PRINT("doca_verbs_ah_attr_destroy failed: %d\n", ret);
        }
    }
    if (qp_local_backup_gpu) {
        int ret = doca_gpu_verbs_destroy_qp_hl(qp_local_backup_gpu);
        if (ret) {
            NVSHMEMI_WARN_PRINT("doca_gpu_verbs_destroy_qp_hl failed for qp_local_backup_gpu\n");
        }
    }
    if (qp_local_backup_cpu) {
        int ret = doca_gpu_verbs_destroy_qp_hl(qp_local_backup_cpu);
        if (ret) {
            NVSHMEMI_WARN_PRINT("doca_gpu_verbs_destroy_qp_hl failed for qp_local_backup_cpu\n");
        }
    }
    if (net_dev) {
        int ret = doca_verbs_dev_close(net_dev);
        if (ret) {
            NVSHMEMI_WARN_PRINT("doca_verbs_dev_close failed: %d\n", ret);
        }
    }
    if (state_) {
        if (common_device.pd) {
            int ret = state_->ftable.dealloc_pd(common_device.pd);
            if (ret) {
                INFO(state_->log_level, "ibv_dealloc_pd failed Err: %d:%s.\n", errno,
                     strerror(errno));
            }
        }
        if (common_device.context) {
            int ret = state_->ftable.close_device(common_device.context);
            if (ret) {
                NVSHMEMI_WARN_PRINT("ibv_close_device failed Err: %d:%s.\n", errno,
                                    strerror(errno));
            }
        }
    }
}

// nvshmemt_gpunetio_state_t
int nvshmemt_gpunetio_state_t::init_populate_state(nvshmemi_options_s *options) {
    int status = 0;
    log_level = nvshmemt_common_get_log_level(options);
    skip_cst = true;  // will be set to false if multiple devices are selected or if
                      // CST is required for a device
    initialize_cache_state();

    qp_depth = options->QP_DEPTH;
    if (qp_depth > 0) {
        qp_depth = gpunetio_round_up_pow2_or_0(qp_depth);
    }
    if (qp_depth <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH must be a positive number.\n");
    } else if (qp_depth < NVSHMEMI_GPUNETIO_MIN_QP_DEPTH) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH must be at least %d.\n",
                           NVSHMEMI_GPUNETIO_MIN_QP_DEPTH);
    } else if (qp_depth > NVSHMEMI_GPUNETIO_MAX_QP_DEPTH) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_QP_DEPTH can be at most %d.\n", NVSHMEMI_GPUNETIO_MAX_QP_DEPTH);
    }

    num_requests_in_batch = options->GPUNETIO_NUM_REQUESTS_IN_BATCH;
    if (num_requests_in_batch > 0) {
        num_requests_in_batch = gpunetio_round_up_pow2_or_0(num_requests_in_batch);
    }
    if (num_requests_in_batch <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH must be a positive number.\n");
    } else if (num_requests_in_batch > qp_depth) {
        NVSHMEMI_ERROR_RET(
            status, NVSHMEMX_ERROR_INVALID_VALUE,
            "NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH must not be larger than QP depth.\n");
    }

    num_fetch_slots_per_rc = options->GPUNETIO_NUM_FETCH_SLOTS_PER_RC;
    if (num_fetch_slots_per_rc > 0)
        num_fetch_slots_per_rc = gpunetio_round_up_pow2(num_fetch_slots_per_rc);
    if (num_fetch_slots_per_rc <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "NVSHMEM_GPUNETIO_NUM_FETCH_SLOTS_PER_RC must be a positive number.\n");
    }
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_state_t::init_ftables(nvshmemi_options_s *options,
                                            nvshmemi_cuda_fn_table *table) {
    int status = 0;

    cuda_syms = table;
    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, log_level)) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                           "Unable to dlopen libibverbs. Skipping DOCA transport.\n");
    }

    nvshmemt_ib_common_init_mlx5dv(&mlx5dv_handle, &mlx5dv_ftable,
                                   options->DISABLE_DATA_DIRECT, log_level);

    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_state_t::init_gpu(nvshmemi_options_s *options) {
    CUdevice gpu_device_id;
    int status = 0;
    int lowest_stream_priority;
    int highest_stream_priority;

    status = CUPFN(cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        return status;
    }

    char pci_bus_id[MAX_GPU_PCI_ADDRESS_LEN];
    CUDA_RUNTIME_CHECK_RET(
        cudaDeviceGetPCIBusId(pci_bus_id, MAX_GPU_PCI_ADDRESS_LEN, gpu_device_id),
        NVSHMEMX_ERROR_INTERNAL);
    INFO(log_level, "Creating DOCA GPU device handler for GPU with bus ID: %s\n", pci_bus_id);
    status = doca_gpu_create(pci_bus_id, &gpu_device);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "doca_gpu_create failed to create handler for GPU with bus ID: %s\n",
                          pci_bus_id);

    status = nvshmemt_ib_common_check_dmabuf_support(dmabuf_support_for_data_buffers, cuda_syms,
                                                     options->IB_DISABLE_DMABUF);
    if (status) return status;

    CUDA_RUNTIME_CHECK_RET(
        cudaDeviceGetStreamPriorityRange(&lowest_stream_priority, &highest_stream_priority),
        NVSHMEMX_ERROR_INTERNAL);
    CUDA_RUNTIME_CHECK_RET(
        cudaStreamCreateWithPriority(&my_stream, cudaStreamNonBlocking, highest_stream_priority),
        NVSHMEMX_ERROR_INTERNAL);

    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_state_t::init_nic_devices(nvshmem_transport *transport,
                                                nvshmemi_options_s *options) {
    int num_devices = 0;
    ibv_device **dev_list = nullptr;
    int status = 0;
    doca_gpu_dev_verbs_nic_handler nic_handler_request = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    uint32_t atomic_host_endian_size = 0;

    status =
        gpunetio_parse_nic_handler_request(&nic_handler_request, options->GPUNETIO_NIC_HANDLER);
    NVSHMEMI_NZ_ERROR_RET(status, status, "NVSHMEM_GPUNETIO_NIC_HANDLER is not valid.\n");
    INFO(log_level, "NVSHMEM_GPUNETIO_NIC_HANDLER requested: %d\n", nic_handler_request);

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_RET(dev_list, status, NVSHMEMX_ERROR_INTERNAL, "get_device_list failed \n");
    INFO(log_level, "Found %d devices\n", num_devices);

    struct nvshmemt_ib_hca_filter hca_filter = {};
    struct nvshmemt_ib_common_state temp_state = {};
    temp_state.options = options;
    temp_state.log_level = log_level;
    status = nvshmemt_ib_common_parse_hca_filter(hca_filter, temp_state);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE, "HCA filter parsing failed.\n");

    struct nvshmemt_ib_common_device common_devs[MAX_NUM_HCAS] = {};
    int temp_dev_ids[MAX_NUM_PES_PER_NODE];
    int temp_port_ids[MAX_NUM_PES_PER_NODE];
    temp_state.devices = common_devs;
    temp_state.dev_ids = temp_dev_ids;
    temp_state.port_ids = temp_port_ids;

    status = nvshmemt_ib_common_enumerate_devices(
        &ftable, temp_state, sizeof(nvshmemt_ib_common_device), hca_filter, dev_list, num_devices);
    if (status) return status;

    {
        bool device_checked[MAX_NUM_HCAS] = {};
        int write_idx = 0;
        for (int i = 0; i < temp_state.n_dev_ids; i++) {
            int dev_idx = temp_state.dev_ids[i];
            struct nvshmemt_ib_common_device *dev = &common_devs[dev_idx];

            if (!device_checked[dev_idx]) {
                device_checked[dev_idx] = true;
                const char *name = ftable.get_device_name(dev->dev);

                if (!nvshmemt_ib_common_query_mlx5_caps(dev->context)) {
                    NVSHMEMI_WARN_PRINT(
                        "device %s is not enumerated as an mlx5 device. Skipping...", name);
                    ftable.close_device(dev->context);
                    if (dev->pd) ftable.dealloc_pd(dev->pd);
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
                    ftable.close_device(dev->context);
                    if (dev->pd) ftable.dealloc_pd(dev->pd);
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
            log_level,
            "no active IB device that supports GPU-initiated communication is found, exiting...\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = nvshmemt_ib_common_discover_pci_paths(
        transport, temp_state, sizeof(nvshmemt_ib_common_device), &ftable, &mlx5dv_ftable);
    if (status) return status;

    // We need to copy the detected devices into the vector-based structures of gpunetio_state.
    std::array<int, MAX_NUM_HCAS> dev_remap;
    dev_remap.fill(-1);
    for (int i = 0; i < temp_state.n_dev_ids; i++) {
        int dev_id = temp_state.dev_ids[i];
        if (dev_remap[dev_id] == -1) {
            auto new_dev = gpunetio_device::make(this);
            new_dev->common_device = common_devs[dev_id];
            devices.push_back(std::move(new_dev));
            dev_remap[dev_id] = static_cast<int>(devices.size() - 1);
        }
        dev_ids.push_back(dev_remap[dev_id]);
        port_ids.push_back(temp_state.port_ids[i]);
    }

    // GPU data path is only active when the user explicitly enables GDAKI
    if (!options->GPUNETIO_ENABLE_GDAKI) {
        if (options->GPUNETIO_NUM_RC_PER_PE_GPU_provided &&
            options->GPUNETIO_NUM_RC_PER_PE_GPU > 0) {
            INFO(log_level,
                 "NVSHMEM_GPUNETIO_ENABLE_GDAKI is not set; ignoring "
                 "NVSHMEM_GPUNETIO_NUM_RC_PER_PE_GPU=%d and disabling GPU data path QPs",
                 options->GPUNETIO_NUM_RC_PER_PE_GPU);
        }
        options->GPUNETIO_NUM_RC_PER_PE_GPU = 0;
    }

    for (int dev_id : dev_ids) {
        auto &device = *devices[dev_id];
        // Need 8 QPs for achieving bandwidth in data direct device
        if (options->GPUNETIO_ENABLE_GDAKI && device.common_device.data_direct &&
            !options->GPUNETIO_NUM_RC_PER_PE_GPU_provided) {
            options->GPUNETIO_NUM_RC_PER_PE_GPU = 8;
            INFO(log_level,
                 "Setting GPUNETIO_NUM_RC_PER_PE_GPU = 8 as data direct device is detected");
        }

        // Report whether we need to do atomic endianness conversions on 8 byte operands.
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device.common_device.context);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");

        device.nic_handler_request = nic_handler_request;
    }

    if (options->GPUNETIO_NUM_RC_PER_PE_GPU < 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "GPUNETIO_NUM_RC_PER_PE_GPU must be >= 0");
    }
    // The CPU data path is always active in the gpunetio transport, so the CPU RC count must
    // be strictly positive regardless of whether GDAKI is enabled.
    if (options->GPUNETIO_NUM_RC_PER_PE_CPU <= 0) {
        NVSHMEMI_ERROR_RET(status, NVSHMEMX_ERROR_INVALID_VALUE,
                           "GPUNETIO_NUM_RC_PER_PE_CPU must be > 0 (the CPU data path is "
                           "always active in the gpunetio transport)");
    }

    transport->atomic_host_endian_min_size = atomic_host_endian_size;

    // Open NIC in GPUNetIO
    for (auto &device : devices) {
        status = device->open_net_dev();
        if (status) return status;
    }

    return NVSHMEMX_SUCCESS;
}

std::pair<std::unique_ptr<nvshmemt_gpunetio_state_t>, nvshmemx_status>
nvshmemt_gpunetio_state_t::make(nvshmem_transport *transport, nvshmemi_options_s *options,
                                nvshmemi_cuda_fn_table *table) {
    std::unique_ptr<nvshmemt_gpunetio_state_t> state(new nvshmemt_gpunetio_state_t());

    int status = state->init_populate_state(options);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Failed while parsing options.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    status = state->init_ftables(options, table);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Failed to initialize ftables.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    status = state->init_gpu(options);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Failed to get and initialize GPU.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    status = state->init_nic_devices(transport, options);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Failed to parse and select NICs.\n");
        return {nullptr, NVSHMEMX_ERROR_INTERNAL};
    }

    return {std::move(state), NVSHMEMX_SUCCESS};
}

int nvshmemt_gpunetio_state_t::get_cuda_device_id(CUdevice *out) {
    if (CUPFN(cuda_syms, cuCtxGetDevice(out))) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetDevice failed.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return NVSHMEMX_SUCCESS;
}

void nvshmemt_gpunetio_state_t::initialize_cache_state() {
    connect_endpoints_first_call = true;
    last_device_index = 0;
    cur_gpu_qp_index = 0;
    cur_cpu_qp_index = 0;
    first_user_cpu_qp = -1;
    user_cpu_qps.clear();
    last_num_rcs = 0;
    qp_h.clear();
}

// Phase 1: One-time global setup
int nvshmemt_gpunetio_state_t::connect_global_setup(int num_selected_devs, int *selected_dev_ids) {
    int status = get_cuda_device_id(&cached_gpu_device_id);
    if (status) return status;

    // Populate selected device IDs if not already done
    if (this->selected_dev_ids.empty()) {
        this->selected_dev_ids.resize(num_selected_devs);
    }

    // Validate device IDs
    for (int i = 0; i < num_selected_devs; i++) {
        if (selected_dev_ids[i] < 0 || selected_dev_ids[i] >= static_cast<int>(dev_ids.size())) {
            NVSHMEMI_ERROR_PRINT("Invalid device ID %d.\n", selected_dev_ids[i]);
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        this->selected_dev_ids[i] = dev_ids[selected_dev_ids[i]];
    }

    return NVSHMEMX_SUCCESS;
}

gpunetio_ep *nvshmemt_gpunetio_state_t::get_next_cpu_ep(int pe, int qp_index, int n_pes, int mype) {
    if (qp_index == NVSHMEMX_QP_HOST) {
        // A single host EP per PE lives on selected_dev_ids[0]; skip the device round-robin
        int dev_idx = selected_dev_ids[0];
        return devices[dev_idx]->get_cpu_ep_from_qp_index(pe, qp_index, n_pes, mype);
    }

    if (qp_index > NVSHMEMX_QP_DEFAULT && first_user_cpu_qp >= 0 && qp_index >= first_user_cpu_qp &&
        qp_index < cur_cpu_qp_index) {
        int idx = (qp_index - first_user_cpu_qp) / n_pes;
        auto [dev, slot] = user_cpu_qps[idx];
        // Finally return the EP mapped to this index
        return devices[dev]->rc_eps_cpu[slot * n_pes + pe].get();
    }

    int n_devs = static_cast<int>(selected_dev_ids.size());
    int dev_idx = selected_dev_ids[cpu_dev_rr % n_devs];
    cpu_dev_rr++;
    return devices[dev_idx]->get_cpu_ep_from_qp_index(pe, qp_index, n_pes, mype);
}

// Populate and copy over state to GPU
int nvshmemt_gpunetio_state_t::setup_gpu_state(nvshmem_transport_t t) {
    int status = 0;

    // Calculate total RC handle count across all devices first, before
    // dereferencing type_specific_shared_state which may be null when GDAKI=0.
    int num_rc_handles = 0;
    int n_devs_selected = static_cast<int>(selected_dev_ids.size());
    for (int dev_idx : selected_dev_ids) {
        gpunetio_device *device = devices[dev_idx].get();
        num_rc_handles += device->num_gpu_eps_per_pe * t->n_pes;
    }
    assert(num_rc_handles == cur_gpu_qp_index);
    INFO(log_level, "num_rc_handles: %d (last_num_rcs: %d)", num_rc_handles, last_num_rcs);

    if (num_rc_handles <= 0) {
        INFO(log_level,
             "setup_gpu_state: no GPU-data-path QPs to publish (num_rc_handles=0); skipping "
             "device-visible QP state\n");
        return NVSHMEMX_SUCCESS;
    }

    auto *gpunetio_device_state_h =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);
    assert(gpunetio_device_state_h != nullptr);
    nvshmemi_gpunetio_device_qp_t *qp_d = gpunetio_device_state_h->globalmem.qps;
    nvshmemi_gpunetio_device_qp_t *qp_d_temp = nullptr;

    auto qp_d_guard = make_scope_guard([&]() {
        if (qp_d && qp_d != gpunetio_device_state_h->globalmem.qps) {
            cudaError_t err = cudaFree(qp_d);
            CUDA_RUNTIME_ERROR_STRING(err);
        }
    });

    doca_gpu_dev_verbs_qp *qp_tmp;

    // Resize host-side QP array
    qp_h.resize(num_rc_handles);

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
            cudaMemcpyAsync(qp_d_temp, qp_d, last_num_rcs * sizeof(nvshmemi_gpunetio_device_qp_t),
                            cudaMemcpyDeviceToDevice, my_stream),
            NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(
            cudaMemcpyAsync(qp_h.data(), qp_d, last_num_rcs * sizeof(nvshmemi_gpunetio_device_qp_t),
                            cudaMemcpyDeviceToHost, my_stream),
            NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(my_stream), NVSHMEMX_ERROR_INTERNAL);
        CUDA_RUNTIME_CHECK_RET(cudaFree(qp_d), NVSHMEMX_ERROR_INTERNAL);
        qp_d = qp_d_temp;
        qp_d_temp = nullptr;
    }

    // Populate QP host array with data from all devices and all eps
    for (size_t i = 0; i < selected_dev_ids.size(); ++i) {
        int dev_idx = selected_dev_ids[i];
        gpunetio_device *device = devices[dev_idx].get();
        int device_num_eps_per_pe = device->num_gpu_eps_per_pe;
        assert(static_cast<int>(device->gpu_qp_handle_bases.size()) >= device_num_eps_per_pe);

        for (int j = 0; j < device_num_eps_per_pe; ++j) {
            for (int k = 0; k < t->n_pes; ++k) {
                int dst_pe = (j * t->n_pes + 1 + t->my_pe + k) % t->n_pes;
                int device_ep_index = j * t->n_pes + dst_pe;
                assert(device->rc_eps_gpu[device_ep_index] != nullptr);

                // Loopback has only one ep per device
                if (dst_pe == t->my_pe && j > 0) {
                    continue;
                }

                // Copy to the right index in the qp_h array holding *all* QPs for all devices
                int global_ep_index = device->gpu_qp_handle_bases[j] + dst_pe;
                assert(global_ep_index < num_rc_handles);

                if (global_ep_index < last_num_rcs) {
                    continue;
                }

                if (doca_gpu_verbs_get_qp_dev(device->rc_eps_gpu[device_ep_index]->qp->qp_gverbs,
                                              &qp_tmp) != DOCA_SUCCESS) {
                    status = NVSHMEMX_ERROR_INTERNAL;
                    return status;
                }
                CUDA_RUNTIME_CHECK_RET(
                    cudaMemcpyAsync(&(qp_h[global_ep_index].qp), qp_tmp,
                                    sizeof(doca_gpu_dev_verbs_qp), cudaMemcpyDefault, my_stream),
                    NVSHMEMX_ERROR_OUT_OF_MEMORY);
                CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(my_stream),
                                       NVSHMEMX_ERROR_OUT_OF_MEMORY);

                TRACE(
                    log_level,
                    "Exported handle %d for PE %d for device %d at global index %d, pointer is %p",
                    device_ep_index, dst_pe, dev_idx, global_ep_index, &(qp_h[global_ep_index].qp));

                qp_h[global_ep_index].ibuf.buf =
                    device->rc_eps_gpu[device_ep_index]->internal_buf->aligned.gpu_ptr;
                qp_h[global_ep_index].ibuf.nslots = num_fetch_slots_per_rc;
                qp_h[global_ep_index].ibuf.lkey =
                    htobe32(device->rc_eps_gpu[device_ep_index]->internal_buf->mem_handle.lkey);
                qp_h[global_ep_index].ibuf.rkey =
                    htobe32(device->rc_eps_gpu[device_ep_index]->internal_buf->mem_handle.rkey);
                qp_h[global_ep_index].dev_idx = static_cast<uint32_t>(i);
            }
        }
    }

    // Allocate device QP array if first time
    if (qp_d == nullptr) {
        CUDA_RUNTIME_CHECK_RET(
            cudaMalloc(&qp_d, num_rc_handles * sizeof(nvshmemi_gpunetio_device_qp_t)),
            NVSHMEMX_ERROR_OUT_OF_MEMORY);
    }
    CUDA_RUNTIME_CHECK_RET(
        cudaMemcpyAsync(qp_d, qp_h.data(), num_rc_handles * sizeof(nvshmemi_gpunetio_device_qp_t),
                        cudaMemcpyDefault, my_stream),
        NVSHMEMX_ERROR_OUT_OF_MEMORY);
    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(my_stream), NVSHMEMX_ERROR_OUT_OF_MEMORY);

    gpunetio_device_state_h->globalmem.qps = qp_d;
    gpunetio_device_state_h->may_skip_cst = skip_cst;
    gpunetio_device_state_h->num_devices_initialized = n_devs_selected;
    gpunetio_device_state_h->num_rc_per_pe = num_rc_handles / n_devs_selected / t->n_pes;
    gpunetio_device_state_h->num_default_rc_per_pe = options->GPUNETIO_NUM_RC_PER_PE_GPU;
    gpunetio_device_state_h->log2_cumem_granularity = t->log2_cumem_granularity;
    gpunetio_device_state_h->num_requests_in_batch = num_requests_in_batch;

    INFO(log_level, "num_rc_per_pe %d num_rc_handles %d n_devs_selected %d n_pes %d",
         gpunetio_device_state_h->num_rc_per_pe, num_rc_handles, n_devs_selected, t->n_pes);

    // QP group switches for load balancing (allocate only once)
    if (gpunetio_device_state_h->globalmem.qp_group_switches == nullptr) {
        int default_num_rc_handles =
            options->GPUNETIO_NUM_RC_PER_PE_GPU * n_devs_selected * t->n_pes;
        if (num_rc_handles == default_num_rc_handles) {
            int num_qp_groups = std::max(num_rc_handles / n_devs_selected / t->n_pes, 2);
            uint8_t *qp_group_switches_d;
            CUDA_RUNTIME_CHECK_RET(cudaMalloc(reinterpret_cast<void **>(&qp_group_switches_d),
                                              num_qp_groups * sizeof(uint8_t)),
                                   NVSHMEMX_ERROR_OUT_OF_MEMORY);
            CUDA_RUNTIME_CHECK_RET(
                cudaMemsetAsync(qp_group_switches_d, 0, num_qp_groups * sizeof(uint8_t), my_stream),
                NVSHMEMX_ERROR_INTERNAL);
            gpunetio_device_state_h->globalmem.qp_group_switches = qp_group_switches_d;
        }
    }

    last_num_rcs = num_rc_handles;

    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(my_stream), NVSHMEMX_ERROR_INTERNAL);

    qp_d_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_gpunetio_state_t::connect_qps_only(nvshmem_transport_t t, int *out_qp_indices,
                                                int num_qps) {
    assert(out_qp_indices != nullptr);

    const bool use_gdaki = options->GPUNETIO_ENABLE_GDAKI;

    for (int i = 0; i < num_qps; i++) {
        out_qp_indices[i] = use_gdaki ? cur_gpu_qp_index : cur_cpu_qp_index;
        if (!use_gdaki && first_user_cpu_qp < 0) first_user_cpu_qp = cur_cpu_qp_index;

        int n_devs_selected = static_cast<int>(selected_dev_ids.size());
        int selected_dev_idx = last_device_index % n_devs_selected;
        int dev_idx = selected_dev_ids[selected_dev_idx];
        int portid = port_ids[selected_dev_idx];
        gpunetio_device &device = *devices[dev_idx];

        if (use_gdaki) {
            int status = device.add_endpoints(t, portid, 1, gpunetio_qp_kind::GPU);
            if (status) {
                NVSHMEMI_ERROR_PRINT("gpunetio_device::add_endpoints (GPU) failed on QP #%d.\n", i);
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
        {
            int status = device.add_endpoints(t, portid, 1, gpunetio_qp_kind::CPU);
            if (status) {
                NVSHMEMI_ERROR_PRINT("gpunetio_device::add_endpoints (CPU) failed on QP #%d.\n", i);
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }

        if (!use_gdaki) {
            user_cpu_qps.push_back({dev_idx, device.num_cpu_eps_per_pe - 1});
        }

        if (device.num_default_cpu_eps_per_pe > 1) has_multiple_cpu_qps = true;
        last_device_index++;
    }

    return setup_gpu_state(t);
}

int nvshmemt_gpunetio_state_t::connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                                 int num_selected_devs, int *out_qp_indices,
                                                 int num_qps) {
    int status = 0;
    int init_dev_cnt = 0;

    if (!connect_endpoints_first_call) {
        // Short path for subsequent calls (QP-specific API)
        return connect_qps_only(t, out_qp_indices, num_qps);
    }

    // Phase 1: Global setup (only on first call)
    status = connect_global_setup(num_selected_devs, selected_dev_ids);
    if (status) {
        NVSHMEMI_ERROR_PRINT("connect_global_setup failed.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    // Phase 2-3: Per-device processing (cached per device)
    for (int i = 0; i < num_selected_devs; i++) {
        int dev_idx = dev_ids[selected_dev_ids[i]];
        gpunetio_device &device = *devices[dev_idx];
        int portid = port_ids[selected_dev_ids[i]];

        status = device.create_ah(portid);
        if (status) {
            NVSHMEMI_ERROR_PRINT("gpunetio_device::create_ah failed.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }

        if (options->GPUNETIO_NUM_RC_PER_PE_GPU > 0) {
            status = device.add_endpoints(t, portid, options->GPUNETIO_NUM_RC_PER_PE_GPU,
                                          gpunetio_qp_kind::GPU);
            if (status) return status;
        }

        // host EP + default-pool QPs
        status = device.add_endpoints(t, portid, options->GPUNETIO_NUM_RC_PER_PE_CPU + 1,
                                      gpunetio_qp_kind::CPU);
        if (status) return status;
        device.num_default_cpu_eps_per_pe = device.num_cpu_eps_per_pe - 1;

        init_dev_cnt++;
    }

    has_multiple_cpu_qps = (options->GPUNETIO_NUM_RC_PER_PE_CPU > 1);

    // Multiple devices break our CST optimizations
    if (init_dev_cnt > 1) {
        skip_cst = false;
    }

    // Phase 4: GPU setup (only once)
    status = setup_gpu_state(t);
    if (status) return status;

    if (init_dev_cnt < num_selected_devs) {
        NVSHMEMI_WARN_PRINT("Failed to initialize all selected devices. Perf may be limited.\n");
    }

    connect_endpoints_first_call = false;

    return status;
}

nvshmemt_gpunetio_state_t::~nvshmemt_gpunetio_state_t() {
    // Tear down devices
    devices.clear();

    if (device_lkeys_d) {
        cudaError_t err = cudaFree(device_lkeys_d);
        CUDA_RUNTIME_ERROR_STRING(err);
        device_lkeys_d = nullptr;
    }
    if (device_rkeys_d) {
        cudaError_t err = cudaFree(device_rkeys_d);
        CUDA_RUNTIME_ERROR_STRING(err);
        device_rkeys_d = nullptr;
    }
    if (my_stream) {
        cudaError_t err = cudaStreamDestroy(my_stream);
        CUDA_RUNTIME_ERROR_STRING(err);
        my_stream = nullptr;
    }
    if (gpu_device) {
        int ret = doca_gpu_destroy(gpu_device);
        if (ret) {
            NVSHMEMI_WARN_PRINT("doca_gpu_destroy failed for device %p\n",
                                static_cast<void *>(gpu_device));
        }
        gpu_device = nullptr;
    }
    if (ibv_handle) {
        nvshmemt_ibv_ftable_fini(&ibv_handle);
    }
    nvshmemt_ib_common_fini_mlx5dv(&mlx5dv_handle);
}

// Transport C wrappers
static int nvshmemt_gpunetio_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                               int num_selected_devs, int *out_qp_indices,
                                               int num_qps) {
    auto *state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    return state->connect_endpoints(t, selected_dev_ids, num_selected_devs, out_qp_indices,
                                    num_qps);
}

static int nvshmemt_gpunetio_can_reach_peer(int *access, nvshmem_transport_pe_info * /*peer_info*/,
                                            nvshmem_transport_t t) {
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);

    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    if (gpunetio_state->options->GPUNETIO_ENABLE_GDAKI) {
        *access |= NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
                   NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;
    }

    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_show_info(nvshmem_transport * /*transport*/, int /*style*/) {
    NVSHMEMI_ERROR_PRINT("gpunetio show info not implemented\n");
    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_finalize(nvshmem_transport_t transport) {
    assert(transport != nullptr);

    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(transport->state);
    auto *gpunetio_device_state_h =
        static_cast<nvshmemi_gpunetio_device_state_t *>(transport->type_specific_shared_state);

    // GPU buffers are reachable only via transport->type_specific_shared_state, destroy them here
    if (gpunetio_device_state_h) {
        if (gpunetio_device_state_h->globalmem.qps) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.qps);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.qps = nullptr;
        }
        if (gpunetio_device_state_h->globalmem.qp_group_switches) {
            cudaError_t err = cudaFree(gpunetio_device_state_h->globalmem.qp_group_switches);
            CUDA_RUNTIME_ERROR_STRING(err);
            gpunetio_device_state_h->globalmem.qp_group_switches = nullptr;
        }
    }

    delete gpunetio_state;
    transport->state = nullptr;

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }
    free(transport);
    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_add_device_remote_mem_handles(nvshmem_transport_t t,
                                                           int transport_stride,
                                                           nvshmem_mem_handle_t *mem_handles,
                                                           uint64_t heap_offset, size_t size) {
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    int n_pes = t->n_pes;
    size_t num_rkeys;

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

    auto *gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);
    if (gpunetio_device_state) {
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
                                rkeys_array_size, cudaMemcpyHostToDevice,
                                gpunetio_state->my_stream),
                NVSHMEMX_ERROR_INTERNAL);

            CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                                   NVSHMEMX_ERROR_INTERNAL);
        }

        gpunetio_device_state->globalmem.rkeys =
            static_cast<nvshmemi_gpunetio_device_key_t *>(gpunetio_state->device_rkeys_d);
    }

    error_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

// Memory handle management start
static int nvshmemt_gpunetio_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf,
                                            size_t length, nvshmem_transport_t t, bool local_only) {
    int status = 0;
    nvshmem_transport_t transport = t;
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(transport->state);

    __be32 device_lkey;
    gpunetio_mem_handle *handle;

    nvshmemi_gpunetio_device_local_only_mhandle_t *device_mhandle_d = nullptr;
    bool did_emplace = false;

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
        gpunetio_device *device =
            gpunetio_state->devices[gpunetio_state->selected_dev_ids[i]].get();
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

    auto *gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);

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
            if (gpunetio_device_state)
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

        if (gpunetio_device_state) {
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

                CUDA_RUNTIME_CHECK_RET(
                    cudaMalloc(&gpunetio_state->device_lkeys_d, lkeys_array_size),
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
    }

    CUDA_RUNTIME_CHECK_RET(cudaStreamSynchronize(gpunetio_state->my_stream),
                           NVSHMEMX_ERROR_INTERNAL);

    error_guard.dismiss();
    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                                nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    nvshmemi_gpunetio_device_state_t *gpunetio_device_state =
        static_cast<nvshmemi_gpunetio_device_state_t *>(t->type_specific_shared_state);

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
        } else if (gpunetio_device_state) {
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

// Host-side RMA functions: progress, quiet, fence, rma, amo

static int nvshmemt_gpunetio_host_progress(nvshmem_transport_t t) {
    nvshmemt_gpunetio_state_t *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(t->state);
    int n_pes = t->n_pes;
    for (int dev_idx : gpunetio_state->selected_dev_ids) {
        gpunetio_state->devices[dev_idx]->progress(n_pes);
    }

    return NVSHMEMX_SUCCESS;
}

static void gpunetio_activate_progress_function(nvshmem_transport_t t) {
    t->host_ops.progress = nvshmemt_gpunetio_host_progress;
    t->no_proxy = false;
}

static int nvshmemt_gpunetio_host_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index) {
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(tcurr->state);
    int n_pes = tcurr->n_pes;

    int pe_begin, pe_end;
    if (pe == NVSHMEMX_PE_ANY) {
        pe_begin = 0;
        pe_end = n_pes;
    } else {
        pe_begin = pe;
        pe_end = pe + 1;
    }

    // User QP: Drain only the specific slot mapped to this QP
    if (qp_index > NVSHMEMX_QP_DEFAULT && gpunetio_state->first_user_cpu_qp >= 0 &&
        qp_index >= gpunetio_state->first_user_cpu_qp &&
        qp_index < gpunetio_state->cur_cpu_qp_index) {
        // User QP: Calculate from QP index to the vector index of user_cpu_qps
        int idx = (qp_index - gpunetio_state->first_user_cpu_qp) / n_pes;
        auto [dev, slot] = gpunetio_state->user_cpu_qps[idx];
        auto &device = gpunetio_state->devices[dev];
        // Finally drain the EP mapped to this index
        for (int pe_idx = pe_begin; pe_idx < pe_end; pe_idx++) {
            auto &ep = device->rc_eps_cpu[slot * n_pes + pe_idx];
            if (!ep) continue;
            while (ep->head_op_id != ep->tail_op_id) {
                device->cpu_progress_locked();
            }
        }
        return NVSHMEMX_SUCCESS;
    }

    for (int dev_idx : gpunetio_state->selected_dev_ids) {
        auto &device = gpunetio_state->devices[dev_idx];
        int num_default = device->num_default_cpu_eps_per_pe;
        int num_total = device->num_cpu_eps_per_pe;

        int slot_begin = 0, slot_end = 0;
        if (qp_index == NVSHMEMX_QP_HOST) {
            slot_begin = 0;
            slot_end = 1;
        } else if (qp_index == NVSHMEMX_QP_DEFAULT) {
            slot_begin = 1;
            slot_end = 1 + num_default;
        } else if (qp_index == NVSHMEMX_QP_ANY) {
            slot_begin = 1;
            slot_end = num_total;
        } else if (qp_index == NVSHMEMX_QP_ALL) {
            slot_begin = 0;
            slot_end = num_total;
        }

        for (int pe_idx = pe_begin; pe_idx < pe_end; pe_idx++) {
            for (int k = slot_begin; k < slot_end; k++) {
                auto &ep = device->rc_eps_cpu[k * n_pes + pe_idx];
                if (!ep) continue;
                while (ep->head_op_id != ep->tail_op_id) {
                    device->cpu_progress_locked();
                }
            }
        }

        // Host QP only lives on selected_dev_ids[0], exit early
        if (qp_index == NVSHMEMX_QP_HOST) break;
    }

    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_host_fence(struct nvshmem_transport *tcurr, int pe, int qp_index,
                                        int is_multi) {
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(tcurr->state);
    bool fence_required = false;
    int n_devs = static_cast<int>(gpunetio_state->selected_dev_ids.size());

    if (qp_index == NVSHMEMX_QP_DEFAULT) {
        fence_required = gpunetio_state->has_multiple_cpu_qps || n_devs > 1;
    } else if (qp_index == NVSHMEMX_QP_ANY || qp_index == NVSHMEMX_QP_ALL) {
        fence_required = gpunetio_state->has_multiple_cpu_qps || n_devs > 1 ||
                         gpunetio_state->first_user_cpu_qp >= 0;
    }

    if (fence_required || is_multi) {
        return nvshmemt_gpunetio_host_quiet(tcurr, pe, qp_index);
    }

    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_host_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                      rma_memdesc_t *remote, rma_memdesc_t *local,
                                      rma_bytesdesc_t bytesdesc, int qp_index) {
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(tcurr->state);
    int status = 0;

    gpunetio_ep *ep = gpunetio_state->get_next_cpu_ep(pe, qp_index, tcurr->n_pes, tcurr->my_pe);
    if (!ep) return NVSHMEMX_SUCCESS;

    auto *qp_cpu = ep->qp->qp_gverbs->qp_cpu;
    uint16_t wqe_bb_idx = ep->wqe_bb_idx;

    status = ep->check_poll_avail(false);
    if (status) return status;

    auto *wqe = reinterpret_cast<gpunetio_rw_wqe *>(
        qp_cpu->sq_wqe_daddr +
        (static_cast<uintptr_t>(wqe_bb_idx) & qp_cpu->sq_wqe_mask) * GPUNETIO_WQE_BB);
    memset(wqe, 0, sizeof(gpunetio_rw_wqe));

    wqe->ctrl.fm_ce_se = GPUNETIO_WQE_CTRL_CQ_UPDATE;
    size_t wqe_size = sizeof(gpunetio_rw_wqe);
    wqe->ctrl.qpn_ds = htobe32(static_cast<uint32_t>(wqe_size / GPUNETIO_WQE_DS) |
                               (static_cast<uint32_t>(qp_cpu->sq_num) << 8));

    if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
        wqe->ctrl.opmod_idx_opcode = htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_READ |
                                             (static_cast<uint32_t>(wqe_bb_idx) << 8));
    } else {
        wqe->ctrl.opmod_idx_opcode = htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE |
                                             (static_cast<uint32_t>(wqe_bb_idx) << 8));
    }

    auto *remote_handle = reinterpret_cast<nvshmemt_ib_common_mem_handle *>(remote->handle);
    wqe->raddr.raddr = htobe64(reinterpret_cast<uintptr_t>(remote->ptr));
    wqe->raddr.rkey = htobe32(remote_handle->rkey);

    if (verb.desc != NVSHMEMI_OP_P) {
        assert(bytesdesc.nelems < (UINT32_MAX / bytesdesc.elembytes));
        wqe->data.data_seg.byte_count =
            htobe32(static_cast<uint32_t>(bytesdesc.nelems * bytesdesc.elembytes));
        auto *local_handle = reinterpret_cast<nvshmemt_ib_common_mem_handle *>(local->handle);
        wqe->data.data_seg.lkey = htobe32(local_handle->lkey);
        wqe->data.data_seg.addr = htobe64(reinterpret_cast<uintptr_t>(local->ptr));
    } else {
        uint32_t bytecount = static_cast<uint32_t>(bytesdesc.nelems * bytesdesc.elembytes);
        wqe->data.data_inl.byte_count = htobe32(bytecount | 0x80000000);
        switch (bytecount) {
            case 8:
                wqe->data.data_inl.data.data_64 = *static_cast<uint64_t *>(local->ptr);
                break;
            case 4:
                wqe->data.data_inl.data.data_32 = *static_cast<uint32_t *>(local->ptr);
                break;
            case 2:
                wqe->data.data_inl.data.data_16 = *static_cast<uint16_t *>(local->ptr);
                break;
            case 1:
                wqe->data.data_inl.data.data_8 = *static_cast<uint8_t *>(local->ptr);
                break;
            default:
                NVSHMEMI_ERROR_PRINT("Invalid length argument to p: %u\n", bytecount);
                return NVSHMEMX_ERROR_INVALID_VALUE;
        }
    }

    assert(wqe_size <= GPUNETIO_WQE_BB);

    ep->wqe_bb_idx++;
    gpunetio_cpu_post_send(ep, 1);

    if (!verb.is_nbi && verb.desc != NVSHMEMI_OP_P) {
        status = ep->check_poll_avail(true);
        if (status) return status;
    }
    return NVSHMEMX_SUCCESS;
}

// Host-side AMO helpers for CPU data-path QPs
static int gpunetio_amo_32(gpunetio_ep *ep, amo_verb_t verb, amo_memdesc_t *remote) {
    gpunetio_device *device = ep->device_;
    auto *qp_cpu = ep->qp->qp_gverbs->qp_cpu;
    uint16_t wqe_bb_idx = ep->wqe_bb_idx;
    uint32_t swap_add_value = static_cast<uint32_t>(remote->val);
    uint32_t compare = static_cast<uint32_t>(remote->cmp);

    int status = ep->check_poll_avail(false);
    if (status) return status;

    auto *remote_handle =
        reinterpret_cast<nvshmemt_ib_common_mem_handle *>(remote->remote_memdesc.handle);

    auto *wqe = reinterpret_cast<gpunetio_atomic_32_wqe *>(
        qp_cpu->sq_wqe_daddr +
        (static_cast<uintptr_t>(wqe_bb_idx) & qp_cpu->sq_wqe_mask) * GPUNETIO_WQE_BB);
    memset(wqe, 0, sizeof(gpunetio_atomic_32_wqe));

    size_t wqe_size = sizeof(gpunetio_atomic_32_wqe);
    wqe->ctrl.fm_ce_se = GPUNETIO_WQE_CTRL_CQ_UPDATE;
    wqe->ctrl.qpn_ds = htobe32(static_cast<uint32_t>(wqe_size / GPUNETIO_WQE_DS) |
                               (static_cast<uint32_t>(qp_cpu->sq_num) << 8));
    wqe->raddr.raddr = htobe64(reinterpret_cast<uintptr_t>(remote->remote_memdesc.ptr));
    wqe->raddr.rkey = htobe32(remote_handle->rkey);

    wqe->data.byte_count = htobe32(4);
    if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
        wqe->data.lkey = htobe32(device->dummy_mr_lkey);
        wqe->data.addr = htobe64(reinterpret_cast<uintptr_t>(device->dummy_mr_buf));
    } else {
        auto *ret_handle = reinterpret_cast<nvshmemt_ib_common_mem_handle *>(remote->ret_handle);
        assert(ret_handle != nullptr);
        wqe->data.lkey = htobe32(ret_handle->lkey);
        wqe->data.addr = htobe64(reinterpret_cast<uintptr_t>(remote->retptr));
    }

    switch (verb.desc) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(1U);
            wqe->fa_seg.field_boundary = 0;
            break;

        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = UINT32_MAX;
            break;

        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_FETCH_ADD:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(swap_add_value);
            wqe->fa_seg.field_boundary = 0;
            break;

        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = htobe32(~swap_add_value);
            break;

        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = htobe32(swap_add_value);
            break;

        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(swap_add_value);
            wqe->fa_seg.field_boundary = UINT32_MAX;
            break;

        case NVSHMEMI_AMO_FETCH:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = 0;
            wqe->fa_seg.field_boundary = 0;
            break;

        case NVSHMEMI_AMO_COMPARE_SWAP:
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = htobe32(compare);
            wqe->cs_seg.compare_mask = UINT32_MAX;
            wqe->cs_seg.swap_mask = UINT32_MAX;
            break;

        default:
            NVSHMEMI_ERROR_PRINT("gpunetio AMO 32: unsupported opcode %d\n", verb.desc);
            return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    assert(wqe_size <= GPUNETIO_WQE_BB);
    ep->wqe_bb_idx++;
    gpunetio_cpu_post_send(ep, 1);

    // TODO: re-enable fetch-atomic completion wait (disabled to match IBDEVX behavior)
    // if (verb.desc >= NVSHMEMI_AMO_END_OF_NONFETCH) {
    //     status = ep->check_poll_avail(true);
    //     if (status) return status;
    // }
    return NVSHMEMX_SUCCESS;
}

static int gpunetio_amo_64(gpunetio_ep *ep, amo_verb_t verb, amo_memdesc_t *remote) {
    gpunetio_device *device = ep->device_;
    auto *qp_cpu = ep->qp->qp_gverbs->qp_cpu;
    uint16_t wqe_bb_idx = ep->wqe_bb_idx;

    int status = ep->check_poll_avail(false, /*min_free_slots=*/2);
    if (status) return status;

    auto *remote_handle =
        reinterpret_cast<nvshmemt_ib_common_mem_handle *>(remote->remote_memdesc.handle);

    doca_gpunetio_ib_mlx5_wqe_ctrl_seg *ctrl = nullptr;
    doca_gpunetio_ib_mlx5_wqe_raddr_seg *raddr = nullptr;
    doca_gpunetio_ib_mlx5_wqe_data_seg *data = nullptr;
    size_t wqe_size = 0;

    auto *wqe_1 = reinterpret_cast<void *>(
        qp_cpu->sq_wqe_daddr +
        (static_cast<uintptr_t>(wqe_bb_idx) & qp_cpu->sq_wqe_mask) * GPUNETIO_WQE_BB);
    auto *wqe_2 = reinterpret_cast<void *>(
        qp_cpu->sq_wqe_daddr +
        (static_cast<uintptr_t>(wqe_bb_idx + 1) & qp_cpu->sq_wqe_mask) * GPUNETIO_WQE_BB);

    memset(wqe_1, 0, GPUNETIO_WQE_BB);
    memset(wqe_2, 0, GPUNETIO_WQE_BB);

    ctrl = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_ctrl_seg *>(wqe_1);
    raddr = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_raddr_seg *>(
        static_cast<char *>(wqe_1) + sizeof(doca_gpunetio_ib_mlx5_wqe_ctrl_seg));

    auto *cs_data = reinterpret_cast<gpunetio_atomic_64_masked_compare_swap_seg *>(
        static_cast<char *>(wqe_1) + sizeof(doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
        sizeof(doca_gpunetio_ib_mlx5_wqe_raddr_seg));
    auto *cs_mask = reinterpret_cast<gpunetio_atomic_64_masked_compare_swap_seg *>(
        reinterpret_cast<char *>(cs_data) + sizeof(gpunetio_atomic_64_masked_compare_swap_seg));

    raddr->raddr = htobe64(reinterpret_cast<uintptr_t>(remote->remote_memdesc.ptr));
    raddr->rkey = htobe32(remote_handle->rkey);

    switch (verb.desc) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            auto *wqe = reinterpret_cast<gpunetio_atomic_64_masked_fa_wqe *>(wqe_1);
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(1ULL);
            wqe->fa_seg.field_boundary = 0;
            data = &wqe->data;
            wqe_size = sizeof(gpunetio_atomic_64_masked_fa_wqe);
            break;
        }

        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET:
            ctrl->opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = UINT64_MAX;
            data = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_data_seg *>(wqe_2);
            wqe_size = 80;
            break;

        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            auto *wqe = reinterpret_cast<gpunetio_atomic_64_masked_fa_wqe *>(wqe_1);
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(remote->val);
            wqe->fa_seg.field_boundary = 0;
            data = &wqe->data;
            wqe_size = sizeof(gpunetio_atomic_64_masked_fa_wqe);
            break;
        }

        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND:
            ctrl->opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = htobe64(~remote->val);
            data = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_data_seg *>(wqe_2);
            wqe_size = 80;
            break;

        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR:
            ctrl->opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = htobe64(remote->val);
            data = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_data_seg *>(wqe_2);
            wqe_size = 80;
            break;

        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            auto *wqe = reinterpret_cast<gpunetio_atomic_64_masked_fa_wqe *>(wqe_1);
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(remote->val);
            wqe->fa_seg.field_boundary = UINT64_MAX;
            data = &wqe->data;
            wqe_size = sizeof(gpunetio_atomic_64_masked_fa_wqe);
            break;
        }

        case NVSHMEMI_AMO_FETCH: {
            auto *wqe = reinterpret_cast<gpunetio_atomic_64_masked_fa_wqe *>(wqe_1);
            wqe->ctrl.opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = 0;
            wqe->fa_seg.field_boundary = 0;
            data = &wqe->data;
            wqe_size = sizeof(gpunetio_atomic_64_masked_fa_wqe);
            break;
        }

        case NVSHMEMI_AMO_COMPARE_SWAP:
            ctrl->opmod_idx_opcode =
                htobe32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                        (static_cast<uint32_t>(wqe_bb_idx) << 8) | GPUNETIO_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = htobe64(remote->cmp);
            cs_mask->compare = UINT64_MAX;
            cs_mask->swap = UINT64_MAX;
            data = reinterpret_cast<doca_gpunetio_ib_mlx5_wqe_data_seg *>(wqe_2);
            wqe_size = 80;
            break;

        default:
            NVSHMEMI_ERROR_PRINT("gpunetio AMO 64: unsupported opcode %d\n", verb.desc);
            return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    data->byte_count = htobe32(8);
    if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
        data->lkey = htobe32(device->dummy_mr_lkey);
        data->addr = htobe64(reinterpret_cast<uintptr_t>(device->dummy_mr_buf));
    } else {
        auto *ret_handle = reinterpret_cast<nvshmemt_ib_common_mem_handle *>(remote->ret_handle);
        assert(ret_handle != nullptr);
        data->lkey = htobe32(ret_handle->lkey);
        data->addr = htobe64(reinterpret_cast<uintptr_t>(remote->retptr));
    }

    ctrl->fm_ce_se = GPUNETIO_WQE_CTRL_CQ_UPDATE;
    ctrl->qpn_ds = htobe32(static_cast<uint32_t>(wqe_size / GPUNETIO_WQE_DS) |
                           (static_cast<uint32_t>(qp_cpu->sq_num) << 8));

    if (wqe_size > GPUNETIO_WQE_BB) {
        assert(wqe_size <= GPUNETIO_WQE_BB * 2);
        ep->wqe_bb_idx += 2;
        gpunetio_cpu_post_send(ep, 2);
    } else {
        ep->wqe_bb_idx++;
        gpunetio_cpu_post_send(ep, 1);
    }

    return NVSHMEMX_SUCCESS;
}

static int nvshmemt_gpunetio_host_amo(struct nvshmem_transport *tcurr, int pe, void * /*curetptr*/,
                                      amo_verb_t verb, amo_memdesc_t *remote,
                                      amo_bytesdesc_t bytesdesc, int qp_index) {
    auto *gpunetio_state = static_cast<nvshmemt_gpunetio_state_t *>(tcurr->state);

    gpunetio_ep *ep = gpunetio_state->get_next_cpu_ep(pe, qp_index, tcurr->n_pes, tcurr->my_pe);
    if (!ep) return NVSHMEMX_SUCCESS;

    if (bytesdesc.elembytes == 4) {
        return gpunetio_amo_32(ep, verb, remote);
    } else if (bytesdesc.elembytes == 8) {
        return gpunetio_amo_64(ep, verb, remote);
    } else {
        NVSHMEMI_ERROR_PRINT("gpunetio AMO: invalid atomic length %d bytes.\n",
                             bytesdesc.elembytes);
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
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

    nvshmemt_ib_common_sanitize_timeout(options.get());
    nvshmemt_ib_common_sanitize_retry_cnt(options.get());

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
    auto [gpunetio_state, state_status] =
        nvshmemt_gpunetio_state_t::make(transport_owner.get(), options.get(), table);
    if (state_status) return state_status;
    transport->state = gpunetio_state.get();

    transport->host_ops.can_reach_peer = nvshmemt_gpunetio_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_gpunetio_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_gpunetio_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_gpunetio_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_gpunetio_show_info;
    transport->host_ops.finalize = nvshmemt_gpunetio_finalize;
    transport->host_ops.rma = nvshmemt_gpunetio_host_rma;
    transport->host_ops.amo = nvshmemt_gpunetio_host_amo;
    transport->host_ops.fence = nvshmemt_gpunetio_host_fence;
    transport->host_ops.quiet = nvshmemt_gpunetio_host_quiet;
    transport->host_ops.enforce_cst = nullptr;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_gpunetio_add_device_remote_mem_handles;
    transport->host_ops.put_signal = nvshmemt_put_signal;
    // We update the progress function in connect_endpoints when we get the NIC handlers from
    // GPUNetIO for the different QPs.
    transport->host_ops.progress = nullptr;
    transport->no_proxy = true;
    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->type = options->GPUNETIO_ENABLE_GDAKI ? NVSHMEM_TRANSPORT_LIB_CODE_GPUNETIO
                                                     : NVSHMEM_TRANSPORT_LIB_CODE_NONE;
    transport->atomics_complete_on_quiet = true;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    gpunetio_state->options = std::move(options);
    gpunetio_state.release();
    *t = transport_owner.release();

    return NVSHMEMX_SUCCESS;
}
