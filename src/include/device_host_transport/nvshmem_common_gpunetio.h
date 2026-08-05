/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_GPUNETIO_COMMON_H_
#define _NVSHMEMI_GPUNETIO_COMMON_H_

#define NVSHMEMI_GPUNETIO_QP_MANAGEMENT_PADDING 24
#define NVSHMEMI_GPUNETIO_STATE_PADDING 108

#define NVSHMEMI_GPUNETIO_SCALAR_INVALID -1
#define NVSHMEMI_GPUNETIO_USSCALAR_INVALID 0xFFFF
#define NVSHMEMI_GPUNETIO_USCALAR_INVALID 0xFFFFFFFF
#define NVSHMEMI_GPUNETIO_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFF

#define NVSHMEMI_GPUNETIO_MIN_QP_DEPTH 128
#define NVSHMEMI_GPUNETIO_MAX_QP_DEPTH 32768
#define NVSHMEMI_GPUNETIO_IBUF_SLOT_SIZE 256  // 32 threads * sizeof(uint64_t)

#define NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS 64
#define NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS 64
#define NVSHMEMI_GPUNETIO_MAX_CONST_DCTS 128

/* This is determined by the size of nvshmem_mem_handle_t*/
#define NVSHMEMI_GPUNETIO_MAX_DEVICES_PER_PE 15

#if !defined __CUDACC_RTC__
#include <stddef.h>  // for size_t
#include <stdint.h>  // for uint64_t, uint32_t, uint16_t, uint8_t
#include <limits.h>

#define nvshmemi_init_gpunetio_device_qp_management(qp_man)                            \
    do {                                                                               \
        qp_man.version = (1 << 16) + sizeof(nvshmemi_gpunetio_device_qp_management_t); \
        qp_man.post_send_lock = NVSHMEMI_GPUNETIO_SCALAR_INVALID;                      \
        qp_man.tx_wq.resv_head = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                   \
        qp_man.tx_wq.ready_head = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                  \
        qp_man.tx_wq.prod_idx = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.cons_idx = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.get_head = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.get_tail = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                    \
        qp_man.ibuf.head = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                         \
        qp_man.ibuf.tail = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                         \
    } while (0);

#define nvshmemi_init_gpunetio_device_qp(qp)                            \
    do {                                                                \
        qp.version = (1 << 16) + sizeof(nvshmemi_gpunetio_device_qp_t); \
        qp.dev_idx = NVSHMEMI_GPUNETIO_USCALAR_INVALID;                 \
        qp.ibuf.nslots = NVSHMEMI_GPUNETIO_USCALAR_INVALID;             \
        qp.ibuf.buf = NULL;                                             \
        qp.ibuf.lkey = NVSHMEMI_GPUNETIO_USCALAR_INVALID;               \
        qp.ibuf.rkey = NVSHMEMI_GPUNETIO_USCALAR_INVALID;               \
        qp.qp = NULL;                                                   \
        nvshmemi_init_gpunetio_device_qp_management(qp.mvars);          \
    } while (0);

#define nvshmemi_init_gpunetio_device_local_only_memhandle(mhandle)                          \
    do {                                                                                     \
        mhandle.version = (1 << 16) + sizeof(nvshmemi_gpunetio_device_local_only_mhandle_t); \
        mhandle.is_sysmem_scope = false;                                                     \
        mhandle.start = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                                  \
        mhandle.end = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;                                    \
        mhandle.next = NULL;                                                                 \
        for (int i = 0; i < NVSHMEMI_GPUNETIO_MAX_DEVICES_PER_PE; i++) {                     \
            mhandle.lkeys[i] = NVSHMEMI_GPUNETIO_USCALAR_INVALID;                            \
        }                                                                                    \
    } while (0);

#define nvshmemi_init_gpunetio_device_state(state)                            \
    do {                                                                      \
        state.version = (1 << 16) + sizeof(nvshmemi_gpunetio_device_state_t); \
        state.num_rc_per_pe = NVSHMEMI_GPUNETIO_USCALAR_INVALID;              \
        state.num_requests_in_batch = NVSHMEMI_GPUNETIO_USCALAR_INVALID;      \
        state.log2_cumem_granularity = NVSHMEMI_GPUNETIO_ULSCALAR_INVALID;    \
        state.num_devices_initialized = NVSHMEMI_GPUNETIO_SCALAR_INVALID;     \
        state.num_default_rc_per_pe = NVSHMEMI_GPUNETIO_SCALAR_INVALID;       \
        state.nic_buf_on_gpumem = false;                                      \
        state.may_skip_cst = false;                                           \
        state.use_address_stable_amo = false;                                 \
        state.globalmem.qp_group_switches = NULL;                             \
        state.globalmem.local_only_mhandle_head = NULL;                       \
        state.globalmem.qps = NULL;                                           \
        state.globalmem.lkeys = NULL;                                         \
        state.globalmem.rkeys = NULL;                                         \
        state.extra = NULL;                                                   \
        state.num_qp_groups = NVSHMEMI_GPUNETIO_USCALAR_INVALID;              \
        state.rc_map_type = NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_INVALID;     \
        state.region_batch_rma_pending_qps = NULL;                            \
        state.region_batch_rma_threshold = 0;                                 \
    } while (0);

#else
#include <cuda/std/cstddef>
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include "gpunetio/doca_gpunetio_host.h"
#include <infiniband/mlx5dv.h>  // for mlx5_wqe_av
#include <linux/types.h>        // for __be32

#define NVSHMEMI_GPUNETIO_MIN_MAJOR_VERSION 2
static_assert(DOCA_GPUNETIO_VERSION_MAJOR >= NVSHMEMI_GPUNETIO_MIN_MAJOR_VERSION,
              "GPUNetIO major version too old. NVSHMEM requires DOCA_GPUNETIO_VERSION_MAJOR >= 2 "
              "for ABI compatibility.");

// Variables for queue management.
// They are always in global memory.
struct alignas(8) nvshmemi_gpunetio_device_qp_management_v1 {
    int version;
    int post_send_lock;
    struct {
        // All indexes are in wqebb unit
        uint64_t resv_head;   // last reserved wqe idx + 1
        uint64_t ready_head;  // last ready wqe idx + 1
        uint64_t prod_idx;    // posted wqe idx + 1 (producer index + 1)
        uint64_t cons_idx;    // polled wqe idx + 1 (consumer index + 1)
        uint64_t get_head;    // last wqe idx + 1 with a "fetch" operation (g, get, amo_fetch)
        uint64_t get_tail;    // last wqe idx + 1 polled with cst; get_tail > get_head is possible
    } tx_wq;
    struct {
        uint64_t head;
        uint64_t tail;
    } ibuf;
    char padding[NVSHMEMI_GPUNETIO_QP_MANAGEMENT_PADDING];
};
static_assert(sizeof(nvshmemi_gpunetio_device_qp_management_v1) == 96,
              "gpunetio_device_qp_management_v1 must be 96 bytes.");

typedef nvshmemi_gpunetio_device_qp_management_v1 nvshmemi_gpunetio_device_qp_management_t;

struct nvshmemi_gpunetio_device_qp_v1 {
    int version;
    uint32_t dev_idx;
    struct {
        uint32_t nslots;  // num slots for fetch; always a power of 2
        void *buf;        // first NVSHMEMI_GPUNETIO_IBUF_SLOT_SIZE is for non-fetch
        __be32 lkey;
        __be32 rkey;
    } ibuf;                                           // Internal buffer
    nvshmemi_gpunetio_device_qp_management_v1 mvars;  // management variables
    struct doca_gpu_dev_verbs_qp qp;
};
static_assert(sizeof(nvshmemi_gpunetio_device_qp_v1) == 424,
              "gpunetio_device_qp_v1 must be 424 bytes.");

typedef nvshmemi_gpunetio_device_qp_v1 nvshmemi_gpunetio_device_qp_t;

struct nvshmemi_gpunetio_device_local_only_mhandle_v1 {
    int version;
    bool is_sysmem_scope;
    uint64_t start;
    uint64_t end;
    nvshmemi_gpunetio_device_local_only_mhandle_v1 *next;
    __be32 lkeys[NVSHMEMI_GPUNETIO_MAX_DEVICES_PER_PE];
};
static_assert(sizeof(nvshmemi_gpunetio_device_local_only_mhandle_v1) == 96,
              "gpunetio_device_local_only_mhandle_v1 must be 96 bytes.");

typedef nvshmemi_gpunetio_device_local_only_mhandle_v1
    nvshmemi_gpunetio_device_local_only_mhandle_t;

typedef enum {
    NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_CTA = 0,
    NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_SM,
    NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_WARP,
    NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_NONE,
    NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_INVALID = INT_MAX
} nvshmemi_gpunetio_device_qp_map_type_t;

// This is a stable structure.
struct nvshmemi_gpunetio_device_key_t {
    __be32 key;
    uint64_t next_addr;  // end of this address range + 1
};

struct nvshmemi_gpunetio_device_state_v1 {
    int version;
    uint32_t num_rc_per_pe;
    uint32_t num_requests_in_batch; /* always a power of 2 */
    int num_devices_initialized;
    size_t log2_cumem_granularity;
    int num_default_rc_per_pe;
    bool nic_buf_on_gpumem;
    bool may_skip_cst;
    bool use_address_stable_amo;

    struct {
        // lkeys[idx] gives the lkey of chunk idx.
        nvshmemi_gpunetio_device_key_t lkeys[NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS];
        // rkeys[idx * npes + pe] gives rkey of chunk idx targeting peer pe.
        nvshmemi_gpunetio_device_key_t rkeys[NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS];
    } constmem;

    struct {
        uint32_t *qp_group_switches;
        nvshmemi_gpunetio_device_local_only_mhandle_t *local_only_mhandle_head;
        nvshmemi_gpunetio_device_qp_t *qps;
        // For lkeys that cannot be contained in constmem.lkeys.
        // lkeys[idx - NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS] gives the lkey of chunk idx.
        nvshmemi_gpunetio_device_key_t *lkeys;
        // For rkeys that cannot be contained in constmem.rkeys.
        // rkeys[(idx * npes + pe) - NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS] gives rkey of chunk idx
        // targeting peer pe.
        nvshmemi_gpunetio_device_key_t *rkeys;
    } globalmem;
    void *extra;
    uint32_t num_qp_groups;
    nvshmemi_gpunetio_device_qp_map_type_t rc_map_type;
    void *region_batch_rma_pending_qps;
    uint32_t region_batch_rma_threshold;
    uint8_t reserved[NVSHMEMI_GPUNETIO_STATE_PADDING];
};
static_assert(sizeof(nvshmemi_gpunetio_device_state_v1) == 2256,
              "gpunetio_device_state_v1 must be 2256 bytes.");

typedef nvshmemi_gpunetio_device_state_v1 nvshmemi_gpunetio_device_state_t;

#include "device_host/nvshmemi_extern_constant.h"

#ifdef EXTERN_CONSTANT
EXTERN_CONSTANT nvshmemi_gpunetio_device_state_t nvshmemi_gpunetio_device_state_d;
#undef EXTERN_CONSTANT
#endif
#endif
