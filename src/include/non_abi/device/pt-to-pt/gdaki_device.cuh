/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_GDAKI_DEVICE_H_
#define _NVSHMEMI_GDAKI_DEVICE_H_

#include <cuda_runtime.h>
#include <cuda/atomic>
#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_common_gpunetio.h"
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/device/common/nvshmemi_batch_rma_pending_qps.cuh"
#include "non_abi/device/common/nvshmemi_region_state.cuh"
#include "non_abi/nvshmem_build_options.h"
#include "utils_device.h"

#ifdef NVSHMEM_GDAKI_DEBUG
#define GDAKI_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define GDAKI_DEBUG_PRINT(...) ((void)0)
#endif

#define NVSHMEMI_GDAKI_PTX_OPTIMIZATION_MFENCE

#define GDAKI_FULL_WARP 0xffffffffU

/* When we exceed a specific number of threads doing quiet
 * we end up with cache thrashing which causes a significant
 * perf hit. TODO: Tune this number for each supported arch.
 */
#define GDAKI_MAX_THREADS_PER_QUIET 32

// MLX5 accepts up to 1 GiB per command
#define GDAKI_MAX_TRANSFER_SIZE 1073741824LLU

#if defined(__clang_llvm_bitcode_lib__) && !defined(__CUDACC__)
// Plain Clang-to-NVPTX bitcode: use address_space(4) for pointer types
#define CONSTANT_ADDRESS_SPACE __attribute__((address_space(4)))
#else
// CUDA mode or nvcc: __constant__ handles memory space, no explicit AS needed
#define CONSTANT_ADDRESS_SPACE
#endif

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
#ifdef __clang_llvm_bitcode_lib__
#define ACCESS_ONCE(x) (*(volatile __typeof__(x) *)&(x))
#else
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif
#endif

/**
 * DO NOT use BSWAP(READ_ONCE(x)) as it could create a bug.
 * BSWAP is a pre-processor function. It will be unrolled to many READ_ONCE.
 */
#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#ifdef __CUDA_ARCH__
#include "gpunetio/doca_gpunetio_device.h"

// Helper functions

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *
    gdaki_get_state() {
    return &nvshmemi_gpunetio_device_state_d;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t gdaki_get_smid() {
    uint32_t smid;
    asm("mov.u32  %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t gdaki_get_ctaid() {
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int gdaki_get_proxy_pe(int pe) {
    if (nvshmemi_device_state_d.enable_rail_opt == 1) {
        return (pe / nvshmemi_device_state_d.node_npes) * nvshmemi_device_state_d.node_npes +
               nvshmemi_device_state_d.node_mype;
    }
    return pe;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_mfence() {
#ifdef NVSHMEMI_GDAKI_PTX_OPTIMIZATION_MFENCE
    // Prevent code reordering from both compiler and GPU
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
    __threadfence_block();
#endif /* NVSHMEMI_GDAKI_PTX_OPTIMIZATION_MFENCE */
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_warp_sync() { __syncwarp(); }

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE size_t
gdaki_cal_transfer_size(size_t req_size, size_t lchunk_size, size_t rchunk_size) {
    return min(static_cast<size_t>(GDAKI_MAX_TRANSFER_SIZE),
               min(req_size, min(rchunk_size, lchunk_size)));
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_reserve_ibuf_slots(nvshmemi_gpunetio_device_qp_t *qp, unsigned long long int num_slots) {
    nvshmemi_gpunetio_device_qp_management_t *mvars = &qp->mvars;
    uint32_t nslots = qp->ibuf.nslots;
    uint64_t base_idx = atomicAdd((unsigned long long int *)&mvars->ibuf.head, num_slots);
    uint64_t idx = base_idx + num_slots;

    // Wait until the slots become available.
    while (
        idx -
            doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &mvars->ibuf.tail) >
        nslots);

    // Prevent the reordering of the above wait loop.
    gdaki_mfence();

    return base_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_release_ibuf(
    nvshmemi_gpunetio_device_qp_t *qp, unsigned long long int base_idx,
    unsigned long long int num_slots) {
    nvshmemi_gpunetio_device_qp_management_t *mvars = &qp->mvars;
    unsigned long long int new_idx = base_idx + num_slots;
    gdaki_mfence();
    // Wait here.
    while (atomicCAS((unsigned long long int *)&mvars->ibuf.tail, (unsigned long long int)base_idx,
                     new_idx) != base_idx);
    gdaki_mfence();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_get_ibuf_addr(nvshmemi_gpunetio_device_qp_t *qp, uint64_t idx) {
    idx = idx & (qp->ibuf.nslots - 1);
    // buf[0] is reserved for non-fetch operations
    return (uint64_t)qp->ibuf.buf + NVSHMEMI_GPUNETIO_IBUF_SLOT_SIZE * (idx + 1);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool gdaki_can_coalesce_warp(
    unsigned int amask, nvshmemi_gpunetio_device_qp_t *qp) {
    int pred_same_qp;

    if (amask != GDAKI_FULL_WARP) {
        return false;
    }

    __match_all_sync(amask, qp->qp.sq_num, &pred_same_qp);
    return pred_same_qp;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool gdaki_can_coalesce_warp_pe(
    unsigned int amask, int pe) {
    int pred_same_pe;

    if (amask != GDAKI_FULL_WARP) {
        return false;
    }

    __match_all_sync(amask, pe, &pred_same_pe);
    return pred_same_pe;
}

// Multiple threads may update get_head concurrently.
// Only the latest one w.r.t. wqe_idx is important.
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void doca_update_get_head(
    nvshmemi_gpunetio_device_qp_t *qp, uint64_t new_get_head) {
    nvshmemi_gpunetio_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_head,
              (unsigned long long int)new_get_head);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void doca_update_get_tail(
    nvshmemi_gpunetio_device_qp_t *qp, uint64_t new_get_tail) {
    nvshmemi_gpunetio_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_tail,
              (unsigned long long int)new_get_tail);
}

template <typename T>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t
gdaki_get_num_wqes_in_atomic(nvshmemi_amo_t amo_op) {
    if (sizeof(T) == 8) {
        // RC
        switch (amo_op) {
            case NVSHMEMI_AMO_SIGNAL:
            case NVSHMEMI_AMO_SIGNAL_SET:
            case NVSHMEMI_AMO_SWAP:
            case NVSHMEMI_AMO_SET:
            case NVSHMEMI_AMO_FETCH_AND:
            case NVSHMEMI_AMO_AND:
            case NVSHMEMI_AMO_FETCH_OR:
            case NVSHMEMI_AMO_OR:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_gpunetio_device_qp_t *
gdaki_get_qp(int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    if (pe == nvshmemi_device_state_d.mype) {
        // Local QP is always at the same slot
        return &state->globalmem.qps[nvshmemi_device_state_d.mype];
    } else {
        int npes = nvshmemi_device_state_d.npes;
        int ndevices_initialized = state->num_devices_initialized;
        uint32_t id = qp_index;
        uint32_t idx;
        uint32_t warpid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>() /
                          nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();

        if (qp_index == NVSHMEMX_QP_DEFAULT) {
            switch (state->rc_map_type) {
                case NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_CTA:
                    id = gdaki_get_ctaid();
                    break;
                case NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_SM:
                    id = gdaki_get_smid();
                    break;
                case NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_WARP:
                    id = gdaki_get_ctaid() *
                             nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_BLOCK>() /
                             nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>() +
                         warpid;
                    break;
                case NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_NONE: {
                    uint32_t num_default_rcs = state->num_default_rc_per_pe * ndevices_initialized;
                    id = num_default_rcs == 1
                             ? 0
                             : (atomicAdd(&state->globalmem.qp_group_switches[0], 1u) + 1) %
                                   num_default_rcs;
                    idx = id * npes + pe;
                    break;
                }
                default:
                    assert(0);
                    break;
            }

            if (state->rc_map_type != NVSHMEMI_GPUNETIO_DEVICE_QP_MAP_TYPE_NONE) {
                // RC QPs are laid out as [NIC][QP slot][PE]. Keep the mapping ID's QP-slot
                // calculation independent from the NIC selected by the round-robin counter.
                uint32_t qp_slot = (id / ndevices_initialized) % state->num_default_rc_per_pe;
                uint32_t dev_idx = 0;
                if (ndevices_initialized > 1) {
                    // Rotate through NICs on each iteration.
                    uint32_t dev_offset =
                        atomicAdd(&state->globalmem.qp_group_switches[qp_slot], 1u) + 1;
                    dev_idx = dev_offset % ndevices_initialized;
                }
                idx = (dev_idx * state->num_default_rc_per_pe + qp_slot) * npes + pe;
            }
        } else if (qp_index == NVSHMEMX_QP_ANY) {
            uint32_t rc_modulo = state->num_rc_per_pe * ndevices_initialized;
            // The last slot is used for switching on the QP_ANY group.
            uint32_t qp_switch_group = state->num_qp_groups;
            // Rotate through all RC QPs. With the QP-specific API, nvshmemx_create_qp creates
            // QPs on a specific NIC on each call, so the NIC is selected in a round-robin manner.
            id = (atomicAdd(&state->globalmem.qp_group_switches[qp_switch_group], 1u) + 1) %
                 rc_modulo;
            idx = id * npes + pe;
        } else {
            idx = id + pe;
        }

        return &state->globalmem.qps[idx];
    }
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_mix_amo_target(uint64_t value) {
    // Stateless SplitMix64 finalizer for the target PE and symmetric-heap offset.
    value ^= value >> NVSHMEMI_SPLITMIX64_SHIFT_1;
    value *= NVSHMEMI_SPLITMIX64_MULTIPLIER_1;
    value ^= value >> NVSHMEMI_SPLITMIX64_SHIFT_2;
    value *= NVSHMEMI_SPLITMIX64_MULTIPLIER_2;
    return value ^ (value >> NVSHMEMI_SPLITMIX64_SHIFT_3);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool gdaki_use_address_stable_amo(
    int pe, nvshmemx_qp_handle_t qp_index) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    return state->use_address_stable_amo && qp_index == NVSHMEMX_QP_DEFAULT &&
           state->num_devices_initialized > 1 && pe != nvshmemi_device_state_d.mype;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_gpunetio_device_qp_t *
gdaki_get_amo_qp(int pe, uint64_t remote_addr,
                 nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    if (!gdaki_use_address_stable_amo(pe, qp_index)) {
        return gdaki_get_qp(pe, qp_index);
    }

    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    uint64_t remote_offset = remote_addr - (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t target = (static_cast<uint64_t>(static_cast<uint32_t>(pe)) << 32) ^ remote_offset;
    uint64_t hash = gdaki_mix_amo_target(target);
    uint32_t num_selected_slots = state->num_devices_initialized;
    uint32_t num_default_qps_per_slot = state->num_default_rc_per_pe;
    assert(num_default_qps_per_slot > 0);
    uint32_t selected_dev_slot = hash % num_selected_slots;
    uint32_t default_qp_slot = (hash / num_selected_slots) % num_default_qps_per_slot;
    uint32_t qp_idx = (selected_dev_slot * num_default_qps_per_slot + default_qp_slot) *
                          nvshmemi_device_state_d.npes +
                      pe;
    return &state->globalmem.qps[qp_idx];
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_get_lkey(
    uint64_t addr, __be32 *lkey, size_t *chunk_size, bool *is_sysmem_scope, uint32_t dev_idx) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t heap_end = heap_start + nvshmemi_device_state_d.heap_size - 1;
    size_t max_len = 1ULL << 30;
    if (heap_start <= addr && addr <= heap_end) {
        // addr in the symmetric heap
        uint64_t idx = ((addr - heap_start) >> state->log2_cumem_granularity) *
                           state->num_devices_initialized +
                       dev_idx;

        if (idx < NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS) {
            CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_key_t *device_key;
            device_key = state->constmem.lkeys + idx;

            assert(addr < device_key->next_addr);

            *lkey = device_key->key;
            *chunk_size = device_key->next_addr - addr;

        } else {
            nvshmemi_gpunetio_device_key_t *device_key;
            device_key = state->globalmem.lkeys + idx - NVSHMEMI_GPUNETIO_MAX_CONST_LKEYS;

            assert(addr < device_key->next_addr);

            *lkey = device_key->key;
            *chunk_size = device_key->next_addr - addr;
        }

        *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
        *is_sysmem_scope = (nvshmemi_device_state_d.symmetric_heap_kind == 1);
        return;
    } else {
        nvshmemi_gpunetio_device_local_only_mhandle_t *mhandle =
            state->globalmem.local_only_mhandle_head;

        while (mhandle) {
            if (mhandle->start <= addr && addr <= mhandle->end) {
                *lkey = mhandle->lkeys[dev_idx];
                *chunk_size = mhandle->end - addr + 1;
                *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
                *is_sysmem_scope = mhandle->is_sysmem_scope;
                return;
            }
            mhandle = mhandle->next;
        }
    }
    // lkey is not found.
    assert(0);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_get_raddr_rkey(
    uint64_t addr, int dst_pe, int proxy_pe, uint64_t *out_raddr, __be32 *out_rkey,
    size_t *out_chunk_size, uint32_t dev_idx) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t roffset = addr - heap_start;
    int npes;
    // nvcc from CUDA12.0 - 12.2 seems to have a bug. It causes
    // nvshmemi_device_state_d.npes to become 0 in this function.
    // WAR: Force reload of nvshmemi_device_state_d.npes. We may reload from L1
    // most of the time, so the performance hit is minimal.
    asm volatile("ld.b32 %0, [%1];" : "=r"(npes) : "l"(&nvshmemi_device_state_d.npes));

    uint64_t idx =
        ((roffset >> state->log2_cumem_granularity) * npes * state->num_devices_initialized) +
        (proxy_pe * state->num_devices_initialized) + dev_idx;
    uint64_t raddr;

    if (idx < NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS) {
        CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_key_t *device_key;
        device_key = state->constmem.rkeys + idx;
        assert(roffset < device_key->next_addr);
        *out_rkey = device_key->key;
        *out_chunk_size = device_key->next_addr - roffset;
    } else {
        nvshmemi_gpunetio_device_key_t *device_key;
        device_key = state->globalmem.rkeys + idx - NVSHMEMI_GPUNETIO_MAX_CONST_RKEYS;
        assert(roffset < device_key->next_addr);
        *out_rkey = device_key->key;
        *out_chunk_size = device_key->next_addr - roffset;
    }
    raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_remote[proxy_pe] + roffset;
    if (dst_pe != proxy_pe) {
        raddr += (dst_pe % nvshmemi_device_state_d.node_npes - nvshmemi_device_state_d.node_mype) *
                 nvshmemi_device_state_d.heap_size;
    }

    *out_raddr = raddr;
}

static_assert(NVSHMEMI_GPUNETIO_MAX_QP_DEPTH <= 32768,
              "static_assert(NVSHMEMI_GPUNETIO_MAX_QP_DEPTH <= 32768) failed");
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_wait_for_slot_availability(
    nvshmemi_gpunetio_device_qp_t *qp, uint64_t wqe_idx) {
    int status = 0;
    uint16_t nwqes = qp->qp.sq_wqe_num;

    // We don't want wqe_idx - nwqes to wraparound.
    if (likely(wqe_idx >= nwqes)) {
        GDAKI_DEBUG_PRINT(
            "gdaki_wait_for_slot_availability thread %d block %d  wqe_idx %ld nwqes %d cqe_ci "
            "%ld\n",
            threadIdx.x, blockIdx.x, wqe_idx, nwqes,
            doca_gpu_dev_verbs_qp_get_cq_sq(&(qp->qp))->cqe_ci);
        status = doca_gpu_dev_verbs_poll_cq_collapsed_at(&(qp->qp), wqe_idx - nwqes);

        if (status) {
            GDAKI_DEBUG_PRINT("gdaki_poll_cq failed with error=%d.\n", status);
        }

        assert(status == 0);
    }
    gdaki_mfence();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_submit_ready(nvshmemi_gpunetio_device_qp_t *qp) {
    uint64_t ready_idx =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->qp.sq_ready_index);
    uint64_t prod_idx =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->qp.sq_wqe_pi);

    if (ready_idx > prod_idx) {
        doca_gpu_dev_verbs_submit(&(qp->qp), ready_idx,
                                  DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
    }

    return ready_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_quiet(nvshmemi_gpunetio_device_qp_t *qp) {
    uint64_t prod_idx =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->qp.sq_ready_index);

    if (prod_idx == 0) {
        return 0;
    }

    GDAKI_DEBUG_PRINT("gdaki_quiet qp %x thread %d block %d wqe_idx %ld cqe_ci %ld\n",
                      qp->qp.sq_num, threadIdx.x, blockDim.x, prod_idx,
                      doca_gpu_dev_verbs_qp_get_cq_sq(&(qp->qp))->cqe_ci);

    int status = doca_gpu_dev_verbs_poll_cq_collapsed_at(&(qp->qp), prod_idx);
    if (status != DOCA_SUCCESS) {
        GDAKI_DEBUG_PRINT("doca_gpu_dev_verbs_poll_cq_collapsed_at failed %d.\n", status);
    }
    assert(status == DOCA_SUCCESS);

    GDAKI_DEBUG_PRINT("gdaki_quiet exit qp %x thread %d block %d wqe_idx %ld cqe_ci %ld\n",
                      qp->qp.sq_num, threadIdx.x, blockDim.x, prod_idx,
                      doca_gpu_dev_verbs_qp_get_cq_sq(&(qp->qp))->cqe_ci);

    return prod_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_quiet_with_cst(nvshmemi_gpunetio_device_qp_t *qp, bool enforce_cst) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    nvshmemi_gpunetio_device_qp_management_t *mvars = &qp->mvars;

    uint64_t get_head;
    uint64_t ticket;
    uint64_t get_tail;

    if (state->may_skip_cst) {
        ticket = gdaki_quiet(qp);
    } else {
        // We want to read get_head before calling gdaki_quiet. Thus, ticket =
        // gdaki_quiet(qp) cannot be combined.
        get_head =
            doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &mvars->tx_wq.get_head);
        ticket = gdaki_quiet(qp);
        get_tail =
            doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &mvars->tx_wq.get_tail);

        // TODO: Change to WAIT + DUMP
        // In that case, we don't have to do quiet first
        if (get_tail < get_head) {
            doca_update_get_tail(qp, ticket);
        }
    }

    return ticket;
}

__device__ static __forceinline__ void gdaki_atomic_wqe(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr0,
    struct doca_gpu_dev_verbs_wqe *wqe_ptr1, uint16_t wqe_idx, const void *val_1, const void *val_2,
    uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, size_t bytes, nvshmemi_amo_t amo_op,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags) {
    switch (amo_op) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, 1, 0, 0, 0, 0, 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, 1, 0, 0, 0, 0, 0);
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint32_t *)val_1, 0, UINT32_MAX,
                    0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint64_t *)val_1, 0, UINT64_MAX,
                    0);
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, *(uint32_t *)val_1, 0, 0, 0, 0, 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, *(uint64_t *)val_1, 0, 0, 0, 0, 0);
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint32_t *)val_1, 0,
                    ~(*(uint32_t *)val_1), 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint64_t *)val_1, 0,
                    ~(*(uint64_t *)val_1), 0);
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint32_t *)val_1, 0,
                    *(uint32_t *)val_1, 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, wqe_ptr1, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint64_t *)val_1, 0,
                    *(uint64_t *)val_1, 0);
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, *(uint32_t *)val_1, UINT32_MAX, 0, 0, 0,
                    0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, *(uint64_t *)val_1, UINT64_MAX, 0, 0, 0,
                    0);
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, 0, 0, 0, 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, 0, 0, 0, 0);
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA,
                    ctrl_flags, raddr, rkey, laddr, lkey, *(uint32_t *)val_1, 0, 0, 0, 0, 0);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic(
                    qp, wqe_ptr0, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA, ctrl_flags,
                    raddr, rkey, laddr, lkey, 8, *(uint64_t *)val_1, 0);
            }
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            if (bytes == 4) {
                doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
                    qp, wqe_ptr0, nullptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS,
                    ctrl_flags, raddr, rkey, laddr, lkey, 0, 0, *(uint32_t *)val_1,
                    *(uint32_t *)val_2, UINT32_MAX, UINT32_MAX);
            } else {
                doca_gpu_dev_verbs_wqe_prepare_atomic(
                    qp, wqe_ptr0, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_CS, ctrl_flags,
                    raddr, rkey, laddr, lkey, 8, *(uint64_t *)val_2, *(uint64_t *)val_1);
            }
            break;
        }
        default: {
            assert(0);
        }
    }
}

__device__ static __forceinline__ void nvshmemi_gdaki_batch_rma_mark_qp_pending(
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state,
    nvshmemi_gpunetio_device_qp_t *qp, const nvshmemi_region_info_t *region_info) {
    uint32_t qp_count = state->num_rc_per_pe * state->num_devices_initialized *
                        static_cast<uint32_t>(nvshmemi_device_state_d.npes);
    uintptr_t qp_address = reinterpret_cast<uintptr_t>(qp);
    uintptr_t qps_address = reinterpret_cast<uintptr_t>(state->globalmem.qps);
    uint32_t qp_index = static_cast<uint32_t>((qp_address - qps_address) / sizeof(*qp));
    uint32_t region_slot_index = nvshmemi_region_slot_index(region_info);
    if (region_slot_index == UINT32_MAX) {
        return;
    }
    void *pending_qps = nvshmemi_batch_rma_pending_qps_for_slot(state->region_batch_rma_pending_qps,
                                                                qp_count, region_slot_index);
    nvshmemi_batch_rma_mark_qp_pending(pending_qps, qp_count, qp_index);
}

__device__ static __forceinline__ void gdaki_submit_db(
    nvshmemi_gpunetio_device_qp_t *qp, uint64_t base_wqe_idx, uint32_t num_wqes,
    const nvshmemi_region_info_t *region_info = nullptr) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();

    uint64_t mask = ~((uint64_t)(state->num_requests_in_batch - 1));
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    bool no_concurrent_submissions =
        new_wqe_idx ==
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &(qp->qp.sq_rsvd_index));
    bool defer_submission =
        state->region_batch_rma_pending_qps != nullptr &&
        nvshmemi_region_info_has_hints(region_info, NVSHMEMI_REGION_HINT_BATCH_RMA) &&
        nvshmemi_region_slot_index(region_info) != UINT32_MAX;
    bool batch_limit_reached = ((base_wqe_idx & mask) != (new_wqe_idx & mask)) ||
                               (num_wqes >= state->num_requests_in_batch);
    bool do_post_send = batch_limit_reached || (!defer_submission && no_concurrent_submissions);

    if (do_post_send) {
        doca_gpu_dev_verbs_submit(&(qp->qp), new_wqe_idx,
                                  DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);
    } else if (defer_submission) {
        nvshmemi_gdaki_batch_rma_mark_qp_pending(state, qp, region_info);
    }
}

template <nvshmemi_region_operation_t REGION_OPERATION>
__device__ static void gdaki_submit_region_db(nvshmemi_gpunetio_device_qp_t *qp,
                                              uint64_t base_wqe_idx, uint32_t num_wqes) {
    constexpr uint32_t supported_hints =
        nvshmemi_region_operation_traits<REGION_OPERATION>::supported_hints;
    nvshmemi_region_info_t region_info = nvshmemi_region_resolve_active_current(supported_hints);
    const nvshmemi_region_info_t *region = region_info.region_id != 0 ? &region_info : nullptr;
    gdaki_submit_db(qp, base_wqe_idx, num_wqes, region);
}

template <nvshmemi_region_operation_t REGION_OPERATION>
__device__ static __forceinline__ void gdaki_submit_nbi_db(nvshmemi_gpunetio_device_qp_t *qp,
                                                           uint64_t base_wqe_idx,
                                                           uint32_t num_wqes) {
    if constexpr (REGION_OPERATION != NVSHMEMI_REGION_OPERATION_NONE) {
        gdaki_submit_region_db<REGION_OPERATION>(qp, base_wqe_idx, num_wqes);
    } else {
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
    }
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
gdaki_cst(nvshmemi_gpunetio_device_qp_t *qp) {
    const int num_wqes = 1;

    uint64_t base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
    gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);

    struct doca_gpu_dev_verbs_wqe *wqe_ptr =
        doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), base_wqe_idx);

    // DUMP causes the NIC to read GPU memory, which enforces target-side consistency.
    doca_gpu_dev_verbs_wqe_prepare_dump(&(qp->qp), wqe_ptr, base_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                        (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char));

    doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, base_wqe_idx);
    gdaki_submit_db(qp, base_wqe_idx, num_wqes);

    return gdaki_quiet(qp);
}

template <nvshmemi_op_t channel_op, bool nbi,
          nvshmemi_region_operation_t REGION_OPERATION = NVSHMEMI_REGION_OPERATION_NONE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_rma_thread(
    uint64_t rptr, uint64_t lptr, size_t remaining_size, int dst_pe, int proxy_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT,
    nvshmemi_gpunetio_device_qp_t *selected_qp = nullptr) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    unsigned int amask = __activemask();
    bool can_coalesce_warp = selected_qp == nullptr && gdaki_can_coalesce_warp_pe(amask, proxy_pe);
    int my_tid;
    int tg_size;

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;

    nvshmemi_gpunetio_device_qp_t *qp;

    if (selected_qp != nullptr) {
        qp = selected_qp;
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    } else if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_qp(proxy_pe, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        qp = gdaki_get_qp(proxy_pe, qp_index);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    const bool need_additional_wqe = need_immediate_cst;
    int num_wqes_per_cmd = 1;  // RC only 1 WQE

    bool did_quiet = false;

    if (unlikely(remaining_size == 0)) {
        return;
    }

    while (remaining_size > 0) {
        amask = __activemask();

        bool is_data_buf_in_sysmem;

        __be32 lkey;
        size_t lchunk_size;
        gdaki_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);

        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        gdaki_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

        size_t transfer_size = gdaki_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);

        can_coalesce_warp = selected_qp == nullptr && gdaki_can_coalesce_warp(amask, qp);
        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

        uint64_t base_wqe_idx;

        if (my_tid == 0) {
            base_wqe_idx =
                doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
            GDAKI_DEBUG_PRINT("rma_thread base_wqe_idx %ld num_wqe %d\n", base_wqe_idx, num_wqes);
            gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        }

        uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        // RC always 1 WQE for write/read
        struct doca_gpu_dev_verbs_wqe *wqe_ptr;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);

        // Generate CQE only if we create the last WQE in the group.
        enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
            (!need_additional_wqe && (my_tid == tg_size - 1))
                ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                doca_gpu_dev_verbs_wqe_prepare_write(
                    &(qp->qp), wqe_ptr, my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                    fm_ce_se, 0, raddr, rkey, lptr, lkey, transfer_size);
                break;
            case NVSHMEMI_OP_GET:
                doca_gpu_dev_verbs_wqe_prepare_read(&(qp->qp), wqe_ptr, my_wqe_idx, fm_ce_se, raddr,
                                                    rkey, lptr, lkey, transfer_size);
                break;
            default:
                GDAKI_DEBUG_PRINT("Unsupported channel_op.\n");
                assert(0);
        }

        if (can_coalesce_warp) {
            gdaki_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_immediate_cst) {
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
                doca_gpu_dev_verbs_wqe_prepare_dump(
                    &(qp->qp), wqe_ptr, my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE_FENCE,
                    (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char));
            } else {
                if (need_additional_wqe) {
                    my_wqe_idx += num_wqes_per_cmd;
                    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
                    doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptr, my_wqe_idx,
                                                       DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
                }

                if (need_cst) {
                    // For nbi, we will do CST in QUIET.
                    // GET index must be visible before the new cons index.
                    doca_update_get_head(qp, base_wqe_idx + num_wqes);
                }
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) {
                __threadfence_system();
            }

            doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
            if constexpr (nbi) {
                gdaki_submit_nbi_db<REGION_OPERATION>(qp, base_wqe_idx, num_wqes);
            } else {
                gdaki_submit_db(qp, base_wqe_idx, num_wqes);
            }
        }

        remaining_size -= transfer_size;

        rptr += transfer_size;
        lptr += transfer_size;

        if (can_coalesce_warp) {
            if (!nbi) {
                bool do_coalesce_quiet = __all_sync(amask, remaining_size == 0);
                if (do_coalesce_quiet && my_tid == tg_size - 1) {
                    // CST, if required, has already been enqueued. We simply need to
                    // do gdaki_quiet here.
                    gdaki_quiet(qp);
                }
                did_quiet |= do_coalesce_quiet;
            }
            gdaki_warp_sync();
        }
    }

    if (!nbi && !did_quiet) {
        // CST, if required, has already been enqueued. We simply need to
        // do gdaki_quiet here.
        gdaki_quiet(qp);
    }
}

static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64) failed");
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op, bool nbi,
          nvshmemi_region_operation_t REGION_OPERATION = NVSHMEMI_REGION_OPERATION_NONE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gdaki_rma(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int proxy_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);
    // Use only warp 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;
    bool need_additional_wqe;

    nvshmemi_gpunetio_device_qp_t *qp;

    int num_wqes;
    int num_wqes_per_cmd;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = req_rptr;
    uint64_t lptr = req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem;

    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se;

    if (unlikely(remaining_size == 0)) {
        goto out;
    }

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }
    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = gdaki_get_qp(proxy_pe, qp_index);
    }
    qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);

    need_additional_wqe = need_immediate_cst;
    num_wqes_per_cmd = 1;

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        gdaki_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        gdaki_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = gdaki_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use gdaki_rma_thread to handle it instead.
    if (unlikely(chunk_idx > tg_size)) {
        if (my_tid == 0) {
            gdaki_rma_thread<channel_op, nbi, REGION_OPERATION>(req_rptr, req_lptr, bytes, dst_pe,
                                                                proxy_pe, qp_index);
        }

        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + (need_additional_wqe ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("rma base_wqe_idx %ld num_wqe %d\n", base_wqe_idx, num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    base_wqe_idx = __shfl_sync(GDAKI_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    // Generate CQE only if we create the last WQE in the group.
    fm_ce_se = (!need_additional_wqe && (my_tid == chunk_idx - 1))
                   ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                   : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    if (my_tid < chunk_idx) {
        // RC always 1 WQE for write/read
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                doca_gpu_dev_verbs_wqe_prepare_write(
                    &(qp->qp), wqe_ptr, my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                    fm_ce_se, 0, my_raddr, my_rkey, my_laddr, my_lkey, my_transfer_size);
                break;
            case NVSHMEMI_OP_GET:
                doca_gpu_dev_verbs_wqe_prepare_read(&(qp->qp), wqe_ptr, my_wqe_idx, fm_ce_se,
                                                    my_raddr, my_rkey, my_laddr, my_lkey,
                                                    my_transfer_size);
                break;
            default:
                GDAKI_DEBUG_PRINT("Unsupported channel_op.\n");
                assert(0);
        }
    }

    gdaki_warp_sync();

    if (my_tid == chunk_idx - 1) {
        if (need_immediate_cst) {
            my_wqe_idx += num_wqes_per_cmd;
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_dump(
                &(qp->qp), wqe_ptr, my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE_FENCE,
                (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char));
        } else {
            if (need_additional_wqe) {
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
                doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptr, my_wqe_idx,
                                                   DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
            }

            if (need_cst) {
                // For nbi, we will do CST in QUIET.
                // GET index must be visible before the new cons index.
                doca_update_get_head(qp, base_wqe_idx + num_wqes);
            }
        }

        // Require membar.sys to push data buffer to the point of consistency.
        if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) {
            __threadfence_system();
        }

        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        if constexpr (nbi) {
            gdaki_submit_nbi_db<REGION_OPERATION>(qp, base_wqe_idx, num_wqes);
        } else {
            gdaki_submit_db(qp, base_wqe_idx, num_wqes);
        }
        if (!nbi) {
            // CST, if required, has already been enqueued. We simply need to
            // do gdaki_quiet here.
            gdaki_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

/**
 * RMA P base
 */
static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64) failed");
template <typename T>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_p_impl(
    void *rptr, const T value, int dst_pe, int proxy_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    int my_tid;
    int tg_size;
    nvshmemi_gpunetio_device_qp_t *qp;
    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    unsigned int amask = __activemask();
#ifndef __clang_llvm_bitcode_lib__
    bool can_coalesce_warp = gdaki_can_coalesce_warp_pe(amask, proxy_pe);
#else
    bool can_coalesce_warp = false;
#endif

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_qp(proxy_pe, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        qp = gdaki_get_qp(proxy_pe, qp_index);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }
    gdaki_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                         qp->dev_idx);

    assert(rchunk_size >= sizeof(T));

    int num_wqes = tg_size;
    uint64_t base_wqe_idx;

    if (my_tid == 0) {
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("rma_p_impl base_wqe_idx %ld num_wqe %d polling..\n", base_wqe_idx,
                          num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(GDAKI_FULL_WARP, base_wqe_idx, 0);
    }

    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
        (my_tid == tg_size - 1) ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                                : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    uint64_t my_wqe_idx = base_wqe_idx + my_tid;

    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_write_inl(&(qp->qp), wqe_ptr, my_wqe_idx, fm_ce_se, raddr, rkey,
                                             (uint64_t)&value, sizeof(T));

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (my_tid == tg_size - 1) {
        GDAKI_DEBUG_PRINT(
            "rma_p_impl qpn %x base_wqe_idx %ld my_wqe_idx %ld my_tid %d "
            "block %d thread %d\n",
            qp->qp.sq_num, base_wqe_idx, my_wqe_idx, my_tid, blockIdx.x, threadIdx.x);
        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_p(
    void *rptr, const T value, int dst_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    int proxy_pe = gdaki_get_proxy_pe(dst_pe);
    nvshmemi_gdaki_rma_p_impl<T>(rptr, value, dst_pe, proxy_pe, qp_index);
}

/**
 * RMA G base
 */
template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_gdaki_rma_g_impl(
    void *rptr, int dst_pe, int proxy_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    const bool need_cst = !state->may_skip_cst;

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    T ret;

    nvshmemi_gpunetio_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

#ifndef __clang_llvm_bitcode_lib__
    bool can_coalesce_warp = gdaki_can_coalesce_warp_pe(amask, proxy_pe);
#else
    bool can_coalesce_warp = false;
#endif
    bool can_combine_data = false;
    int pred_contiguous = 0;
    int pred_rkey = 0;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_qp(proxy_pe, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = gdaki_get_qp(proxy_pe, qp_index);
    }
    gdaki_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                         qp->dev_idx);

    if (can_coalesce_warp) {
        __match_all_sync(GDAKI_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)),
                         &pred_contiguous);
        __match_all_sync(GDAKI_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_contiguous && pred_rkey);
    }

    const bool need_additional_wqe = need_cst;

    int num_wqes_per_cmd = 1;

    int num_wqes = (can_combine_data ? num_wqes_per_cmd : num_wqes_per_cmd * tg_size) +
                   (need_additional_wqe ? 1 : 0);

    int num_ibuf_slots = can_coalesce_warp ? 1 : tg_size;

    if (my_tid == 0) {
        base_ibuf_idx = gdaki_reserve_ibuf_slots(qp, num_ibuf_slots);
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("rma_g_impl base_wqe_idx %ld num_wqe %d\n", base_wqe_idx, num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx =
        can_combine_data ? base_wqe_idx : base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = can_coalesce_warp ? base_ibuf_idx : base_ibuf_idx + my_tid;

    // Read WQE with RC always 1 WQE
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);

    uint64_t laddr =
        gdaki_get_ibuf_addr(qp, my_ibuf_idx) + (can_coalesce_warp ? my_tid * sizeof(T) : 0);
    __be32 lkey = qp->ibuf.lkey;

    // Generate CQE only if we create the last WQE in the group.
    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
        (!need_additional_wqe &&
         ((can_combine_data && (my_tid == 0)) || (!can_combine_data && (my_tid == tg_size - 1))))
            ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    if (!can_combine_data) {
        doca_gpu_dev_verbs_wqe_prepare_read(&(qp->qp), wqe_ptr, my_wqe_idx, fm_ce_se, raddr, rkey,
                                            laddr, lkey, sizeof(T));
    } else if (my_tid == 0) {
        doca_gpu_dev_verbs_wqe_prepare_read(&(qp->qp), wqe_ptr, my_wqe_idx, fm_ce_se, raddr, rkey,
                                            laddr, lkey, sizeof(T) * tg_size);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (need_additional_wqe && (my_tid == (tg_size - 1))) {
        my_wqe_idx += num_wqes_per_cmd;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
        fm_ce_se = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE;

        if (need_cst) {
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            doca_gpu_dev_verbs_wqe_prepare_dump(
                &(qp->qp), wqe_ptr, my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE_FENCE,
                // MLX5_WQE_CTRL_CQ_UPDATE 2 << 2 | IBGDA_MLX5_FM_FENCE = 2 << 5,
                (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char));
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptr, my_wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
    }

    if (fm_ce_se > 0) {
        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
        gdaki_quiet(qp);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    ret = READ_ONCE(*(T *)laddr);

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (my_tid == tg_size - 1) {
        gdaki_release_ibuf(qp, base_ibuf_idx, num_ibuf_slots);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    return ret;
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_rma_g(void *rptr, int dst_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    int proxy_pe = gdaki_get_proxy_pe(dst_pe);
    return nvshmemi_gdaki_rma_g_impl<T>(rptr, dst_pe, proxy_pe, qp_index);
}

/**
 * RMA NBI base
 */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_batch_rma_submit_pending_qps(
    const nvshmemi_region_info_t *region_info);

template <nvshmemi_region_operation_t REGION_OPERATION>
__device__ NVSHMEMI_STATIC void nvshmemi_gdaki_submit_region_if_threshold_reached(
    uint32_t threshold) {
    constexpr uint32_t supported_hints =
        nvshmemi_region_operation_traits<REGION_OPERATION>::supported_hints;
    nvshmemi_region_info_t region_info = nvshmemi_region_resolve_active_current(supported_hints);
    if (nvshmemi_region_batch_rma_reached_submission_threshold(&region_info, threshold)) {
        nvshmemi_gdaki_batch_rma_submit_pending_qps(&region_info);
    }
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op,
          nvshmemi_region_operation_t REGION_OPERATION = NVSHMEMI_REGION_OPERATION_NONE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_nbi(
    void *rptr, void *lptr, size_t bytes, int dst_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    int proxy_pe = gdaki_get_proxy_pe(dst_pe);
#ifndef __clang_llvm_bitcode_lib__
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
#else
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) {
#endif
        gdaki_rma_thread<channel_op, true, REGION_OPERATION>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                             dst_pe, proxy_pe, qp_index);
#ifndef __clang_llvm_bitcode_lib__
    } else {
        gdaki_rma<SCOPE, channel_op, true, REGION_OPERATION>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                             dst_pe, proxy_pe, qp_index);
    }
#else
    }
    nvshmemi_threadgroup_sync<SCOPE>();
#endif
    if constexpr (REGION_OPERATION != NVSHMEMI_REGION_OPERATION_NONE) {
        if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0 &&
            unlikely(state->region_batch_rma_threshold != 0)) {
            nvshmemi_gdaki_submit_region_if_threshold_reached<REGION_OPERATION>(
                state->region_batch_rma_threshold);
        }
    }
}

/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma(
    void *rptr, void *lptr, size_t bytes, int dst_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    int proxy_pe = gdaki_get_proxy_pe(dst_pe);
#ifndef __clang_llvm_bitcode_lib__
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
#else
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) {
#endif
        gdaki_rma_thread<channel_op, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe, proxy_pe,
                                            qp_index);
#ifndef __clang_llvm_bitcode_lib__
    } else {
        gdaki_rma<SCOPE, channel_op, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe, proxy_pe,
                                            qp_index);
    }
#else
    }
    nvshmemi_threadgroup_sync<SCOPE>();
#endif
}

/**
 * AMO non-fetch base
 */
template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_amo_nonfetch_impl(
    void *rptr, const T value, int pe, nvshmemi_amo_t op,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;
    nvshmemi_gpunetio_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

#ifndef __clang_llvm_bitcode_lib__
    bool can_coalesce_warp =
        !gdaki_use_address_stable_amo(pe, qp_index) && gdaki_can_coalesce_warp_pe(amask, pe);
#else
    bool can_coalesce_warp = false;
#endif

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_amo_qp(pe, (uint64_t)rptr, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = gdaki_get_amo_qp(pe, (uint64_t)rptr, qp_index);
    }

    gdaki_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = gdaki_get_num_wqes_in_atomic<T>(op);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1);

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

    uint64_t base_wqe_idx;

    if (my_tid == 0) {
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("amo_nonfetch_impl base_wqe_idx %ld num_wqe %d\n", base_wqe_idx,
                          num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
    }

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    struct doca_gpu_dev_verbs_wqe *wqe_ptrs[2];
    wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
    wqe_ptrs[1] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 1);

    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
        (!need_additional_wqe && (my_tid == tg_size - 1))
            ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    gdaki_atomic_wqe(&(qp->qp), wqe_ptrs[0], wqe_ptrs[1], my_wqe_idx, &value, nullptr,
                     (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, raddr, rkey, sizeof(T), op, fm_ce_se);

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }

        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        GDAKI_DEBUG_PRINT(
            "amo_nonfetch_impl pe %d op %d my_tid %d qpn %x base_wqe_idx %ld my_wqe_idx %ld "
            "num_wqes_per_cmd %d num_wqes %d"
            "need_additional_wqe %d\n",
            pe, op, my_tid, qp->qp.sq_num, base_wqe_idx, my_wqe_idx, num_wqes_per_cmd, num_wqes,
            need_additional_wqe);

        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_amo_nonfetch(
    void *rptr, const T value, int pe, nvshmemi_amo_t op,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    /* Float atomics are not supported by GPUNetIO GDAKI because the proxy path is
       not available when GPUNetIO GDAKI is the active transport. */
    assert(!nvshmemi_is_float_type<T>() ||
           (op != NVSHMEMI_AMO_ADD && op != NVSHMEMI_AMO_FETCH_ADD));
    nvshmemi_gdaki_amo_nonfetch_impl<T>(rptr, value, pe, op, qp_index);
}

/**
 * AMO fetch base
 */
template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_amo_fetch_impl(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op,
                              nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    const bool need_cst = !state->may_skip_cst;

    T ret;

    nvshmemi_gpunetio_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

#ifndef __clang_llvm_bitcode_lib__
    bool can_coalesce_warp =
        !gdaki_use_address_stable_amo(pe, qp_index) && gdaki_can_coalesce_warp_pe(amask, pe);
#else
    bool can_coalesce_warp = false;
#endif

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_amo_qp(pe, (uint64_t)rptr, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = gdaki_get_amo_qp(pe, (uint64_t)rptr, qp_index);
    }
    gdaki_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = gdaki_get_num_wqes_in_atomic<T>(op);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1) || need_cst;

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    if (my_tid == 0) {
        base_ibuf_idx = gdaki_reserve_ibuf_slots(qp, tg_size);
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("amo_fetch_impl base_wqe_idx %ld num_wqe %d\n", base_wqe_idx, num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = base_ibuf_idx + my_tid;

    uint64_t laddr = gdaki_get_ibuf_addr(qp, my_ibuf_idx);
    __be32 lkey = qp->ibuf.lkey;

    struct doca_gpu_dev_verbs_wqe *wqe_ptrs[2];
    wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
    wqe_ptrs[1] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 1);

    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
        (!need_additional_wqe && (my_tid == tg_size - 1))
            ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
            : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    gdaki_atomic_wqe(&(qp->qp), wqe_ptrs[0], wqe_ptrs[1], my_wqe_idx, &value, &compare, laddr, lkey,
                     raddr, rkey, sizeof(T), op, fm_ce_se);

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);

            if (need_cst) {
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                doca_gpu_dev_verbs_wqe_prepare_dump(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                                    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE_FENCE,
                                                    (uint64_t)qp->ibuf.buf, qp->ibuf.lkey,
                                                    sizeof(char));
            } else {
                doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                                   DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
            }
        }

        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
        gdaki_quiet(qp);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    ret = READ_ONCE(*(T *)laddr);
    ret = nvshmemi_bswap32_if_4byte(ret);

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    if (my_tid == tg_size - 1) {
        gdaki_release_ibuf(qp, base_ibuf_idx, tg_size);
    }

    if (can_coalesce_warp) {
        gdaki_warp_sync();
    }

    return ret;
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_amo_fetch(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op,
                         nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    /* Float atomics are not supported by GPUNetIO GDAKI because the proxy path is
       not available when GPUNetIO GDAKI is the active transport. */
    assert(!nvshmemi_is_float_type<T>() ||
           (op != NVSHMEMI_AMO_ADD && op != NVSHMEMI_AMO_FETCH_ADD));
    return nvshmemi_gdaki_amo_fetch_impl<T>(rptr, value, compare, pe, op, qp_index);
}

static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 128,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 128) failed");
template <bool is_nbi>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal_thread_impl(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    nvshmemi_gpunetio_device_qp_t *qp;
    size_t lchunk_size;
    size_t rchunk_size;
    size_t sig_rchunk_size;
    uint64_t sig_raddr;
    uint64_t raddr;

    unsigned int amask = __activemask();
    int my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    __be32 lkey;
    __be32 rkey;
    __be32 sig_rkey;

#ifndef __clang_llvm_bitcode_lib__
    bool can_coalesce_warp =
        !gdaki_use_address_stable_amo(pe, qp_index) && gdaki_can_coalesce_warp_pe(amask, pe);
#else
    bool can_coalesce_warp = false;
#endif
    bool is_data_buf_in_sysmem;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = gdaki_get_amo_qp(pe, (uint64_t)sig_rptr, qp_index);
        }
        qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = gdaki_get_amo_qp(pe, (uint64_t)sig_rptr, qp_index);
    }
    gdaki_get_lkey((uint64_t)lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
    gdaki_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
    gdaki_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                         qp->dev_idx);

    const int num_atomic_wqes_per_cmd = gdaki_get_num_wqes_in_atomic<uint64_t>(sig_op);
    const bool need_additional_wqe = (num_atomic_wqes_per_cmd > 1);
    int num_wqes;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se;

    size_t transfer_size = gdaki_cal_transfer_size(bytes, lchunk_size, rchunk_size);
    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    if (transfer_size == bytes) {
        int num_rdma_write_wqes_per_cmd = 1;
        int num_wqes_per_cmd = num_rdma_write_wqes_per_cmd + num_atomic_wqes_per_cmd;
        num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

        if (my_tid == 0) {
            base_wqe_idx =
                doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
            GDAKI_DEBUG_PRINT("put_signal_thread1 base_wqe_idx %ld num_wqe %d\n", base_wqe_idx,
                              num_wqes);
            gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(GDAKI_FULL_WARP, base_wqe_idx, 0);
        }
        my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        struct doca_gpu_dev_verbs_wqe *wqe_ptrs[4];
        wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
        wqe_ptrs[1] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 1);
        wqe_ptrs[2] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 2);
        wqe_ptrs[3] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 3);

        doca_gpu_dev_verbs_wqe_prepare_write(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                             DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE, 0,
                                             raddr, rkey, (uint64_t)lptr, lkey, bytes);

        fm_ce_se = (!need_additional_wqe && (my_tid == tg_size - 1))
                       ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                       : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

        gdaki_atomic_wqe(&(qp->qp), wqe_ptrs[num_rdma_write_wqes_per_cmd],
                         wqe_ptrs[num_rdma_write_wqes_per_cmd + 1],
                         my_wqe_idx + num_rdma_write_wqes_per_cmd, &signal, nullptr,
                         (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey, sizeof(signal),
                         sig_op, fm_ce_se);

        if (my_tid == tg_size - 1) {
            if (need_additional_wqe) {
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
                doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                                   DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
            } else {
                my_wqe_idx += num_rdma_write_wqes_per_cmd;
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (is_data_buf_in_sysmem) {
                __threadfence_system();
            }

            doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
            gdaki_submit_db(qp, base_wqe_idx, num_wqes);
            if (!is_nbi) {
                gdaki_quiet(qp);
            }
        }
    } else {
        if (gdaki_use_address_stable_amo(pe, qp_index)) {
            gdaki_rma_thread<NVSHMEMI_OP_PUT, true>((uintptr_t)rptr, (uintptr_t)lptr, bytes, pe, pe,
                                                    qp_index, qp);
        } else {
            gdaki_rma_thread<NVSHMEMI_OP_PUT, true>((uintptr_t)rptr, (uintptr_t)lptr, bytes, pe,
                                                    pe);
        }

        num_wqes = num_atomic_wqes_per_cmd + (need_additional_wqe ? 1 : 0);

        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("put_signal_thread2 base_wqe_idx %ld num_wqe %d\n", base_wqe_idx,
                          num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);

        my_wqe_idx = base_wqe_idx;

        struct doca_gpu_dev_verbs_wqe *wqe_ptrs[2];
        wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
        wqe_ptrs[1] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 1);

        fm_ce_se = (!need_additional_wqe) ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                                          : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

        gdaki_atomic_wqe(&(qp->qp), wqe_ptrs[0], wqe_ptrs[1], my_wqe_idx, &signal, nullptr,
                         (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey, sizeof(signal),
                         sig_op, fm_ce_se);

        if (need_additional_wqe) {
            my_wqe_idx += num_atomic_wqes_per_cmd;
            wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }

        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
        if (!is_nbi) {
            gdaki_quiet(qp);
        }
    }
}

static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64) failed");
template <threadgroup_t SCOPE, bool is_nbi>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal_impl(
    void *req_rptr, void *req_lptr, size_t bytes, void *sig_rptr, uint64_t signal,
    nvshmemi_amo_t sig_op, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only warp 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();

    nvshmemi_gpunetio_device_qp_t *qp;

    int num_rdma_write_wqes_per_cmd;
    int num_atomic_wqes_per_cmd;
    bool need_additional_wqe;

    int num_wqes;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    struct doca_gpu_dev_verbs_wqe *wqe_ptrs[2];

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = (uint64_t)req_rptr;
    uint64_t lptr = (uint64_t)req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem = false;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = gdaki_get_amo_qp(pe, (uint64_t)sig_rptr, qp_index);
    }

    qp = (nvshmemi_gpunetio_device_qp_t *)__shfl_sync(GDAKI_FULL_WARP, (uintptr_t)qp, 0);

    num_rdma_write_wqes_per_cmd = 1;

    num_atomic_wqes_per_cmd = gdaki_get_num_wqes_in_atomic<uint64_t>(sig_op);

    need_additional_wqe = (num_atomic_wqes_per_cmd > 1);

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        gdaki_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        gdaki_get_raddr_rkey(rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = gdaki_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use nvshmemi_gdaki_put_signal_thread_impl to handle it instead.
    // Note that we need one thread to handle amo.
    if (unlikely(chunk_idx > tg_size - 1)) {
        if (my_tid == 0) {
            nvshmemi_gdaki_put_signal_thread_impl<is_nbi>(req_rptr, req_lptr, bytes, sig_rptr,
                                                          signal, sig_op, pe, qp_index);
        }
        goto out;
    }

    num_wqes = num_rdma_write_wqes_per_cmd * chunk_idx + num_atomic_wqes_per_cmd +
               (need_additional_wqe ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx =
            doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &(qp->qp), num_wqes, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
        GDAKI_DEBUG_PRINT("put_signal base_wqe_idx %ld num_wqe %d\n", base_wqe_idx, num_wqes);
        gdaki_wait_for_slot_availability(qp, base_wqe_idx + num_wqes);
    }

    base_wqe_idx = __shfl_sync(GDAKI_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_rdma_write_wqes_per_cmd);

    wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
    wqe_ptrs[1] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx + 1);

    if (my_tid < chunk_idx) {
        doca_gpu_dev_verbs_wqe_prepare_write(
            &(qp->qp), wqe_ptrs[0], my_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE, 0, my_raddr, my_rkey, my_laddr, my_lkey,
            my_transfer_size);

    } else if (my_tid == chunk_idx) {
        __be32 sig_rkey;
        uint64_t sig_raddr;
        size_t sig_rchunk_size;
        gdaki_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                             qp->dev_idx);

        enum doca_gpu_dev_verbs_wqe_ctrl_flags fm_ce_se =
            (!need_additional_wqe) ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE
                                   : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

        gdaki_atomic_wqe(&(qp->qp), wqe_ptrs[0], wqe_ptrs[1], my_wqe_idx, &signal, nullptr,
                         (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey, sizeof(signal),
                         sig_op, fm_ce_se);

        if (need_additional_wqe) {
            my_wqe_idx += num_atomic_wqes_per_cmd;
            wqe_ptrs[0] = doca_gpu_dev_verbs_get_wqe_ptr(&(qp->qp), my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_nop(&(qp->qp), wqe_ptrs[0], my_wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
    }

    nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == chunk_idx) {
        // Require membar.sys to push data buffer to the point of consistency.
        if (is_data_buf_in_sysmem) {
            __threadfence_system();
        }

        doca_gpu_dev_verbs_mark_wqes_ready(&(qp->qp), base_wqe_idx, my_wqe_idx);
        gdaki_submit_db(qp, base_wqe_idx, num_wqes);
        if (!is_nbi) {
            gdaki_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
#ifndef __clang_llvm_bitcode_lib__
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
#else
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) {
#endif
        if (is_nbi) {
            nvshmemi_gdaki_put_signal_thread_impl<true>(rptr, lptr, bytes, sig_rptr, signal, sig_op,
                                                        pe, qp_index);
        } else {
            nvshmemi_gdaki_put_signal_thread_impl<false>(rptr, lptr, bytes, sig_rptr, signal,
                                                         sig_op, pe, qp_index);
        }
    }
#ifndef __clang_llvm_bitcode_lib__
    else {
        if (is_nbi) {
            nvshmemi_gdaki_put_signal_impl<SCOPE, true>(rptr, lptr, bytes, sig_rptr, signal, sig_op,
                                                        pe, qp_index);
        } else {
            nvshmemi_gdaki_put_signal_impl<SCOPE, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                         sig_op, pe, qp_index);
        }
    }
#endif
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_batch_rma_submit_pending_qps(
    const nvshmemi_region_info_t *region_info) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    uint32_t nrcs = state->num_rc_per_pe * state->num_devices_initialized;
    uint32_t npes = nvshmemi_device_state_d.npes;
    uint32_t qp_count = nrcs * npes;
    nvshmemi_gpunetio_device_qp_t *qps = state->globalmem.qps;
    auto submit_qp = [qps](uint32_t qp_index) { gdaki_submit_ready(&qps[qp_index]); };

    uint32_t region_slot_index = nvshmemi_region_slot_index(region_info);
    if (state->region_batch_rma_pending_qps == nullptr || region_slot_index == UINT32_MAX) {
        return;
    }
    void *pending_qps = nvshmemi_batch_rma_pending_qps_for_slot(state->region_batch_rma_pending_qps,
                                                                qp_count, region_slot_index);
    nvshmemi_batch_rma_submit_pending_qps(pending_qps, qp_count, submit_qp);
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_qp_quiet(
    bool enforce_cst, int pe_hint, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();
    nvshmemi_gpunetio_device_qp_t *qp;
    int npes;
    int start_pe;
    uint32_t nrcs;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();

    scope_size =
        scope_size > GDAKI_MAX_THREADS_PER_QUIET ? GDAKI_MAX_THREADS_PER_QUIET : scope_size;

    if (pe_hint == NVSHMEMX_PE_ANY) {
        npes = nvshmemi_device_state_d.npes;
        start_pe = 0;
    } else {
        npes = 1;
        start_pe = pe_hint;
    }

    if (num_qps == NVSHMEMX_QP_ALL || (qp_handle && qp_handle[0] == NVSHMEMX_QP_ALL)) {
        nrcs = state->num_rc_per_pe * state->num_devices_initialized;
        qp_handle = nullptr;
    } else if (num_qps == NVSHMEMX_QP_DEFAULT ||
               (qp_handle && qp_handle[0] == NVSHMEMX_QP_DEFAULT)) {
        nrcs = state->num_default_rc_per_pe * state->num_devices_initialized;
        qp_handle = nullptr;
    } else {
        nrcs = min(num_qps, state->num_rc_per_pe * state->num_devices_initialized);
    }

    // Match this up with the new qp addition APIs.
    if (index_in_scope < scope_size) {
        for (uint32_t i = index_in_scope; i < nrcs * npes; i += scope_size) {
            int qp_idx = i % nrcs;
            int pe_idx = (i / nrcs) + start_pe;
            if (pe_idx == nvshmemi_device_state_d.mype) {
                continue;
            }
            if (pe_idx < start_pe || pe_idx >= start_pe + npes) {
                continue;
            }

            if (qp_handle == nullptr) {
                qp = &state->globalmem.qps[qp_idx * nvshmemi_device_state_d.npes + pe_idx];
            } else {
                qp = &state->globalmem.qps[qp_handle[qp_idx] + pe_idx];
            }

            gdaki_quiet_with_cst(qp, enforce_cst);
        }
    }

    // Quiet (single) local QP
    if (index_in_scope == 0 &&
        (pe_hint == NVSHMEMX_PE_ANY || pe_hint == nvshmemi_device_state_d.mype)) {
        qp = &state->globalmem.qps[nvshmemi_device_state_d.mype];
        gdaki_quiet_with_cst(qp, enforce_cst);
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_qp_fence(
    int pe_hint, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    // Fence on a single QP is noop, so we can skip it here, otherwise do a quiet
    if (num_qps != 1 || qp_handle == nullptr || qp_handle[0] == NVSHMEMX_QP_DEFAULT ||
        num_qps == NVSHMEMX_QP_ALL) {
        // Fence does not guarantee the completion of prior operations.
        // It is ok for GET to finish without data arrival.
        // Use gdaki_quiet here instead of gdaki_quiet_with_cst since it is cheaper.
        nvshmemi_gdaki_qp_quiet<SCOPE>(false, pe_hint, qp_handle, num_qps);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_enforce_consistency_at_target(
    bool use_membar) {
    CONSTANT_ADDRESS_SPACE nvshmemi_gpunetio_device_state_t *state = gdaki_get_state();

    if (!state->may_skip_cst) {
        int npes = nvshmemi_device_state_d.npes;

        // Run CST on the local loopback QP for each initialized device.
        for (int dev_idx = 0; dev_idx < state->num_devices_initialized; ++dev_idx) {
            int qp_idx =
                dev_idx * state->num_default_rc_per_pe * npes + nvshmemi_device_state_d.mype;
            nvshmemi_gpunetio_device_qp_t *qp = &state->globalmem.qps[qp_idx];
            gdaki_cst(qp);
        }
    }

    if (use_membar) {
        __threadfence_system();
    }
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_GDAKI_DEVICE_H_ */
