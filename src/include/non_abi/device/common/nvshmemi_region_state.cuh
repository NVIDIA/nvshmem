/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_STATE_CUH_
#define _NVSHMEMI_REGION_STATE_CUH_

#include <cuda/atomic>

#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/nvshmemi_region_constants.h"
#include "non_abi/nvshmemi_region_types.h"

#ifdef __CUDA_ARCH__

__device__ __forceinline__ bool nvshmemi_region_hints_are_valid(uint32_t hints) {
    return (hints & ~NVSHMEMI_REGION_SUPPORTED_HINTS) == 0;
}

__device__ __forceinline__ uint32_t nvshmemi_region_probe_start(uint64_t gridid, uint64_t block_id,
                                                                uint32_t slots_len) {
    if (slots_len <= 1) {
        return 0;
    }

    /* grid IDs advance sequentially, so reverse their low-order bits into the high-order table
     * index bits. This spreads concurrent grids across a power-of-two table while adding the flat
     * block index keeps blocks within one grid on distinct adjacent probe starts. */
    constexpr uint32_t gridid_bits = sizeof(gridid) * CHAR_BIT;
    uint32_t slot_bits = static_cast<uint32_t>(__ffs(static_cast<int>(slots_len)) - 1);
    uint32_t grid_base = static_cast<uint32_t>(__brevll(gridid) >> (gridid_bits - slot_bits));
    return (grid_base + static_cast<uint32_t>(block_id)) & (slots_len - 1);
}

constexpr uint64_t NVSHMEMI_REGION_SLOT_PHASE_MASK = 0x3;
constexpr uint64_t NVSHMEMI_REGION_SLOT_GENERATION_MAX = UINT64_MAX >> 2;

__device__ __forceinline__ unsigned long long nvshmemi_region_slot_state(
    nvshmemi_region_slot_state_t phase, uint64_t generation) {
    return (generation << 2) | static_cast<unsigned long long>(phase);
}

__device__ __forceinline__ bool nvshmemi_region_slot_state_is_active(unsigned long long state) {
    return (state & NVSHMEMI_REGION_SLOT_PHASE_MASK) == NVSHMEMI_REGION_SLOT_ACTIVE;
}

__device__ __forceinline__ uint64_t
nvshmemi_region_slot_state_generation(unsigned long long state) {
    return state >> 2;
}

__device__ __forceinline__ bool nvshmemi_region_slot_matches(nvshmemi_region_slot_t *slot,
                                                             unsigned long long state,
                                                             uint64_t gridid, uint64_t block_id,
                                                             uint64_t *generation = nullptr) {
    uint64_t observed_gridid = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->gridid)
                                   .load(cuda::memory_order_acquire);
    uint64_t observed_block = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->block_id)
                                  .load(cuda::memory_order_acquire);
    uint64_t observed_generation =
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->generation)
            .load(cuda::memory_order_relaxed);
    if (observed_generation != nvshmemi_region_slot_state_generation(state) ||
        observed_gridid != gridid || observed_block != block_id) {
        return false;
    }
    if (generation != nullptr) {
        *generation = observed_generation;
    }
    return true;
}

__device__ __forceinline__ bool nvshmemi_region_slot_has_hints(nvshmemi_region_slot_t *slot,
                                                               uint32_t hints) {
    uint32_t slot_hints = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(slot->hints)
                              .load(cuda::memory_order_relaxed);
    return hints == NVSHMEMX_REGION_HINT_NONE || (slot_hints & hints) == hints;
}

__device__ __forceinline__ uint32_t nvshmemi_region_load_active_count() {
    uint32_t *active_count = nvshmemi_device_state_d.region_active_count;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count_ref(*active_count);
    return active_count_ref.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ bool nvshmemi_region_any_active() {
    /* Avoid probing the region table when no active device region can affect an operation. */
    return (nvshmemi_region_load_active_count() & NVSHMEMI_REGION_ACTIVE_COUNT_MASK) != 0;
}

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_find_active_slot(
    uint64_t gridid, uint64_t block_id, uint64_t *generation = nullptr) {
    nvshmemi_region_slot_t *slots = nvshmemi_device_state_d.region_slots;
    uint32_t slots_len = nvshmemi_device_state_d.region_slots_len;
    uint32_t probe_limit = nvshmemi_device_state_d.region_slot_probe_limit;

    if (slots == nullptr || slots_len == 0 || probe_limit == 0) {
        return nullptr;
    }

    uint32_t start = nvshmemi_region_probe_start(gridid, block_id, slots_len);
    for (uint32_t i = 0; i < probe_limit; i++) {
        nvshmemi_region_slot_t *slot = &slots[(start + i) & (slots_len - 1)];
        unsigned long long state =
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(slot->state)
                .load(cuda::memory_order_acquire);
        if (nvshmemi_region_slot_state_is_active(state) &&
            nvshmemi_region_slot_matches(slot, state, gridid, block_id, generation)) {
            return slot;
        }
    }

    return nullptr;
}

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_find_slot(uint64_t gridid,
                                                                             uint64_t block_id) {
    return nvshmemi_region_find_active_slot(gridid, block_id);
}

__device__ __forceinline__ nvshmemi_region_info_t
nvshmemi_region_resolve_active_current(uint32_t supported_hints) {
    nvshmemi_region_info_t info = {};
    uint64_t generation = 0;
    nvshmemi_region_slot_t *slot = nvshmemi_region_find_active_slot(
        nvshmemi_get_grid_id(), nvshmemi_get_flat_blk_idx(), &generation);
    if (slot != nullptr) {
        uint32_t hints = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(slot->hints)
                             .load(cuda::memory_order_relaxed);
        if ((hints & supported_hints) != 0) {
            info.issuer_id = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->issuer_id)
                                 .load(cuda::memory_order_relaxed);
            info.region_id = generation;
            info.hints = hints & supported_hints;
        }
    }
    return info;
}

/* Device issuer IDs encode the owning slot. The generation distinguishes slot reuse. */
__device__ __forceinline__ uint64_t nvshmemi_region_issuer_from_slot(uint32_t slot_index) {
    return static_cast<uint64_t>(slot_index) + 1;
}

__device__ __forceinline__ bool nvshmemi_region_info_has_hints(
    const nvshmemi_region_info_t *region_info, uint32_t hints) {
    return region_info != nullptr && (region_info->hints & hints) == hints;
}

__device__ __forceinline__ uint32_t
nvshmemi_region_slot_index(const nvshmemi_region_info_t *region_info) {
    if (region_info == nullptr || region_info->hints == NVSHMEMX_REGION_HINT_NONE ||
        region_info->issuer_id == 0 ||
        region_info->issuer_id > nvshmemi_device_state_d.region_slots_len) {
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(region_info->issuer_id - 1);
}

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_slot_from_info(
    const nvshmemi_region_info_t *region_info) {
    uint32_t slot_index = nvshmemi_region_slot_index(region_info);
    if (slot_index == UINT32_MAX) {
        return nullptr;
    }

    nvshmemi_region_slot_t *slot = &nvshmemi_device_state_d.region_slots[slot_index];
    unsigned long long state =
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(slot->state)
            .load(cuda::memory_order_acquire);
    uint64_t issuer_id = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->issuer_id)
                             .load(cuda::memory_order_relaxed);
    uint32_t hints = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(slot->hints)
                         .load(cuda::memory_order_acquire);
    uint64_t generation = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->generation)
                              .load(cuda::memory_order_relaxed);
    if (!nvshmemi_region_slot_state_is_active(state) ||
        nvshmemi_region_slot_state_generation(state) != generation ||
        issuer_id != region_info->issuer_id || generation != region_info->region_id ||
        (hints & region_info->hints) != region_info->hints) {
        return nullptr;
    }
    return slot;
}

__device__ __forceinline__ bool nvshmemi_region_batch_rma_should_submit(
    const nvshmemi_region_info_t *region_info, uint32_t threshold) {
    if (!nvshmemi_region_info_has_hints(region_info, NVSHMEMI_REGION_HINT_BATCH_RMA) ||
        threshold == 0) {
        return false;
    }

    nvshmemi_region_slot_t *slot = nvshmemi_region_slot_from_info(region_info);
    if (slot == nullptr) {
        return false;
    }

    /* The ticket is deliberately monotonic for the region lifetime. Concurrent issuers may race
     * with a threshold-triggered submission, but they never contend on a reset or require
     * serialization. The threshold is therefore a best-effort performance control. */
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> operation_ticket(
        slot->batch_rma.operation_ticket);
    uint32_t ticket = operation_ticket.fetch_add(1u, cuda::memory_order_relaxed);
    return ticket % threshold == threshold - 1;
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_REGION_STATE_CUH_ */
