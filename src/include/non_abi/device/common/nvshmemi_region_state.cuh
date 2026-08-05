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

__device__ __forceinline__ bool nvshmemi_region_slot_matches(const nvshmemi_region_slot_t *slot,
                                                             uint64_t gridid, uint64_t block_id) {
    return slot->gridid == gridid && slot->block_id == block_id;
}

__device__ __forceinline__ bool nvshmemi_region_slot_has_hints(const nvshmemi_region_slot_t *slot,
                                                               uint32_t hints) {
    return hints == NVSHMEMX_REGION_HINT_NONE || (slot->hints & hints) == hints;
}

__device__ __forceinline__ bool nvshmemi_region_any_active() {
    /* Avoid probing the region table on the common path with no active device regions. */
    uint32_t *active_count = nvshmemi_device_state_d.region_active_count;
    if (active_count == nullptr) {
        return false;
    }

    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count_ref(*active_count);
    return active_count_ref.load(cuda::memory_order_relaxed) != 0;
}

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_find_slot(uint64_t gridid,
                                                                             uint64_t block_id) {
    nvshmemi_region_slot_t *slots = nvshmemi_device_state_d.region_slots;
    uint32_t slots_len = nvshmemi_device_state_d.region_slots_len;
    uint32_t probe_limit = nvshmemi_device_state_d.region_slot_probe_limit;

    if (!nvshmemi_region_any_active() || slots == nullptr || slots_len == 0 || probe_limit == 0) {
        return nullptr;
    }

    uint32_t start = nvshmemi_region_probe_start(gridid, block_id, slots_len);
    for (uint32_t i = 0; i < probe_limit; i++) {
        nvshmemi_region_slot_t *slot = &slots[(start + i) & (slots_len - 1)];
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> state_ref(slot->state);
        unsigned long long state = state_ref.load(cuda::memory_order_acquire);
        if (state == NVSHMEMI_REGION_SLOT_ACTIVE &&
            nvshmemi_region_slot_matches(slot, gridid, block_id)) {
            return slot;
        }
    }

    return nullptr;
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ nvshmemi_region_info_t
nvshmemi_region_resolve_leader(uint32_t supported_hints) {
    nvshmemi_region_info_t info = {};
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() != 0) {
        return info;
    }

    nvshmemi_region_slot_t *slot =
        nvshmemi_region_find_slot(nvshmemi_get_grid_id(), nvshmemi_get_flat_blk_idx());
    if (slot != nullptr && (slot->hints & supported_hints) != 0) {
        info.issuer_id = slot->issuer_id;
        info.region_id = slot->generation;
        info.hints = slot->hints & supported_hints;
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

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_slot_from_info(
    const nvshmemi_region_info_t *region_info) {
    if (region_info == nullptr || region_info->hints == NVSHMEMX_REGION_HINT_NONE ||
        region_info->issuer_id == 0 ||
        region_info->issuer_id > nvshmemi_device_state_d.region_slots_len) {
        return nullptr;
    }

    nvshmemi_region_slot_t *slot =
        &nvshmemi_device_state_d.region_slots[static_cast<uint32_t>(region_info->issuer_id - 1)];
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> state_ref(slot->state);
    unsigned long long state = state_ref.load(cuda::memory_order_acquire);
    if (state != NVSHMEMI_REGION_SLOT_ACTIVE || slot->issuer_id != region_info->issuer_id ||
        slot->generation != region_info->region_id ||
        (slot->hints & region_info->hints) != region_info->hints) {
        return nullptr;
    }
    return slot;
}

__device__ __forceinline__ uint32_t
nvshmemi_region_slot_index(const nvshmemi_region_info_t *region_info) {
    nvshmemi_region_slot_t *slot = nvshmemi_region_slot_from_info(region_info);
    return slot == nullptr ? UINT32_MAX : static_cast<uint32_t>(region_info->issuer_id - 1);
}

__device__ __forceinline__ bool nvshmemi_region_batch_rma_reached_submission_threshold(
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
