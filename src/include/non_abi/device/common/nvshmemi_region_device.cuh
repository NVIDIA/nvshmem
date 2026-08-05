/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_DEVICE_CUH_
#define _NVSHMEMI_REGION_DEVICE_CUH_

#include "device_host/nvshmem_types.h"
#include "non_abi/device/common/nvshmemi_region_state.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/nvshmemx_error.h"

#ifdef __CUDA_ARCH__

__device__ __forceinline__ int nvshmemi_region_claim_slot(nvshmemx_region_handle_t *handle,
                                                          const nvshmemx_region_attrs_t *attrs) {
    if (handle == NULL) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    uint32_t hints =
        attrs == NULL ? static_cast<uint32_t>(NVSHMEMX_REGION_HINT_NONE) : attrs->hints;
    if (!nvshmemi_region_hints_are_valid(hints)) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (nvshmemi_device_state_d.region_slots == NULL ||
        nvshmemi_device_state_d.region_active_count == NULL ||
        nvshmemi_device_state_d.region_slots_len == 0) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    uint64_t gridid = nvshmemi_get_grid_id();
    uint64_t block_id = nvshmemi_get_flat_blk_idx();
    if (nvshmemi_region_find_slot(gridid, block_id) != NULL) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    uint32_t slots_len = nvshmemi_device_state_d.region_slots_len;
    uint32_t probe_limit = nvshmemi_device_state_d.region_slot_probe_limit;
    uint32_t start = nvshmemi_region_probe_start(gridid, block_id, slots_len);

    for (uint32_t i = 0; i < probe_limit; i++) {
        uint32_t slot_index = (start + i) & (slots_len - 1);
        nvshmemi_region_slot_t *slot = &nvshmemi_device_state_d.region_slots[slot_index];
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> state_ref(slot->state);
        unsigned long long expected = NVSHMEMI_REGION_SLOT_FREE;
        if (state_ref.compare_exchange_strong(expected, NVSHMEMI_REGION_SLOT_INITIALIZING,
                                              cuda::memory_order_acquire,
                                              cuda::memory_order_relaxed)) {
            uint64_t generation = slot->generation + 1;
            if (generation == 0) {
                generation = 1;
            }
            slot->generation = generation;
            slot->issuer_id = nvshmemi_region_issuer_from_slot(slot_index);
            slot->gridid = gridid;
            slot->block_id = block_id;
            slot->hints = hints;
            slot->batch_rma.operation_ticket = 0;
            state_ref.store(NVSHMEMI_REGION_SLOT_ACTIVE, cuda::memory_order_release);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count(
                *nvshmemi_device_state_d.region_active_count);
            active_count.fetch_add(1u, cuda::memory_order_relaxed);
            *handle = generation;
            return NVSHMEMX_SUCCESS;
        }
    }

    return NVSHMEMX_ERROR_OUT_OF_MEMORY;
}

__device__ __forceinline__ nvshmemi_region_slot_t *nvshmemi_region_find_block_slot(
    nvshmemx_region_handle_t handle) {
    nvshmemi_region_slot_t *slot =
        nvshmemi_region_find_slot(nvshmemi_get_grid_id(), nvshmemi_get_flat_blk_idx());
    return slot != NULL && slot->generation == handle ? slot : NULL;
}

template <threadgroup_t SCOPE, nvshmemi_region_operation_t OPERATION>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_region_info_t nvshmemi_region_resolve() {
    constexpr uint32_t supported_hints =
        nvshmemi_region_operation_traits<OPERATION>::supported_hints;
    nvshmemi_region_info_t info = {};
    int my_idx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (my_idx == 0) {
        nvshmemi_region_slot_t *slot =
            nvshmemi_region_find_slot(nvshmemi_get_grid_id(), nvshmemi_get_flat_blk_idx());
        if (slot != NULL && (slot->hints & supported_hints) != 0) {
            info.issuer_id = slot->issuer_id;
            info.region_id = slot->generation;
            info.hints = slot->hints & supported_hints;
        }
    }

    if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        unsigned mask = __activemask();
        info.issuer_id = __shfl_sync(mask, info.issuer_id, 0);
        info.region_id = __shfl_sync(mask, info.region_id, 0);
        info.hints = __shfl_sync(mask, info.hints, 0);
    } else if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        __shared__ nvshmemi_region_info_t shared_info;
        if (my_idx == 0) {
            shared_info = info;
        }
        __syncthreads();
        info = shared_info;
    }

    return info;
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_region_submit(
    const nvshmemi_region_info_t *region_info) {
    nvshmemi_transfer_region_end<SCOPE>(region_info);
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_region_start_block(
    nvshmemx_region_handle_t *handle, const nvshmemx_region_attrs_t *attrs) {
    int status = NVSHMEMX_SUCCESS;
    nvshmemx_region_handle_t local_handle = 0;
    int my_idx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();

    nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_BLOCK>();
    if (my_idx == 0) {
        status = nvshmemi_region_claim_slot(handle, attrs);
        if (status == NVSHMEMX_SUCCESS) {
            local_handle = *handle;
        }
    }

    __shared__ int shared_status;
    __shared__ nvshmemx_region_handle_t shared_handle;
    if (my_idx == 0) {
        shared_status = status;
        shared_handle = local_handle;
    }
    __syncthreads();
    status = shared_status;
    local_handle = shared_handle;

    if (status == NVSHMEMX_SUCCESS && handle != NULL) {
        *handle = local_handle;
    }
    nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_BLOCK>();
    return status;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_region_stop_block(
    nvshmemx_region_handle_t handle) {
    int status = NVSHMEMX_SUCCESS;
    int my_idx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();

    nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_BLOCK>();
    if (my_idx == 0) {
        nvshmemi_region_slot_t *slot = nvshmemi_region_find_block_slot(handle);
        if (slot == NULL) {
            status = NVSHMEMX_ERROR_INVALID_VALUE;
        } else {
            if (slot->hints != NVSHMEMX_REGION_HINT_NONE) {
                nvshmemi_region_info_t region_info = {slot->issuer_id, slot->generation,
                                                      slot->hints};
                nvshmemi_region_submit<NVSHMEMI_THREADGROUP_BLOCK>(&region_info);
            }
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> state_ref(slot->state);
            state_ref.store(NVSHMEMI_REGION_SLOT_FREE, cuda::memory_order_release);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count(
                *nvshmemi_device_state_d.region_active_count);
            active_count.fetch_sub(1u, cuda::memory_order_relaxed);
        }
    }

    __shared__ int shared_status;
    if (my_idx == 0) {
        shared_status = status;
    }
    __syncthreads();
    status = shared_status;

    nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_BLOCK>();
    return status;
}

__device__ __forceinline__ int nvshmemi_region_is_active(uint32_t hints, int *active) {
    if (active == NULL) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (!nvshmemi_region_hints_are_valid(hints)) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    nvshmemi_region_slot_t *slot =
        nvshmemi_region_find_slot(nvshmemi_get_grid_id(), nvshmemi_get_flat_blk_idx());
    *active = slot != NULL && nvshmemi_region_slot_has_hints(slot, hints);
    return NVSHMEMX_SUCCESS;
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_REGION_DEVICE_CUH_ */
