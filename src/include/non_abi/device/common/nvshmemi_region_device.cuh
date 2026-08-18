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

__device__ __forceinline__ uint32_t nvshmemi_region_lock_active_count(
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> &active_count_ref) {
    uint32_t current = active_count_ref.load(cuda::memory_order_relaxed);

    while (true) {
        while ((current & NVSHMEMI_REGION_ACTIVE_COUNT_LOCK) != 0) {
            current = active_count_ref.load(cuda::memory_order_relaxed);
        }
        if (active_count_ref.compare_exchange_weak(
                current, current | NVSHMEMI_REGION_ACTIVE_COUNT_LOCK, cuda::memory_order_acquire,
                cuda::memory_order_relaxed)) {
            return current;
        }
    }
}

__device__ __forceinline__ void nvshmemi_region_increment_active_count() {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count_ref(
        *nvshmemi_device_state_d.region_active_count);
    uint32_t count = nvshmemi_region_lock_active_count(active_count_ref);
    assert(count < NVSHMEMI_REGION_ACTIVE_COUNT_MASK);
    active_count_ref.store(count + 1, cuda::memory_order_release);
}

__device__ __forceinline__ void nvshmemi_region_decrement_active_count() {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> active_count_ref(
        *nvshmemi_device_state_d.region_active_count);
    uint32_t count = nvshmemi_region_lock_active_count(active_count_ref);
    assert(count > 0);
    active_count_ref.store(count - 1, cuda::memory_order_release);
}

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
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> generation_ref(slot->generation);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> issuer_ref(slot->issuer_id);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> gridid_ref(slot->gridid);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> block_id_ref(slot->block_id);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> hints_ref(slot->hints);
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> operation_ticket_ref(
                slot->batch_rma.operation_ticket);

            uint64_t generation = generation_ref.load(cuda::memory_order_relaxed) + 1;
            if (generation == 0 || generation > NVSHMEMI_REGION_SLOT_GENERATION_MAX) {
                generation = 1;
            }
            /* Publish the new generation before mutable metadata. Readers that observe any new
             * release-published key field must also observe this generation and reject an older
             * ACTIVE state. */
            generation_ref.store(generation, cuda::memory_order_relaxed);
            issuer_ref.store(nvshmemi_region_issuer_from_slot(slot_index),
                             cuda::memory_order_relaxed);
            gridid_ref.store(gridid, cuda::memory_order_release);
            block_id_ref.store(block_id, cuda::memory_order_release);
            hints_ref.store(hints, cuda::memory_order_release);
            operation_ticket_ref.store(0, cuda::memory_order_relaxed);
            state_ref.store(nvshmemi_region_slot_state(NVSHMEMI_REGION_SLOT_ACTIVE, generation),
                            cuda::memory_order_release);
            nvshmemi_region_increment_active_count();
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
    return slot != NULL && cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(slot->generation)
                                   .load(cuda::memory_order_relaxed) == handle
               ? slot
               : NULL;
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
            nvshmemi_region_set_block_hints(
                attrs == NULL ? static_cast<uint32_t>(NVSHMEMX_REGION_HINT_NONE) : attrs->hints);
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
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> issuer_ref(slot->issuer_id);
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> generation_ref(slot->generation);
            uint32_t hints = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(slot->hints)
                                 .load(cuda::memory_order_relaxed);
            if (hints != NVSHMEMX_REGION_HINT_NONE) {
                nvshmemi_region_info_t region_info = {
                    issuer_ref.load(cuda::memory_order_relaxed),
                    generation_ref.load(cuda::memory_order_relaxed), hints};
                nvshmemi_transfer_region_end<NVSHMEMI_THREADGROUP_BLOCK>(&region_info);
            }
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> state_ref(slot->state);
            state_ref.store(NVSHMEMI_REGION_SLOT_FREE, cuda::memory_order_release);
            nvshmemi_region_decrement_active_count();
            nvshmemi_region_set_block_hints(NVSHMEMX_REGION_HINT_NONE);
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
