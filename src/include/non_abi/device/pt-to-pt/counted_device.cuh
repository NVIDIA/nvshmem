/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_COUNTED_DEVICE_CUH_
#define _NVSHMEMI_COUNTED_DEVICE_CUH_

#include "non_abi/device/common/nvshmemi_common_device.cuh"

#if defined(__CUDA_ARCH__) && defined(NVSHMEM_CFT_HANDLES_SUPPORT)

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_counted_range_in_heap(const void *ptr,
                                                                             size_t bytes) {
    if (ptr == nullptr || !__isGlobal(ptr)) return false;
    uintptr_t base = reinterpret_cast<uintptr_t>(nvshmemi_device_state_d.heap_base);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr < base) return false;
    size_t offset = static_cast<size_t>(nvshmemi_heap_offset(ptr));
    return offset < nvshmemi_device_state_d.heap_size &&
           bytes <= nvshmemi_device_state_d.heap_size - offset;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_validate_counted_put(
    const void *dest, const void *source, size_t bytes, const uint64_t *signal_addr, int pe) {
    if (pe < 0 || pe >= nvshmemi_device_state_d.npes) return NVSHMEMX_ERROR_INVALID_VALUE;
    if (!nvshmemi_counted_range_in_heap(dest, bytes) ||
        !nvshmemi_counted_range_in_heap(signal_addr, sizeof(*signal_addr)))
        return NVSHMEMX_ERROR_INVALID_VALUE;
    if (source == nullptr || (!__isShared(source) && !__isGlobal(source)))
        return NVSHMEMX_ERROR_INVALID_VALUE;
    if (nvshmemi_ptr_range_overflows(source, bytes)) return NVSHMEMX_ERROR_INVALID_VALUE;
    constexpr size_t counted_counter_alignment = 256;
    const uintptr_t dest_addr = reinterpret_cast<uintptr_t>(dest);
    const uintptr_t signal_addr_value = reinterpret_cast<uintptr_t>(signal_addr);
    const uint64_t signal_offset = nvshmemi_heap_offset(signal_addr);
    if (signal_addr_value >= dest_addr &&
        static_cast<size_t>(signal_addr_value - dest_addr) < bytes)
        return NVSHMEMX_ERROR_INVALID_VALUE;
    if ((dest_addr & 15u) != 0u || (reinterpret_cast<uintptr_t>(source) & 15u) != 0u ||
        (signal_offset & (counted_counter_alignment - 1)) != 0u || (bytes & 15u) != 0u)
        return NVSHMEMX_ERROR_INVALID_VALUE;
    return NVSHMEMX_SUCCESS;
}

#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT) && \
    !defined(__CUDACC_RTC__) && !defined(__clang_llvm_bitcode_lib__) &&  \
    !defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_counted_submit_chunk(
    CUlogicalEndpointId endpoint, uint64_t data_off, uint64_t count_off,
    const void *source_in_shared_memory, uint32_t bytes, handle_barrier_t *completion) {
    fabric_try_put_counted_async(endpoint, data_off, count_off, source_in_shared_memory, bytes,
                                 completion);
    fabric_submit();
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_counted_drain_batch(
    handle_barrier_t *completion, uint32_t pending_bytes) {
    uint64_t token = completion->arrive_relaxed(pending_bytes);
    uint8_t barrier_error = 0;
    while (!completion->try_wait_token_with_err(token, &barrier_error)) {
        if (barrier_error) break;
    }
    completion->fabric_wait_sync_reads();
    return barrier_error ? NVSHMEMX_ERROR_INTERNAL : NVSHMEMX_SUCCESS;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_counted_put_shared_source_block(
    void *dest, const void *source, size_t bytes, uint64_t *signal_addr, int pe, bool is_elected,
    int *cta_status, handle_barrier_t *completion) {
    /* Counted fabric puts read their shared-memory source through the async
     * proxy. Publish generic-proxy stores from every participating thread
     * before the CTA elects one thread to issue the fabric operation. */
    fence_async_proxy();
    __syncthreads();

    if (is_elected) {
        uint64_t data_off = nvshmemi_heap_offset(dest);
        uint64_t count_off = nvshmemi_heap_offset(signal_addr);
        CUlogicalEndpointId endpoint = nvshmemi_ld_and_get_le_id(pe);
        const char *src = reinterpret_cast<const char *>(source);
        size_t transferred = 0;
        uint32_t pending_bytes = 0;

        while (transferred < bytes) {
            uint32_t chunk = static_cast<uint32_t>(
                (bytes - transferred) < static_cast<size_t>(TMA_PUT_MAX_BATCH_SIZE)
                    ? (bytes - transferred)
                    : static_cast<size_t>(TMA_PUT_MAX_BATCH_SIZE));

            if (pending_bytes &&
                pending_bytes + chunk >= static_cast<uint32_t>(TMA_PUT_MAX_BATCH_SIZE)) {
                *cta_status = nvshmemi_counted_drain_batch(completion, pending_bytes);
                if (*cta_status != NVSHMEMX_SUCCESS) break;
                pending_bytes = 0;
            }

            nvshmemi_counted_submit_chunk(endpoint, data_off + transferred, count_off,
                                          src + transferred, chunk, completion);
            pending_bytes += chunk;
            transferred += chunk;
        }

        if (*cta_status == NVSHMEMX_SUCCESS && pending_bytes) {
            *cta_status = nvshmemi_counted_drain_batch(completion, pending_bytes);
        } else {
            completion->fabric_wait_sync_reads();
        }
        completion->inval();
    }
    __syncthreads();
    return *cta_status;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_counted_put_global_source_block(
    void *dest, const void *source, size_t bytes, uint64_t *signal_addr, int pe, bool is_elected,
    uintptr_t smem_base, size_t staging_capacity, int *cta_status, handle_barrier_t *completion) {
    if (is_elected) {
        uint64_t data_off = nvshmemi_heap_offset(dest);
        uint64_t count_off = nvshmemi_heap_offset(signal_addr);
        CUlogicalEndpointId endpoint = nvshmemi_ld_and_get_le_id(pe);
        const char *src = reinterpret_cast<const char *>(source);
        cuda::std::array<char *, TMA_COPY_NUM_STAGES> staging{};
        char *staging_base = nvshmemi_tma_data_buffer(smem_base);
        for (size_t stage = 0; stage < staging.size(); stage++)
            staging[stage] = staging_base + stage * staging_capacity;
        __mbarrier_t *staging_barrier = reinterpret_cast<__mbarrier_t *>(
            nvshmemi_counted_state_slot(smem_base, NVSHMEMI_COUNTED_TMA_BARRIER_SLOT));
        size_t transferred = 0;
        uint32_t pending_bytes = 0;

        __mbarrier_init(staging_barrier, 1);
        fence_async_proxy();

        for (size_t iteration = 0; transferred < bytes; iteration++) {
            size_t chunk_limit = staging_capacity < static_cast<size_t>(TMA_PUT_MAX_BATCH_SIZE)
                                     ? staging_capacity
                                     : static_cast<size_t>(TMA_PUT_MAX_BATCH_SIZE);
            uint32_t chunk = static_cast<uint32_t>(
                (bytes - transferred) < chunk_limit ? (bytes - transferred) : chunk_limit);

            if (pending_bytes &&
                pending_bytes + chunk >= static_cast<uint32_t>(TMA_PUT_MAX_BATCH_SIZE)) {
                *cta_status = nvshmemi_counted_drain_batch(completion, pending_bytes);
                if (*cta_status != NVSHMEMX_SUCCESS) break;
                pending_bytes = 0;
            }

            size_t slot = iteration % staging.size();
            if (iteration >= staging.size()) completion->fabric_wait_sync_reads();
            nvshmemi_tma_g2s_copy_thread(0, staging[slot], staging_barrier, src + transferred,
                                         chunk);

            nvshmemi_counted_submit_chunk(endpoint, data_off + transferred, count_off,
                                          staging[slot], chunk, completion);
            pending_bytes += chunk;
            transferred += chunk;
        }

        if (*cta_status == NVSHMEMX_SUCCESS && pending_bytes) {
            *cta_status = nvshmemi_counted_drain_batch(completion, pending_bytes);
        } else {
            completion->fabric_wait_sync_reads();
        }
        __mbarrier_inval(staging_barrier);
        completion->inval();
    }
    __syncthreads();
    return *cta_status;
}
#endif

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_putmem_signal_counted_nbi_block(
    void *dest, const void *source, size_t bytes, uint64_t *signal_addr, int pe) {
    int status = nvshmemi_validate_counted_put(dest, source, bytes, signal_addr, pe);
    if (status != NVSHMEMX_SUCCESS) return status;
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT) && \
    !defined(__CUDACC_RTC__) && !defined(__clang_llvm_bitcode_lib__) &&  \
    !defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
    if (!nvshmemi_device_state_d.counted_operations_available ||
        nvshmemi_device_state_d.tma_policy != NVSHMEMX_TMA_ENABLE ||
        !nvshmemi_ld_and_check_valid_le_id(pe))
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    if (bytes == 0) return NVSHMEMX_SUCCESS;
    if (!nvshmemi_tma_smem_registered()) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    const bool is_elected = nvshmemi_tma_block_is_elected();
    uintptr_t smem_base = nvshmemi_tma_smem_base();
    int *cta_status = reinterpret_cast<int *>(
        nvshmemi_counted_state_slot(smem_base, NVSHMEMI_COUNTED_STATUS_SLOT));
    handle_barrier_t *completion = reinterpret_cast<handle_barrier_t *>(
        nvshmemi_counted_state_slot(smem_base, NVSHMEMI_COUNTED_HANDLE_BARRIER_SLOT));
    bool source_is_shared = __isShared(source);
    size_t staging_capacity =
        source_is_shared ? 0 : nvshmemi_smem_data_buf_size(TMA_COPY_NUM_STAGES);
    if (!source_is_shared && staging_capacity == 0) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    if (is_elected) {
        *cta_status = NVSHMEMX_SUCCESS;
        completion->init(1);
    }
    __syncthreads();
    if (source_is_shared)
        return nvshmemi_counted_put_shared_source_block(dest, source, bytes, signal_addr, pe,
                                                        is_elected, cta_status, completion);
    return nvshmemi_counted_put_global_source_block(dest, source, bytes, signal_addr, pe,
                                                    is_elected, smem_base, staging_capacity,
                                                    cta_status, completion);
#else
    return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
}

#endif

#endif
