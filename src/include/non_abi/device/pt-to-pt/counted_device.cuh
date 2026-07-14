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
    size_t offset = static_cast<size_t>(addr - base);
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
    constexpr size_t counted_counter_alignment = 256;
    const uintptr_t dest_addr = reinterpret_cast<uintptr_t>(dest);
    const uintptr_t signal_addr_value = reinterpret_cast<uintptr_t>(signal_addr);
    const size_t signal_offset = reinterpret_cast<uintptr_t>(signal_addr) -
                                 reinterpret_cast<uintptr_t>(nvshmemi_device_state_d.heap_base);
    if (signal_addr_value >= dest_addr &&
        static_cast<size_t>(signal_addr_value - dest_addr) < bytes)
        return NVSHMEMX_ERROR_INVALID_VALUE;
    if ((dest_addr & 15u) != 0u || (reinterpret_cast<uintptr_t>(source) & 15u) != 0u ||
        (signal_offset & (counted_counter_alignment - 1)) != 0u || (bytes & 15u) != 0u)
        return NVSHMEMX_ERROR_INVALID_VALUE;
    return NVSHMEMX_SUCCESS;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_putmem_signal_counted_nbi_block(
    void *dest, const void *source, size_t bytes, uint64_t *signal_addr, int pe) {
    int status = nvshmemi_validate_counted_put(dest, source, bytes, signal_addr, pe);
    if (status != NVSHMEMX_SUCCESS) return status;
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT) && \
    !defined(__CUDACC_RTC__) && !defined(__clang_llvm_bitcode_lib__) &&  \
    !defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
    if (!__isShared(source)) return NVSHMEMX_ERROR_NOT_SUPPORTED;
    if (!nvshmemi_device_state_d.counted_operations_available ||
        nvshmemi_device_state_d.tma_policy != NVSHMEMX_TMA_ENABLE ||
        !nvshmemi_ld_and_check_valid_le_id(pe) || !nvshmemi_tma_smem_registered() || bytes == 0 ||
        bytes > static_cast<size_t>(TMA_COPY_MAX_BATCH_SIZE))
        return NVSHMEMX_ERROR_NOT_SUPPORTED;

    unsigned int tid =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uintptr_t smem_base = nvshmemi_tma_smem_base();
    int *cta_status = reinterpret_cast<int *>(
        nvshmemi_counted_state_slot(smem_base, NVSHMEMI_COUNTED_STATUS_SLOT));
    handle_barrier_t *completion = reinterpret_cast<handle_barrier_t *>(
        nvshmemi_counted_state_slot(smem_base, NVSHMEMI_COUNTED_HANDLE_BARRIER_SLOT));

    if (tid == 0) {
        *cta_status = NVSHMEMX_SUCCESS;
        completion->init(1);
    }
    __syncthreads();
    /* Counted fabric puts read their shared-memory source through the async
     * proxy.  Publish generic-proxy stores from every participating thread
     * before the CTA elects thread 0 to issue the fabric operation. */
    fence_async_proxy();
    __syncthreads();

    if (tid == 0) {
        uint64_t token = completion->arrive_relaxed(static_cast<uint32_t>(bytes));
        uint64_t data_off = static_cast<uint64_t>(
            reinterpret_cast<const char *>(dest) -
            reinterpret_cast<const char *>(nvshmemi_device_state_d.heap_base));
        uint64_t count_off = static_cast<uint64_t>(
            reinterpret_cast<const char *>(signal_addr) -
            reinterpret_cast<const char *>(nvshmemi_device_state_d.heap_base));
        fabric_try_put_counted_async(nvshmemi_ld_and_get_le_id(pe), data_off, count_off, source,
                                     static_cast<uint32_t>(bytes), completion);
        fabric_submit();
        uint8_t barrier_error = 0;
        while (!completion->try_wait_token_with_err(token, &barrier_error)) {
            if (barrier_error) break;
        }
        if (barrier_error) *cta_status = NVSHMEMX_ERROR_INTERNAL;
        completion->fabric_wait_sync_reads();
        completion->inval();
    }
    __syncthreads();
    return *cta_status;
#else
    return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
}

#endif

#endif
