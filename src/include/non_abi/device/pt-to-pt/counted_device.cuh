/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_COUNTED_DEVICE_CUH_
#define _NVSHMEMI_COUNTED_DEVICE_CUH_

#include "non_abi/device/common/nvshmemi_common_device.cuh"

#if defined(__CUDA_ARCH__) && defined(NVSHMEM_ENABLE_CFT_HANDLES)

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
    return NVSHMEMX_ERROR_NOT_SUPPORTED;
}

#endif

#endif
