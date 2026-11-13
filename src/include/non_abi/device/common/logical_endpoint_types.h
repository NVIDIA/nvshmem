/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEM_LOGICAL_ENDPOINT_DEFINES_H_
#define _NVSHMEM_LOGICAL_ENDPOINT_DEFINES_H_

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#if CUDART_VERSION < 13030
typedef uint32_t CUlogicalEndpointId;
#endif

// Update this to check CUDA, driver, version and architecture requirements for CFT handles
#define LE_HW_SW_REQUIREMENTS_MET ((__CUDA_ARCH__ >= 1000) && (CUDART_VERSION >= 13030))
#define LE_ATOMIC_HW_SW_REQUIREMENTS_MET ((__CUDA_ARCH__ >= 1000) && (CUDART_VERSION >= 13040))

inline constexpr int CFT_HANDLE_TX_SIZE = 16;
inline constexpr int TMA_COPY_NUM_STAGES = 2;

inline constexpr std::uint64_t LE_ID_VALID_MASK = 0xFFFFFFFF00000000ull;
inline constexpr std::uint64_t LE_ID_MASK = 0x00000000FFFFFFFFull;

__host__ __device__ constexpr std::uint64_t LE_ID_WITH_VALID_FLAG(std::uint64_t le_id) {
    return LE_ID_VALID_MASK | le_id;
}

__host__ __device__ constexpr std::uint64_t PARSE_LE_ID(std::uint64_t le_id) {
    return le_id & LE_ID_MASK;
}

__host__ __device__ constexpr bool IS_VALID_LE_ID(std::uint64_t le_id) {
    return (le_id & LE_ID_VALID_MASK) != 0;
}

__host__ __device__ inline bool nvshmemi_is_le_atomic_address_eligible(const void *heap_base,
                                                                       const void *addr) {
    if (heap_base == nullptr || addr == nullptr) {
        return false;
    }
    const std::uintptr_t heap_offset =
        reinterpret_cast<std::uintptr_t>(addr) - reinterpret_cast<std::uintptr_t>(heap_base);
    return (heap_offset % CFT_HANDLE_TX_SIZE) == 0;
}

inline bool nvshmemi_is_le_atomic_implemented_on_host(std::uint64_t le_id_with_flag,
                                                      bool tma_enabled, const void *heap_base,
                                                      const void *addr) {
    return IS_VALID_LE_ID(le_id_with_flag) && tma_enabled &&
           nvshmemi_is_le_atomic_address_eligible(heap_base, addr);
}

inline constexpr int TMA_GET_MAX_BATCH_SIZE = 1 << 20;  // 1MB
inline constexpr int TMA_PUT_MAX_BATCH_SIZE = 1 << 24;  // 16MB

#endif  // _NVSHMEM_LOGICAL_ENDPOINT_DEFINES_H_
