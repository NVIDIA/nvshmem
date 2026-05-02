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

inline constexpr int CFT_HANDLE_TX_SIZE = 16;
inline constexpr int TMA_COPY_NUM_STAGES = 2;

inline constexpr std::uint64_t LE_ID_VALID_MASK = 0xFFFF0000ull;
inline constexpr std::uint64_t LE_ID_MASK = 0x0000FFFFull;

__host__ __device__ constexpr std::uint64_t LE_ID_WITH_VALID_FLAG(std::uint64_t le_id) {
    return LE_ID_VALID_MASK | le_id;
}

__host__ __device__ constexpr std::uint64_t PARSE_LE_ID(std::uint64_t le_id) {
    return le_id & LE_ID_MASK;
}

__host__ __device__ constexpr bool IS_VALID_LE_ID(std::uint64_t le_id) {
    return (le_id & LE_ID_VALID_MASK) != 0;
}

// MAX_BATCH_SIZE is 1MB for try_get, and 16MB for try_put
inline constexpr int TMA_COPY_MAX_BATCH_SIZE = 1 << 24;  // 16MB

#endif  // _NVSHMEM_LOGICAL_ENDPOINT_DEFINES_H_
