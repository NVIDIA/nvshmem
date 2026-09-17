/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMX_STATUS_H
#define NVSHMEMX_STATUS_H

#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
extern "C" {
#endif

#define NVSHMEMX_STATUS_LIST(X)                \
    X(NVSHMEMX_SUCCESS)                        \
    X(NVSHMEMX_ERROR_INVALID_VALUE)            \
    X(NVSHMEMX_ERROR_OUT_OF_MEMORY)            \
    X(NVSHMEMX_ERROR_NOT_SUPPORTED)            \
    X(NVSHMEMX_ERROR_SYMMETRY)                 \
    X(NVSHMEMX_ERROR_GPU_NOT_SELECTED)         \
    X(NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED) \
    X(NVSHMEMX_ERROR_INTERNAL)

#define NVSHMEMX_GENERATE_ENUM(e) e,
enum nvshmemx_status {
    NVSHMEMX_STATUS_LIST(NVSHMEMX_GENERATE_ENUM) NVSHMEMX_ERROR_SENTINEL = INT_MAX
};
#undef NVSHMEMX_GENERATE_ENUM

#if !defined __CUDACC_RTC__
#define NVSHMEMX_STATUS_CASE(e) \
    case e:                     \
        return #e;
static inline const char *nvshmemx_status_string(int status) {
    switch (status) { NVSHMEMX_STATUS_LIST(NVSHMEMX_STATUS_CASE) }
    return "NVSHMEMX_ERROR_<unknown>";
}
#undef NVSHMEMX_STATUS_CASE
#else
static __device__ inline const char *nvshmemx_status_string(int /*status*/) { return ""; }
#endif

#undef NVSHMEMX_STATUS_LIST

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
}
#endif

#endif
