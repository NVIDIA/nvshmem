/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NVSHMEM_COMMON_DEVICE_DEFINES_CUH_
#define _NVSHMEM_COMMON_DEVICE_DEFINES_CUH_
#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"

#if defined(__CUDACC_RTC__) && defined(__NVSHMEM_NUMBA_SUPPORT__)
// NVRTC + Numba: Numba does not device-link against libnvshmem_device, so
// extern __constant__ would leave the symbol unresolved.  Instead, use a
// per-module __constant__ (no extern) that the host populates at runtime
// via cuModuleGetGlobal + cuMemcpyHtoD on the single NVRTC-compiled module.
#define EXTERN_CONSTANT __constant__
#endif
#include "device_host/nvshmemi_extern_constant.h"

#ifdef EXTERN_CONSTANT
EXTERN_CONSTANT nvshmemi_device_host_state_t nvshmemi_device_state_d;
#undef EXTERN_CONSTANT
#endif

#if defined(__NVSHMEM_NUMBA_SUPPORT__) || (defined(__CUDACC__) && !defined(__CUDACC_RDC__) && \
                                           !defined(__CUDACC_RTC__) && !defined(__clang__))
// Non-RDC / Numba: emit a per-TU version symbol for nvshmemx_cumodule_init;
// __CUDACC__ ensures this is never reached by a plain host compiler.
__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#endif

#ifdef __NVSHMEM_NUMBA_SUPPORT__
/* disable device-side asserts for Numba builds */
#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)
#endif

#if defined(__CUDA_ARCH__)
inline constexpr bool nvshmemi_device_has_nvls_multimem =
    (__CUDA_ARCH__ >= 900) && (CUDART_VERSION >= 12010);
#else
inline constexpr bool nvshmemi_device_has_nvls_multimem = false;
#endif

typedef enum {
    nvshmemi_threadgroup_thread = 0,
    NVSHMEMI_THREADGROUP_THREAD = 0,
    nvshmemi_threadgroup_warp = 1,
    NVSHMEMI_THREADGROUP_WARP = 1,
    nvshmemi_threadgroup_warpgroup = 2,
    NVSHMEMI_THREADGROUP_WARPGROUP = 2,
    nvshmemi_threadgroup_block = 3,
    NVSHMEMI_THREADGROUP_BLOCK = 3
} threadgroup_t;

#ifdef __CUDA_ARCH__

__device__ __forceinline__ uint64_t nvshmemi_get_grid_id() {
    uint64_t grid_id;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_id));
    return grid_id;
}

__device__ __forceinline__ uint32_t nvshmemi_get_flat_blk_idx() {
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

template <threadgroup_t scope>
__device__ __forceinline__ int nvshmemi_thread_id_in_threadgroup() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return 0;
        case NVSHMEMI_THREADGROUP_WARP:
            int myIdx;
            asm volatile("mov.u32  %0,  %%laneid;" : "=r"(myIdx));
            return myIdx;
        case NVSHMEMI_THREADGROUP_WARPGROUP:
            return (
                (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) %
                (4 * warpSize));
        case NVSHMEMI_THREADGROUP_BLOCK:
            return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            return -1;
    }
}

template <threadgroup_t scope>
__device__ inline int nvshmemi_threadgroup_size() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return 1;
        case NVSHMEMI_THREADGROUP_WARP:
            return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
                       ? (blockDim.x * blockDim.y * blockDim.z)
                       : warpSize;
        case NVSHMEMI_THREADGROUP_WARPGROUP:
            // warpgroup = 4 warps
            return ((blockDim.x * blockDim.y * blockDim.z) < (4 * warpSize))
                       ? (blockDim.x * blockDim.y * blockDim.z)
                       : (4 * warpSize);
        case NVSHMEMI_THREADGROUP_BLOCK:
            return (blockDim.x * blockDim.y * blockDim.z);
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            return -1;
    }
}

template <threadgroup_t scope>
__device__ inline void nvshmemi_threadgroup_sync() {
    uint32_t tid, barrierId;
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return;
        case NVSHMEMI_THREADGROUP_WARP:
            __syncwarp();
            break;
        case NVSHMEMI_THREADGROUP_WARPGROUP:
            // break;
            tid =
                threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);

            // each warpgroup has a unique barrier id per CTA
            // barrierId = 0 is used for the threadgroup_sync barrier
            barrierId = 1 + (tid / (4 * warpSize));
            // Ensure blocksize is a multiple of warpgroup size
            assert((blockDim.x * blockDim.y * blockDim.z) % (4 * warpSize) == 0);
            // Hardware limit on named barriers (max 16)
            assert((barrierId < 16) && "Reduce blocksize");
            asm volatile("bar.sync %0, %1;" : : "r"(barrierId), "r"(4 * warpSize));
            break;
        case NVSHMEMI_THREADGROUP_BLOCK:
            __syncthreads();
            break;
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            break;
    }
}
#endif

#endif
