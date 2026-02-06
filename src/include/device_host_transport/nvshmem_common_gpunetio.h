/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEMI_GPUNETIO_COMMON_H_
#define _NVSHMEMI_GPUNETIO_COMMON_H_

#define NVSHMEMI_GPUNETIO_MIN_QP_DEPTH 128
#define NVSHMEMI_GPUNETIO_MAX_QP_DEPTH 32768

#if !defined __CUDACC_RTC__
#include <stddef.h>  // for size_t
#include <stdint.h>  // for uint64_t, uint32_t, uint16_t, uint8_t
#include <limits.h>


#define nvshmemi_init_gpunetio_device_state(state)                            \
    do {                                                                  \
        state.version = (1 << 16) + sizeof(nvshmemi_gpunetio_device_state_t); \
    } while (0);

#else
#include <cuda/std/cstddef>
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include "gpunetio/doca_gpunetio_host.h"
#include <infiniband/mlx5dv.h>  // for mlx5_wqe_av
#include <linux/types.h>        // for __be32

#define NVSHMEMI_GPUNETIO_MIN_MAJOR_VERSION 2
static_assert(DOCA_GPUNETIO_VERSION_MAJOR >= NVSHMEMI_GPUNETIO_MIN_MAJOR_VERSION,
    "GPUNetIO major version too old. NVSHMEM requires DOCA_GPUNETIO_VERSION_MAJOR >= 2 for ABI compatibility.");

typedef struct {
    int version;
    // TODO
} nvshmemi_gpunetio_device_state_t;
// TODO: static_assert

#if defined(__CUDACC_RDC__) || defined(__NVSHMEM_NUMBA_SUPPORT__)
#define EXTERN_CONSTANT extern __constant__
#elif defined(__clang__)
#ifdef __CUDACC__
// Clang CUDA mode: use __constant__ only (avoid address_space to fix LLVM21)
#define EXTERN_CONSTANT extern __constant__
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only
#define EXTERN_CONSTANT extern __attribute__((address_space(4)))
#endif
#endif

#ifdef EXTERN_CONSTANT
EXTERN_CONSTANT nvshmemi_gpunetio_device_state_t nvshmemi_gpunetio_device_state_d;
#undef EXTERN_CONSTANT
#endif
#endif
