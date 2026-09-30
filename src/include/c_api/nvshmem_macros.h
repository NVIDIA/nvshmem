/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEM_MACROS_H_
#define _NVSHMEM_MACROS_H_

#include <cuda_runtime.h>
#include "non_abi/c/nvshmem_build_options.h"

/* Our bitcode/LTOIR test and perftest build rules define NVSHMEM_DEVICE_DECLARATIONS_ONLY
 * before including nvshmem.h or nvshmemx.h. The umbrella headers then omit device
 * implementation headers, and the build links the precompiled device library.
 * Normal CUDA applications and CuTe DSL users do not need this test build mode.
 *
 * The test kernels need host/device declarations on both NVCC passes. Do not
 * force host calls to inline because their function bodies are not included.
 */
#if defined(NVSHMEM_DEVICE_DECLARATIONS_ONLY)
#ifdef __CUDA_ARCH__
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__ __attribute__((always_inline))
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#endif
#elif defined(__CUDA_ARCH__)
#ifdef NVSHMEMI_HOST_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#endif
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX
#endif

#endif
