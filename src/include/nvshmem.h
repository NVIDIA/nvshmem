/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEM_H_
#define _NVSHMEM_H_

#include "non_abi/c/nvshmem_build_options.h"
/* NVRTC only compiles device code. Leave out host headers */
#if !defined __CUDACC_RTC__ && !defined __clang_llvm_bitcode_lib__ && \
    !defined __NVSHMEM_NUMBA_SUPPORT__ && !defined NVSHMEM_BUILD_LTOIR_LIBRARY
#include "nvshmem_host.h"
#endif
/* Our bitcode/LTOIR test and perftest builds skip these implementation headers
 * and link the precompiled device library instead.
 */
#if !defined NVSHMEM_DEVICE_DECLARATIONS_ONLY
#include "device/nvshmem_defines.h"
#include "device/nvshmem_coll_defines.cuh"
#include "device/nvshmemx_defines.h"
#include "device/nvshmemx_coll_defines.cuh"
#endif
#endif
