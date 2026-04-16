/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "non_abi/nvshmem_build_options.h"

#ifndef _NVSHMEMX_H_
#define _NVSHMEMX_H_

/* NVRTC only compiles device code. Leave out host headers */
#if !defined __CUDACC_RTC__ && !defined __clang_llvm_bitcode_lib__ && \
    !defined __NVSHMEM_NUMBA_SUPPORT__ && !defined NVSHMEM_BUILD_LTOIR_LIBRARY
#include "host/nvshmemx_api.h"
#include "device/tile/nvshmemx_tile_api.hpp"
#include "device/nvshmemx_collective_launch_apis.h"
#endif
#if !defined NVSHMEM_HOSTLIB_ONLY
#include "device/nvshmemx_defines.h"
#include "device/nvshmemx_coll_defines.cuh"
#include "device/tile/nvshmemx_tile_api_defines.cuh"
#endif

#endif
