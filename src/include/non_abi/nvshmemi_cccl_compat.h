/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CUDA 13 / CCCL include-path compatibility.
 * Some CCCL headers moved from cuda/std/... to cccl/cuda/std/... in certain
 * toolchain versions.  Prefer the canonical paths and fall back to the new
 * locations so the same source builds across both old and new toolchains.
 */

#ifndef _NVSHMEMI_CCCL_COMPAT_H_
#define _NVSHMEMI_CCCL_COMPAT_H_

#if __has_include(<cuda/std/tuple>)
#include <cuda/std/tuple>
#else
#include <cccl/cuda/std/tuple>
#endif

#if __has_include(<cuda/std/type_traits>)
#include <cuda/std/type_traits>
#else
#include <cccl/cuda/std/type_traits>
#endif

#if __has_include(<cuda/std/utility>)
#include <cuda/std/utility>
#else
#include <cccl/cuda/std/utility>
#endif

#endif /* _NVSHMEMI_CCCL_COMPAT_H_ */
