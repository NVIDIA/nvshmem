/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TYPE_TRAITS_CUH_
#define _NVSHMEMI_TYPE_TRAITS_CUH_

#include <cuda_fp16.h>

template <typename T>
__host__ __device__ constexpr bool nvshmemi_is_float_type() {
    return false;
}

template <>
__host__ __device__ constexpr bool nvshmemi_is_float_type<__half>() {
    return true;
}

template <>
__host__ __device__ constexpr bool nvshmemi_is_float_type<float>() {
    return true;
}

template <>
__host__ __device__ constexpr bool nvshmemi_is_float_type<double>() {
    return true;
}

#endif /* _NVSHMEMI_TYPE_TRAITS_CUH_ */
