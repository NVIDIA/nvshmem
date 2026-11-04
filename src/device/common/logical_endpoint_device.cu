/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "non_abi/c/nvshmem_build_options.h"
#include "non_abi/device/common/nvshmemi_common_device.cuh"

/* Explicitly instantiate scalar p's CFT slow path in the device library so it inherits the
 * library's --maxrregcount=32 policy instead of increasing caller register demand. */
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(__CUDA_ARCH__)
#define NVSHMEMI_CFT_P_INSTANTIATE(TYPE) \
    template __device__ bool nvshmemi_cft_p_slowpath<TYPE>(TYPE *, TYPE, int);
NVSHMEMI_CFT_P_INSTANTIATE(__nv_bfloat16)
NVSHMEMI_CFT_P_INSTANTIATE(half)
NVSHMEMI_CFT_P_INSTANTIATE(float)
NVSHMEMI_CFT_P_INSTANTIATE(double)
NVSHMEMI_CFT_P_INSTANTIATE(char)
NVSHMEMI_CFT_P_INSTANTIATE(signed char)
NVSHMEMI_CFT_P_INSTANTIATE(unsigned char)
NVSHMEMI_CFT_P_INSTANTIATE(short)
NVSHMEMI_CFT_P_INSTANTIATE(unsigned short)
NVSHMEMI_CFT_P_INSTANTIATE(int)
NVSHMEMI_CFT_P_INSTANTIATE(unsigned int)
NVSHMEMI_CFT_P_INSTANTIATE(long)
NVSHMEMI_CFT_P_INSTANTIATE(unsigned long)
NVSHMEMI_CFT_P_INSTANTIATE(long long)
NVSHMEMI_CFT_P_INSTANTIATE(unsigned long long)
#undef NVSHMEMI_CFT_P_INSTANTIATE
#endif
