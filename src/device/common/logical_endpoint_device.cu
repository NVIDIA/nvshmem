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

/*
 * Keep only concrete device-library entry points here. The generic CFT helpers remain in
 * nvshmemi_common_device.cuh so NVCC can instantiate the required type, operation, scope,
 * size-rule, and local-space combinations at their call sites. Moving those templates here
 * would require maintaining and exporting the complete explicit-instantiation matrix.
 *
 * Block get/put use concrete wrappers so they inherit the existing device-library
 * --maxrregcount=32 policy. Compiling the full block slow paths in caller translation units can
 * otherwise exceed the register budget of 1024-thread kernels and make them unlaunchable.
 */
__device__ NVSHMEMI_NOINLINE bool nvshmemi_cft_get_block(const void *source, void *dest,
                                                         size_t bytes, int pe, bool is_blocking) {
    if (!nvshmemi_is_le_implemented<nvshmemi_le_size_rule::tx_size_multiple>(
            pe, bytes, NVSHMEMI_THREADGROUP_BLOCK, source, dest)) {
        return false;
    }
    nvshmemi_handle_get<NVSHMEMI_THREADGROUP_BLOCK, NVSHMEMI_LOCAL_SPACE_UNKNOWN>(
        source, dest, bytes, pe, is_blocking);
    return true;
}

__device__ NVSHMEMI_NOINLINE bool nvshmemi_cft_put_block(const void *source, void *dest,
                                                         size_t bytes, int pe, bool is_blocking) {
    if (!nvshmemi_is_le_implemented<nvshmemi_le_size_rule::any_size>(
            pe, bytes, NVSHMEMI_THREADGROUP_BLOCK, dest, source)) {
        return false;
    }
    nvshmemi_handle_put<NVSHMEMI_THREADGROUP_BLOCK, NVSHMEMI_LOCAL_SPACE_UNKNOWN>(
        source, dest, bytes, pe, is_blocking);
    return true;
}
#endif
