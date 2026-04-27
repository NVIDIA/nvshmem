/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __logical_endpoint_device_cuh__
#define __logical_endpoint_device_cuh__

#ifdef __CUDA_ARCH__

#include "device_host/logical_endpoint_types.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

__device__ __forceinline__ bool nvshmemi_ld_and_check_valid_le_id(int pe) {
#if defined(CFT_HANDLES_ENABLED)
    const long long unsigned *le_ids =
        reinterpret_cast<const long long unsigned *>(nvshmemi_device_state_d.unicast_le_ids_);
    return IS_VALID_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return false;
#endif
}

__device__ __forceinline__ CUlogicalEndpointId nvshmemi_ld_and_get_le_id(int pe) {
#if defined(CFT_HANDLES_ENABLED)
    const long long unsigned *le_ids =
        reinterpret_cast<const long long unsigned *>(nvshmemi_device_state_d.unicast_le_ids_);
    return (CUlogicalEndpointId)PARSE_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return (CUlogicalEndpointId)-1;
#endif
}

#if PRIORITIZE_LOGICAL_ENDPOINT == 1
__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int pe) {
    return nvshmemi_ld_and_check_valid_le_id(pe);
}
#else
__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int) {
    return false;
}
#endif

#endif  // __CUDA_ARCH__
#endif  // __logical_endpoint_device_cuh__
