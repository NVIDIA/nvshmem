/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEMI_DEVICE_RMA_COMPLETION_CUH
#define NVSHMEMI_DEVICE_RMA_COMPLETION_CUH

#include "device/nvshmem_device_macros.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

namespace nvshmemi {
template <threadgroup_t Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void complete_device_rma(bool) {}
}  // namespace nvshmemi
#endif
