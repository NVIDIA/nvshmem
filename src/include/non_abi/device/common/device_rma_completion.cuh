/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEMI_DEVICE_RMA_COMPLETION_CUH
#define NVSHMEMI_DEVICE_RMA_COMPLETION_CUH

#include "device/nvshmem_device_macros.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

namespace nvshmemi {
// CallerSynchronizes means the caller provides group barriers before and after
// the hook, allowing an implementation to share those barriers.
template <threadgroup_t Scope, bool CallerSynchronizes = false>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void complete_device_rma(bool) {}
}  // namespace nvshmemi
#endif
