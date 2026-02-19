/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "reduce_common.cuh"

REPT_FOR_BITWISE_TYPES(INSTANTIATE_NVSHMEMI_CALL_RDXN_ON_STREAM_KERNEL, OR)
