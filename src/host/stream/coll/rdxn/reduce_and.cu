/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reduce_common.cuh"

REPT_FOR_BITWISE_TYPES(INSTANTIATE_NVSHMEMI_CALL_RDXN_ON_STREAM_KERNEL, AND)
