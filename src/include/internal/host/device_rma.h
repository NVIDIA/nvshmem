/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEMI_DEVICE_RMA_H
#define NVSHMEMI_DEVICE_RMA_H
#include <cuda_runtime.h>
#include "c_api/nvshmemx_status.h"

inline int nvshmemi_device_rma_init() { return NVSHMEMX_SUCCESS; }
inline int nvshmemi_device_rma_finalize() { return NVSHMEMX_SUCCESS; }
inline void nvshmemi_device_rma_on_stream(cudaStream_t, bool) {}
#endif
