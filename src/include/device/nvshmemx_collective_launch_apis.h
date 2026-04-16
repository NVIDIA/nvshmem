/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_
#define _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_

#include <cuda_runtime.h>

#if !defined __CUDACC_RTC__
int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream);
int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize);
#endif

#endif
