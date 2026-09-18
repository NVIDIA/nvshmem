/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_
#define _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_

#include <cuda_runtime.h>

#if !defined __CUDACC_RTC__
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wrapper for CUDA launch configuration passed to nvshmemx_collective_launch_attr.
 * Holds a cudaLaunchConfig_t today; reserved as the extension point for any
 * future NVSHMEM-only launch options.
 *
 * Contract for the embedded cuda_config:
 *   - gridDim, blockDim, dynamicSmemBytes are passed straight through to the launch.
 *   - stream is overridden internally to NVSHMEM's collective-launch stream;
 *     the caller's stream is used only as the fence point (an end-event is
 *     recorded on the internal stream and waited on by the caller's stream).
 *   - attrs/numAttrs are merged with NVSHMEM's required attributes. If the user
 *     sets an attribute that NVSHMEM would otherwise add (e.g. Cooperative),
 *     the user's value wins.
 */
typedef struct nvshmemx_collective_launch_attr {
    cudaLaunchConfig_t cuda_config;
} nvshmemx_collective_launch_attr_t;

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream);
int nvshmemx_collective_launch_attr(const nvshmemx_collective_launch_attr_t *attr, const void *func,
                                    void **args);
int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize);

#ifdef __cplusplus
}
#endif
#endif

#endif
