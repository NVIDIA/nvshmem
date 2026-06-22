/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_CONVERSIONS__
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "utils.h"

__device__ int error_d;

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void test_nvshmem_half_atomic_no_conversions_kernel(__half *remote) {
    const int mype = nvshmem_my_pe();
    __half value = __float2half_rn(1.0f);
    __half old = nvshmemx_half_atomic_fetch_add(remote, value, mype);

    if (__half2float(old) < 0.0f) {
        printf("PE %d observed invalid old value for half fetch_add with conversions disabled\n",
               mype);
        error_d = 1;
    }

    nvshmem_barrier_all();
    nvshmemx_half_atomic_add(remote, value, mype);

    nvshmem_barrier_all();

    if (__half2float(*remote) < 2.0f) {
        printf("PE %d observed invalid final value for half atomics with conversions disabled\n",
               mype);
        error_d = 1;
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;

    init_wrapper(&argc, &argv);

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    __half *remote = (__half *)nvshmem_calloc(1, sizeof(__half));
    nvshmem_barrier_all();

    test_nvshmem_half_atomic_no_conversions_kernel<<<1, 1>>>(remote);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    nvshmem_free(remote);
    finalize_wrapper();
    return ret_val;
}
