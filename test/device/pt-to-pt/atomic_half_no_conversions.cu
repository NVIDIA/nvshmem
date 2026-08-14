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

static __device__ __forceinline__ __half half_from_bits(uint16_t bits) {
    const __half_raw raw = {bits};
    return __half(raw);
}

static __device__ __forceinline__ uint16_t half_bits(__half value) {
    return static_cast<__half_raw>(value).x;
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void test_nvshmem_half_atomic_no_conversions_kernel(__half *remote,
                                                               size_t dynamic_smem_size) {
    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    const int mype = nvshmem_my_pe();
    const __half value = __float2half_rn(1.0f);
    constexpr uint16_t negative_zero_bits = 0x8000;
    constexpr uint16_t nan_payload_bits = 0x7d55;

    remote[0] = half_from_bits(0);
    remote[1] = half_from_bits(negative_zero_bits);
    nvshmem_barrier_all();

    __half old = nvshmemx_half_atomic_fetch_add(remote, value, mype);

    if (__half2float(old) < 0.0f) {
        printf("PE %d observed invalid old value for half fetch_add with conversions disabled\n",
               mype);
        error_d = 1;
    }

    nvshmem_barrier_all();

    if (half_bits(remote[1]) != negative_zero_bits) {
        printf("PE %d half fetch_add changed adjacent -0 bits from 0x%04x to 0x%04x\n", mype,
               (unsigned int)negative_zero_bits, (unsigned int)half_bits(remote[1]));
        error_d = 1;
    }

    remote[1] = half_from_bits(nan_payload_bits);
    nvshmem_barrier_all();

    nvshmemx_half_atomic_add(remote, value, mype);

    nvshmem_barrier_all();

    if (half_bits(remote[1]) != nan_payload_bits) {
        printf("PE %d half add changed adjacent NaN bits from 0x%04x to 0x%04x\n", mype,
               (unsigned int)nan_payload_bits, (unsigned int)half_bits(remote[1]));
        error_d = 1;
    }

    if (__half2float(*remote) < 2.0f) {
        printf("PE %d observed invalid final value for half atomics with conversions disabled\n",
               mype);
        error_d = 1;
    }
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;

    init_wrapper(&argc, &argv);

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    __half *remote = (__half *)nvshmem_calloc(2, sizeof(__half));
    nvshmem_barrier_all();

    CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_nvshmem_half_atomic_no_conversions_kernel,
                                      _dynamic_smem_size);
    test_nvshmem_half_atomic_no_conversions_kernel<<<1, 1, _dynamic_smem_size>>>(
        remote, _dynamic_smem_size);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    nvshmem_free(remote);
    finalize_wrapper();
    return ret_val;
}
