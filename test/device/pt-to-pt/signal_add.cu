/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "signal_add.cubin"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>
#include "utils.h"

__device__ int error_d;

#define TEST_NVSHMEM_SIG_CUBIN()                                                    \
    void *args_sig[] = {(void *)&remote};                                           \
    CUfunction test_sig_cubin;                                                      \
    init_test_case_kernel(&test_sig_cubin,                                          \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_signal_add_kernel)); \
    CU_CHECK(cuLaunchKernel(test_sig_cubin, 1, 1, 1, 1, 1, 1, 0, 0, args_sig, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void test_nvshmem_signal_add_kernel(uint64_t *remote) {
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    for (int i = 0; i < npes; i++)
        nvshmemx_signal_op(remote, (uint64_t)(mype + 1), NVSHMEM_SIGNAL_ADD, i);

    nvshmem_barrier_all();

    uint64_t remote_val = nvshmem_signal_fetch(remote);
    if (remote_val != (uint64_t)(npes * (npes + 1) / 2)) {
        printf("PE %i observed error\n", mype);
        error_d = 1;
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    uint64_t *remote = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));
    nvshmem_barrier_all();
    if (use_cubin) {
        TEST_NVSHMEM_SIG_CUBIN();
    } else {
        test_nvshmem_signal_add_kernel<<<1, 1>>>(remote);
    }
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
