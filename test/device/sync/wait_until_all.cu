/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

#define N 100

__global__ void test_kernel(uint64_t *flags, int *status, int mype, int npes) {
    for (int i = 0; i < npes; i++) {
        nvshmemx_signal_op(&flags[mype], 1, NVSHMEM_SIGNAL_SET, i);
    }
    nvshmem_quiet();
    nvshmem_uint64_wait_until_all(flags, npes, status, NVSHMEM_CMP_EQ, 1);

    /* Check the flags array */
    for (int i = 0; i < npes; i++) {
        if (flags[i] != 1) {
            printf("Incorrect flag value = %lu, expected = %d\n", flags[i], 1);
            error_d = 1;
        }
    }
}

int main(int argc, char **argv) {
    int ret_val = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    int zero = 0;
    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    uint64_t *flags;
    if (use_mmap) {
        flags = (uint64_t *)allocate_mmap_buffer(npes * sizeof(uint64_t), _mem_handle_type, use_egm,
                                                 true);
    } else {
        flags = (uint64_t *)nvshmem_malloc(npes * sizeof(uint64_t));
        cudaMemset(flags, 0, npes * sizeof(uint64_t));
    }
    int *status = NULL;
    nvshmem_barrier_all();

    cudaDeviceSynchronize();
    test_kernel<<<1, 1>>>(flags, status, mype, npes);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);
    if (use_mmap) {
        free_mmap_buffer(flags);
    } else {
        nvshmem_free(flags);
    }
    finalize_wrapper();

    return ret_val;
}
