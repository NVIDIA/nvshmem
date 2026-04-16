/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>
#include "utils.h"

__device__ int error_d;

__global__ void test_kernel(uint64_t *flags, int *status, int *found, int mype, int npes) {
    for (int i = 0; i < npes; i++) {
        nvshmemx_signal_op(&flags[mype], 1, NVSHMEM_SIGNAL_SET, i);
    }

    int ncompleted = 0;
    size_t completed_idx;

    while (ncompleted < npes) {
        completed_idx = nvshmem_uint64_test_any(flags, npes, status, NVSHMEM_CMP_EQ, 1);
        if (completed_idx != SIZE_MAX) {
            ncompleted++;
            status[completed_idx] = 1;
        } else {
            /* Overlap some computation here */
        }
    }
    /* Check the flags array */
    for (int i = 0; i < npes; i++) {
        if (flags[i] != 1) error_d = 1;
    }
    /* Sanity check of shmem_test_any's fairness */
    ncompleted = 0;

    while (ncompleted < npes) {
        int idx = nvshmem_uint64_test_any(flags, npes, NULL, NVSHMEM_CMP_EQ, 1);
        if (found[idx] == 0) {
            found[idx] = 1;
            ncompleted++;
        }
    }

    for (int i = 0; i < npes; i++) {
        if (found[i] != 1) {
            error_d = 2;
        }
    }

    /* Sanity check case with NULL status array */
    completed_idx = nvshmem_uint64_test_any(flags, npes, NULL, NVSHMEM_CMP_EQ, 1);

    if (completed_idx >= (size_t)npes) error_d = 3;
}

int main(int argc, char **argv) {
    int ret_val = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

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

    int *status;
    cudaMalloc(&status, npes * sizeof(int));
    cudaMemset(status, 0, npes * sizeof(int));

    int *found;
    cudaMalloc(&found, npes * sizeof(int));
    cudaMemset(found, 0, npes * sizeof(int));

    nvshmem_barrier_all();

    cudaDeviceSynchronize();
    test_kernel<<<1, 1>>>(flags, status, found, mype, npes);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0, cudaMemcpyDeviceToHost);

    cudaFree(found);
    cudaFree(status);
    if (use_mmap) {
        free_mmap_buffer(flags);
    } else {
        nvshmem_free(flags);
    }
    finalize_wrapper();

    return ret_val;
}
