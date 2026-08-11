/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

#define N 100

__global__ void test_kernel(int *my_data, int *all_data, uint64_t *flags, int *status, int mype,
                            int npes) {
    for (int i = 0; i < N; i++) {
        my_data[i] = mype * N + i;
    }

    for (int i = 0; i < npes; i++) {
        nvshmem_int_put_nbi(&all_data[mype * N], my_data, N, i);
    }

    nvshmem_fence();

    for (int i = 0; i < npes; i++) {
        nvshmemx_signal_op(&flags[mype], 1, NVSHMEM_SIGNAL_SET, i);
    }

    int total_sum = 0;

    size_t completed_idx;
    for (int i = 0; i < npes; i++) {
        completed_idx = nvshmem_uint64_wait_until_any(flags, npes, status, NVSHMEM_CMP_NE, 0);
        for (int j = 0; j < N; j++) {
            total_sum += all_data[completed_idx * N + j];
        }
        status[completed_idx] = 1;
    }

    /* Check the flags array */
    for (int i = 0; i < npes; i++) {
        if (flags[i] != 1) {
            printf("Incorrect flag value = %lu, expected = %d\n", flags[i], 1);
            error_d = 1;
        }
    }

    /* check result */
    int M = N * npes - 1;
    if (total_sum != M * (M + 1) / 2) {
        printf("Incorrect total_sum = %d, expected = %d\n", total_sum, M * (M + 1) / 2);
        error_d = 2;
    }

    /* Sanity check the case with NULL status array */
    completed_idx = nvshmem_uint64_wait_until_any(flags, npes, NULL, NVSHMEM_CMP_EQ, 1);

    if (completed_idx >= (size_t)npes) {
        printf("Incorrect completed_idx = %zu, expected < %d\n", completed_idx, npes);
        error_d = 3;
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

    int *my_data, *all_data;
    uint64_t *flags;
    if (use_mmap) {
        my_data = (int *)allocate_mmap_buffer(N * sizeof(int), _mem_handle_type, use_egm);
        all_data = (int *)allocate_mmap_buffer(N * npes * sizeof(int), _mem_handle_type, use_egm);
        flags = (uint64_t *)allocate_mmap_buffer(npes * sizeof(uint64_t), _mem_handle_type, use_egm,
                                                 true);
    } else {
        my_data = (int *)nvshmem_malloc(N * sizeof(int));
        all_data = (int *)nvshmem_malloc(N * npes * sizeof(int));
        flags = (uint64_t *)nvshmem_malloc(npes * sizeof(uint64_t));
        cudaMemset(flags, 0, npes * sizeof(uint64_t));
    }
    int *status;
    cudaMalloc(&status, npes * sizeof(int));
    cudaMemset(status, 0, npes * sizeof(int));

    nvshmem_barrier_all();

    cudaDeviceSynchronize();
    test_kernel<<<1, 1>>>(my_data, all_data, flags, status, mype, npes);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0, cudaMemcpyDeviceToHost);

    if (use_mmap) {
        free_mmap_buffer(all_data);
        free_mmap_buffer(my_data);
        free_mmap_buffer(flags);
    } else {
        nvshmem_free(all_data);
        nvshmem_free(my_data);
        nvshmem_free(flags);
    }
    cudaFree(status);
    finalize_wrapper();

    return ret_val;
}
