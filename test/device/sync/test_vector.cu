/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

__global__ void nvshmem_test_all_vector_test_kernel(uint64_t *ivars, int *status,
                                                    uint64_t *cmp_values) {
    const int my_pe = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    int i = 0;
    int expected_sum = 0;
    int total_sum = 0;

    for (i = 0; i < npes; i++) {
        nvshmemx_signal_op(&ivars[my_pe], (uint64_t)my_pe, NVSHMEM_SIGNAL_SET, i);
        cmp_values[i] = i;
    }

    expected_sum = (npes - 1) * npes / 2;
    size_t ncompleted = 0;
    while (ncompleted == 0) {
        ncompleted =
            nvshmem_uint64_test_all_vector(ivars, npes, status, NVSHMEM_CMP_EQ, cmp_values);
    }

    for (i = 0; i < npes; i++) {
        total_sum += ivars[i];
    }

    if (expected_sum != total_sum) {
        printf("Incorrect total_sum = %d, expected sum = %dn", total_sum, expected_sum);
        error_d = 1;
    }
}

void test_nvshmem_test_all_vector(void) {
    const int npes = nvshmem_n_pes();
    uint64_t *ivars = (uint64_t *)nvshmem_calloc(npes, sizeof(uint64_t));
    int *status;
    cudaMalloc((void **)&status, npes * sizeof(int));
    cudaMemset(status, 0, npes * sizeof(int));
    uint64_t *cmp_values;
    cudaMalloc((void **)&cmp_values, npes * sizeof(uint64_t));
    nvshmem_barrier_all();
    nvshmem_test_all_vector_test_kernel<<<1, 1>>>(ivars, status, cmp_values);
    cudaDeviceSynchronize();
}

__global__ void nvshmem_test_any_vector_test_kernel(uint64_t *ivars, int *status,
                                                    uint64_t *cmp_values) {
    const int my_pe = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    int i = 0;
    int total_sum = 0;

    for (i = 0; i < npes; i++) {
        nvshmemx_signal_op(&ivars[my_pe], (uint64_t)my_pe, NVSHMEM_SIGNAL_SET, i);
        cmp_values[i] = i;
    }

    int expected_sum = (npes - 1) * npes / 2;

    int ncompleted = 0;
    while (ncompleted < npes) {
        size_t ndone =
            nvshmem_uint64_test_any_vector(ivars, npes, status, NVSHMEM_CMP_EQ, cmp_values);
        if (ndone != SIZE_MAX) {
            status[ndone] = 1;
            total_sum += ivars[ndone];
            ncompleted++;
        }
    }

    if (expected_sum != total_sum) {
        printf("Incorrect total_sum = %d, expected_sum = %dn", total_sum, expected_sum);
        error_d = 1;
    }
}

void test_nvshmem_test_any_vector(void) {
    const int npes = nvshmem_n_pes();
    uint64_t *ivars = (uint64_t *)nvshmem_calloc(npes, sizeof(uint64_t));
    int *status;
    cudaMalloc((void **)&status, npes * sizeof(int));
    cudaMemset(status, 0, npes * sizeof(int));
    uint64_t *cmp_values;
    cudaMalloc((void **)&cmp_values, npes * sizeof(uint64_t));
    nvshmem_barrier_all();
    nvshmem_test_any_vector_test_kernel<<<1, 1>>>(ivars, status, cmp_values);
    cudaDeviceSynchronize();
}

__global__ void nvshmem_test_some_vector_test_kernel(uint64_t *ivars, size_t *indices, int *status,
                                                     uint64_t *cmp_values) {
    const int my_pe = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    int i = 0;
    int expected_sum = 0;
    int total_sum = 0;

    for (i = 0; i < npes; i++) {
        nvshmemx_signal_op(&ivars[my_pe], (uint64_t)my_pe, NVSHMEM_SIGNAL_SET, i);
        cmp_values[i] = i;
    }

    expected_sum = (npes - 1) * npes / 2;

    int ncompleted = 0;
    while (ncompleted < npes) {
        size_t ndone = nvshmem_uint64_test_some_vector(ivars, npes, indices, status, NVSHMEM_CMP_EQ,
                                                       cmp_values);
        if (ndone != 0) {
            for (size_t j = 0; j < ndone; j++) {
                total_sum += ivars[indices[j]];
                status[indices[j]] = 1;
            }
            ncompleted += ndone;
        }
    }

    if (expected_sum != total_sum) {
        printf("Incorrect total_sum = %d, expected_sum = %dn", total_sum, expected_sum);
        error_d = 1;
    }
}

void test_nvshmem_test_some_vector(void) {
    const int npes = nvshmem_n_pes();
    uint64_t *ivars = (uint64_t *)nvshmem_calloc(npes, sizeof(uint64_t));
    int *status;
    cudaMalloc((void **)&status, npes * sizeof(int));
    cudaMemset(status, 0, npes * sizeof(int));
    uint64_t *cmp_values;
    cudaMalloc((void **)&cmp_values, npes * sizeof(uint64_t));

    size_t *indices;
    cudaMalloc((void **)&indices, npes * sizeof(size_t));
    nvshmem_barrier_all();
    nvshmem_test_some_vector_test_kernel<<<1, 1>>>(ivars, indices, status, cmp_values);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
    int ret_val = 0;
    init_wrapper(&argc, &argv);

    int zero = 0;
    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    test_nvshmem_test_all_vector();
    test_nvshmem_test_any_vector();
    test_nvshmem_test_some_vector();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();

    return ret_val;
}
