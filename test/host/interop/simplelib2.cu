/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "nvshmem.h"
#include "nvshmemx.h"
#include "simplelib2.h"

__device__ int num_errors_d;
__global__ void simplelib2_nvshmem_kernel(int *array) {
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;
    nvshmem_int_p(array, my_pe, next_pe);
    nvshmem_barrier_all();

    if (array[0] != prev_pe) {
        printf("simplelib2: incorrect value found, expected = %d, found = %d\n", prev_pe, array[0]);
        num_errors_d = 1;
    }
}

int simplelib2_dowork() {
    int *array = (int *)nvshmem_calloc(1, sizeof(int));
    int num_errors = 0;
    simplelib2_nvshmem_kernel<<<1, 1>>>(array);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&num_errors, num_errors_d, sizeof(int));
    nvshmem_free(array);
    return num_errors;
}

void simplelib2_init() { nvshmem_init(); }

void simplelib2_finalize() { nvshmem_finalize(); }
