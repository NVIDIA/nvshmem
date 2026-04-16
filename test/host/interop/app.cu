/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "simplelib1.h"
#include "simplelib2.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>

__device__ int num_errors_d;
__global__ void app_nvshmem_kernel(int *array) {
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;
    nvshmem_int_p(array, my_pe, next_pe);
    nvshmem_barrier_all();

    if (array[0] != prev_pe) {
        printf("app: incorrect value found, expected = %d, found = %d\n", prev_pe, array[0]);
        num_errors_d = 1;
    }
}

int app_dowork() {
    int *array = (int *)nvshmem_calloc(1, sizeof(int));
    int num_errors = 0;
    app_nvshmem_kernel<<<1, 1>>>(array);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&num_errors, num_errors_d, sizeof(int));
    nvshmem_free(array);
    return num_errors;
}

int main(int argc, char **argv) {
    nvshmem_init();
    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    cudaSetDevice(mype_node / npes_per_gpu);
    nvshmem_barrier_all();

    simplelib1_init();
    simplelib2_init();

    int num_errors = app_dowork();
    num_errors += simplelib1_dowork();
    num_errors += simplelib2_dowork();

    nvshmem_finalize();
    simplelib1_finalize();
    simplelib2_finalize();

    return num_errors;
}
