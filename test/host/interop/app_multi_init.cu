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

#define INIT_DEFAULT_ITERS 150

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
    const char *test_num_iter = getenv("NVSHMEMTEST_INIT_NUM_ITERS");
    int num_iters = test_num_iter ? atoi(test_num_iter) : INIT_DEFAULT_ITERS;

    nvshmem_init();

    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int dev_count;
    int status = 0;

    cudaGetDeviceCount(&dev_count);

    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    cudaSetDevice(mype_node / npes_per_gpu);

    nvshmem_barrier_all();

    status = app_dowork();
    if (status) {
        fprintf(stderr, "app: app_dowork failed before initializing libraries.\n");
        nvshmem_finalize();
        goto out;
    }

    for (int i = 0; i < num_iters; i++) {
        simplelib1_init();
        simplelib2_init();
        status = app_dowork();
        status += simplelib1_dowork();
        status += simplelib2_dowork();
        simplelib2_finalize();
        simplelib1_finalize();
        if (status) {
            fprintf(stderr, "dowork failed in the first loop at iter: %d.\n", i);
            nvshmem_finalize();
            goto out;
        }
    }

    nvshmem_barrier_all();
    nvshmem_finalize();

    for (int i = 0; i < num_iters; i++) {
        simplelib1_init();
        status = simplelib1_dowork();
        simplelib2_init();
        status += simplelib2_dowork();
        simplelib2_finalize();
        status += simplelib1_dowork();
        simplelib1_finalize();
        if (status) {
            fprintf(stderr, "dowork failed in the second loop at iter: %d.\n", i);
            goto out;
        }
    }
out:
    return status;
}
