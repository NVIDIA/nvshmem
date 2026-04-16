/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define N 1

__device__ int errors_d;

__global__ void hello_world(void) {
    int val;

    printf("Device - world PE %d of %d, node PE %d of %d\n", nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD),
           nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD), nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE),
           nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE));

    val = nvshmem_team_my_pe(NVSHMEM_TEAM_INVALID);

    if (val != -1) {
        printf("Error: device nvshmem_team_my_pe(NVSHMEM_TEAM_INVALID) = %d\n", val);
        ++errors_d;
    }

    val = nvshmem_team_n_pes(NVSHMEM_TEAM_INVALID);

    if (val != -1) {
        printf("Error: device nvshmem_team_n_pes(NVSHMEM_TEAM_INVALID) = %d\n", val);
        ++errors_d;
    }
}

int main(int argc, char **argv) {
    int errors_h = 0;
    int val = 0;
    init_wrapper(&argc, &argv);

    nvshmem_barrier_all(); /* Ensure NVSHMEM device init has completed */
    cudaMemcpyToSymbol(errors_d, &val, sizeof(int), 0, cudaMemcpyHostToDevice);

    printf("Host   - world PE %d of %d, node PE %d of %d\n", nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD),
           nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD), nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE),
           nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE));

    val = nvshmem_team_my_pe(NVSHMEM_TEAM_INVALID);

    if (val != -1) {
        printf("Error: host nvshmem_team_my_pe(NVSHMEM_TEAM_INVALID) = %d\n", val);
        ++errors_h;
    }

    val = nvshmem_team_n_pes(NVSHMEM_TEAM_INVALID);

    if (val != -1) {
        printf("Error: host nvshmem_team_n_pes(NVSHMEM_TEAM_INVALID) = %d\n", val);
        ++errors_h;
    }

    hello_world<<<1, N>>>();

    cudaMemcpyFromSymbol(&val, errors_d, sizeof(int), 0, cudaMemcpyDeviceToHost);
    finalize_wrapper();
    return errors_h + val;
}
