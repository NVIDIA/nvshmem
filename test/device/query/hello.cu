/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define N 4

__global__ void hello_world(void) {
    printf("Hello World from device PE %d <%d> of %d\n", nvshmem_my_pe(), threadIdx.x,
           nvshmem_n_pes());
}

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    nvshmem_barrier_all(); /* Ensure NVSHMEM device init has completed */

    printf("Hello World from host PE %d of %d\n", nvshmem_my_pe(), nvshmem_n_pes());

    hello_world<<<1, N>>>();

    finalize_wrapper();
    return 0;
}
