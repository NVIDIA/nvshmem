/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvshmem.h>
#include <cstdio>

__global__ void kernel_nvshmem(int* destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    assert(npes > 0);
    int peer = (mype + 1) % npes;
    nvshmem_int_p(destination, 3 * peer + 14, peer);
    nvshmem_barrier_all();
}
