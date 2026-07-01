/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

__global__ void test_nvshmem_test_kernel(uint64_t *remote, int mype, int npes) {
    nvshmemx_signal_op(remote, (uint64_t)mype + 1, NVSHMEM_SIGNAL_SET, (mype + 1) % npes);

    while (!nvshmem_uint64_test(remote, NVSHMEM_CMP_NE, 0));
    if (*remote != ((uint64_t)mype + npes - 1) % npes + 1) {
        printf("PE %d received incorrect value", mype);
        error_d = 1;
    }
}

int main(int argc, char *argv[]) {
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    int zero = 0, ret_val;
    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    uint64_t *remote;
    if (use_mmap) {
        remote =
            (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t), _mem_handle_type, use_egm, true);
    } else {
        remote = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
        cudaMemset(remote, 0, sizeof(uint64_t));
    }
    nvshmem_barrier_all();
    /* The kernel is designed for 1 thread */
    test_nvshmem_test_kernel<<<1, 1>>>(remote, mype, npes);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();

    return ret_val;
}
