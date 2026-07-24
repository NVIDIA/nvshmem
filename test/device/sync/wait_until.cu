/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

__global__ void test_nvshmem_uint64_wait_until_kernel(uint64_t *remote, int mype, int npes) {
    nvshmemx_signal_op(remote, (uint64_t)mype + 1, NVSHMEM_SIGNAL_SET, (mype + 1) % npes);

    nvshmem_uint64_wait_until(remote, NVSHMEM_CMP_NE, 0);

    if (*remote != ((uint64_t)mype + npes - 1) % npes + 1) {
        printf("PE %d received incorrect value", mype);
        error_d = 1;
    }
}

#define WAIT_UNTIL_X_KERNEL(WAIT_UNTIL_X)                                                      \
    __global__ void test_nvshmem_uint64_##WAIT_UNTIL_X##_kernel(uint64_t *remote, int mype,    \
                                                                int npes) {                    \
        nvshmemx_signal_op(remote, (uint64_t)mype + 1, NVSHMEM_SIGNAL_SET, (mype + 1) % npes); \
        int status = 0;                                                                        \
                                                                                               \
        nvshmem_uint64_##WAIT_UNTIL_X(remote, 1, &status, NVSHMEM_CMP_NE, 0);                  \
                                                                                               \
        if (*remote != ((uint64_t)mype + npes - 1) % npes + 1) {                               \
            printf("PE %d received incorrect value", mype);                                    \
            error_d = 1;                                                                       \
        }                                                                                      \
    }

WAIT_UNTIL_X_KERNEL(wait_until_all)
WAIT_UNTIL_X_KERNEL(wait_until_any)

__global__ void test_nvshmem_uint64_wait_until_some_kernel(uint64_t *remote, int mype, int npes) {
    nvshmemx_signal_op(remote, (uint64_t)mype + 1, NVSHMEM_SIGNAL_SET, (mype + 1) % npes);
    int status = 0;
    size_t indices;

    nvshmem_uint64_wait_until_some(remote, 1, &indices, &status, NVSHMEM_CMP_NE, 0);

    if (*remote != ((uint64_t)mype + npes - 1) % npes + 1) {
        printf("PE %d received incorrect value", mype);
        error_d = 1;
    }
}

#define TEST_NVSHMEM_WAIT_UNTIL_X(WAIT_UNTIL_X)                                                    \
    do {                                                                                           \
        const int mype = nvshmem_my_pe();                                                          \
        const int npes = nvshmem_n_pes();                                                          \
        uint64_t *remote;                                                                          \
        if (use_mmap) {                                                                            \
            remote = (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t), _mem_handle_type, use_egm, \
                                                      true);                                       \
        } else {                                                                                   \
            remote = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));                                 \
            cudaMemset(remote, 0, sizeof(uint64_t));                                               \
        }                                                                                          \
        nvshmem_barrier_all();                                                                     \
        /* The kernel is designed for 1 thread */                                                  \
        test_nvshmem_uint64_##WAIT_UNTIL_X##_kernel<<<1, 1>>>(remote, mype, npes);                 \
        cudaDeviceSynchronize();                                                                   \
    } while (0)

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    TEST_NVSHMEM_WAIT_UNTIL_X(wait_until);
    TEST_NVSHMEM_WAIT_UNTIL_X(wait_until_all);
    TEST_NVSHMEM_WAIT_UNTIL_X(wait_until_any);
    TEST_NVSHMEM_WAIT_UNTIL_X(wait_until_some);

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
