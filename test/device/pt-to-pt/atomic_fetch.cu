/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_fetch.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "utils.h"
#include <assert.h>
__device__ int error_d;

enum op { ATOMIC_FETCH = 0 };
#define NUM_FETCHES (1 << 14)

#define REPT_MACRO_FOR_TYPES(MACRO_NAME, OP)       \
    MACRO_NAME(OP, int, int);                      \
    MACRO_NAME(OP, long, long);                    \
    MACRO_NAME(OP, unsigned int, uint);            \
    MACRO_NAME(OP, unsigned long, ulong);          \
    MACRO_NAME(OP, unsigned long long, ulonglong); \
    MACRO_NAME(OP, int32_t, int32);                \
    MACRO_NAME(OP, uint32_t, uint32);              \
    MACRO_NAME(OP, uint64_t, uint64);              \
    MACRO_NAME(OP, size_t, size);

#define TEST_NVSHMEM_ATOMIC_FETCH_CUBIN(TYPENAME, TYPE, OP)                                  \
    void *args_##TYPENAME##_fetch_##OP[] = {(void *)&remote, (void *)&fetched_values};       \
    CUfunction test_##TYPENAME##_fetch_##OP##_cubin;                                         \
    init_test_case_kernel(&test_##TYPENAME##_fetch_##OP##_cubin,                             \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_##TYPENAME##_##OP##_kernel)); \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_fetch_##OP##_cubin, 1, 1, 1, 1, 1, 1, 0, 0,    \
                            args_##TYPENAME##_fetch_##OP, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define TEST_NVSHMEM_ATOMIC_FETCH_KERNEL(OP, TYPE, TYPENAME)                                      \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(TYPE *remote, TYPE *fetched_values) { \
        const int mype = nvshmem_my_pe();                                                         \
        const int npes = nvshmem_n_pes();                                                         \
        for (int i = threadIdx.x; i < NUM_FETCHES; i += blockDim.x) remote[i] = (TYPE)(mype + i); \
        nvshmemx_barrier_all_block();                                                             \
        for (int i = threadIdx.x; i < NUM_FETCHES; i += blockDim.x) {                             \
            switch (OP) {                                                                         \
                case ATOMIC_FETCH:                                                                \
                    fetched_values[i] =                                                           \
                        nvshmem_##TYPENAME##_atomic_fetch(&remote[i], (mype + i + 1) % npes);     \
                    break;                                                                        \
                default:                                                                          \
                    printf("Invalid operation (%d)\n", OP);                                       \
                    assert(0);                                                                    \
            }                                                                                     \
        }                                                                                         \
        __syncthreads();                                                                          \
        for (int i = threadIdx.x; i < NUM_FETCHES; i += blockDim.x) {                             \
            if (fetched_values[i] != (TYPE)(((mype + i + 1) % npes) + i)) {                       \
                printf(                                                                           \
                    "pe %d received incorrect value at idx %d with "                              \
                    "test_nvshmem_atomic_fetch(%s, %s)\n",                                        \
                    mype, i, #OP, #TYPE);                                                         \
                error_d = 1;                                                                      \
            }                                                                                     \
        }                                                                                         \
    }
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_FETCH_KERNEL, ATOMIC_FETCH)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_FETCH(OP, TYPE, TYPENAME)                                           \
    do {                                                                                        \
        TYPE *remote;                                                                           \
        if (use_mmap) {                                                                         \
            remote = (TYPE *)allocate_mmap_buffer(sizeof(TYPE) * NUM_FETCHES, _mem_handle_type, \
                                                  use_egm);                                     \
            DEBUG_PRINT("Allocating mmaped buffer\n");                                          \
        } else {                                                                                \
            remote = (TYPE *)nvshmem_malloc(sizeof(TYPE) * NUM_FETCHES);                        \
        }                                                                                       \
        TYPE *fetched_values;                                                                   \
        cudaMalloc(&fetched_values, sizeof(TYPE) * NUM_FETCHES);                                \
        nvshmem_barrier_all();                                                                  \
        if (use_cubin) {                                                                        \
            TEST_NVSHMEM_ATOMIC_FETCH_CUBIN(TYPENAME, TYPE, OP);                                \
        } else {                                                                                \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, fetched_values);          \
        }                                                                                       \
        cudaDeviceSynchronize();                                                                \
        cudaFree(fetched_values);                                                               \
        if (use_mmap) {                                                                         \
            free_mmap_buffer(remote);                                                           \
        } else {                                                                                \
            nvshmem_free(remote);                                                               \
        }                                                                                       \
    } while (0)

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_FETCH, ATOMIC_FETCH)

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
