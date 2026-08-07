/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_inc.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>
#include "utils.h"

__device__ int error_d;

enum op { ATOMIC_INC = 0, ATOMIC_FETCH_INC };

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

#define TEST_NVSHMEM_ATOMIC_INC_CUBIN(TYPENAME, TYPE, OP)                                    \
    void *args_##TYPENAME##_inc_##OP[] = {(void *)&remote, (void *)&_dynamic_smem_size};     \
    CUfunction test_##TYPENAME##_inc_##OP##_cubin;                                           \
    init_test_case_kernel(&test_##TYPENAME##_inc_##OP##_cubin,                               \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_##TYPENAME##_##OP##_kernel)); \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_inc_##OP##_cubin, 1, 1, 1, 1, 1, 1,            \
                            _dynamic_smem_size, 0, args_##TYPENAME##_inc_##OP, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define TEST_NVSHMEM_ATOMIC_INC_KERNEL(OP, TYPE, TYPENAME)                                    \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(TYPE *remote,                     \
                                                            size_t dynamic_smem_size) {       \
        TYPE old;                                                                             \
        const int mype = nvshmem_my_pe();                                                     \
        const int npes = nvshmem_n_pes();                                                     \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                            \
        *remote = (TYPE)0;                                                                    \
        nvshmem_barrier_all();                                                                \
        for (int i = 0; i < npes; i++) switch (OP) {                                          \
                case ATOMIC_INC:                                                              \
                    nvshmem_##TYPENAME##_atomic_inc(remote, i);                               \
                    break;                                                                    \
                case ATOMIC_FETCH_INC:                                                        \
                    old = nvshmem_##TYPENAME##_atomic_fetch_inc(remote, i);                   \
                    if (old > (TYPE)npes) {                                                   \
                        printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, \
                               #TYPE);                                                        \
                        error_d = 1;                                                          \
                    }                                                                         \
                    break;                                                                    \
                default:                                                                      \
                    printf("Invalid operation (%d)\n", OP);                                   \
                    assert(0);                                                                \
            }                                                                                 \
        nvshmem_barrier_all();                                                                \
        if (*remote != (TYPE)npes) {                                                          \
            printf("PE %i observed error with TEST_NVSHMEM_ATOMIC_INC(%s, %s)\n", mype, #OP,  \
                   #TYPE);                                                                    \
            error_d = 1;                                                                      \
        }                                                                                     \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                         \
    }
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_INC_KERNEL, ATOMIC_INC)
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_INC_KERNEL, ATOMIC_FETCH_INC)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_INC(OP, TYPE, TYPENAME)                                         \
    do {                                                                                    \
        TYPE *remote;                                                                       \
        if (use_mmap) {                                                                     \
            remote = (TYPE *)allocate_mmap_buffer(sizeof(TYPE), _mem_handle_type, use_egm); \
            DEBUG_PRINT("Allocating mmaped buffer\n");                                      \
        } else {                                                                            \
            remote = (TYPE *)nvshmem_malloc(sizeof(TYPE));                                  \
        }                                                                                   \
        nvshmem_barrier_all();                                                              \
        if (use_cubin) {                                                                    \
            TEST_NVSHMEM_ATOMIC_INC_CUBIN(TYPENAME, TYPE, OP);                              \
        } else {                                                                            \
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_nvshmem_##TYPENAME##_##OP##_kernel,      \
                                              _dynamic_smem_size);                          \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1, _dynamic_smem_size>>>(          \
                remote, _dynamic_smem_size);                                                \
        }                                                                                   \
        cudaDeviceSynchronize();                                                            \
        nvshmem_barrier_all();                                                              \
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

    REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_INC, ATOMIC_INC)
    // REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_INC, ATOMIC_FETCH_INC)

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
