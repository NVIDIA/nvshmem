/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_xor.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>
#include "utils.h"

__device__ int error_d;

enum op { ATOMIC_XOR = 0, ATOMIC_FETCH_XOR };

#define REPT_MACRO_FOR_TYPES(MACRO_NAME, OP)       \
    MACRO_NAME(OP, unsigned int, uint);            \
    MACRO_NAME(OP, unsigned long, ulong);          \
    MACRO_NAME(OP, unsigned long long, ulonglong); \
    MACRO_NAME(OP, int32_t, int32);                \
    MACRO_NAME(OP, uint32_t, uint32);              \
    MACRO_NAME(OP, uint64_t, uint64);

#define TEST_NVSHMEM_ATOMIC_XOR_CUBIN(TYPENAME, TYPE, OP)                                    \
    void *args_##TYPENAME##_xor_##OP[] = {(void *)&remote, (void *)&_dynamic_smem_size};     \
    CUfunction test_##TYPENAME##_xor_##OP##_cubin;                                           \
    init_test_case_kernel(&test_##TYPENAME##_xor_##OP##_cubin,                               \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_##TYPENAME##_##OP##_kernel)); \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_xor_##OP##_cubin, 1, 1, 1, 1, 1, 1,            \
                            _dynamic_smem_size, 0, args_##TYPENAME##_xor_##OP, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define TEST_NVSHMEM_ATOMIC_XOR_KERNEL(OP, TYPE, TYPENAME)                                        \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(TYPE *remote,                         \
                                                            size_t dynamic_smem_size) {           \
        const int mype = nvshmem_my_pe();                                                         \
        const int npes = nvshmem_n_pes();                                                         \
        *remote = ~(TYPE)0;                                                                       \
        TYPE old = (TYPE)0;                                                                       \
        if ((size_t)npes - 1 > sizeof(TYPE)) return; /* Avoid overflow */                         \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                \
        nvshmem_barrier_all();                                                                    \
        for (int i = 0; i < npes; i++) {                                                          \
            switch (OP) {                                                                         \
                case ATOMIC_XOR:                                                                  \
                    nvshmem_##TYPENAME##_atomic_xor(remote, (TYPE)(1LLU << mype), i);             \
                    break;                                                                        \
                case ATOMIC_FETCH_XOR:                                                            \
                    old = nvshmem_##TYPENAME##_atomic_fetch_xor(remote, (TYPE)(1LLU << mype), i); \
                    if (((old ^ (TYPE)(1LLU << mype)) & (TYPE)(1LLU << mype)) != 0) {             \
                        printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP,     \
                               #TYPE);                                                            \
                        error_d = 1;                                                              \
                    }                                                                             \
                    break;                                                                        \
                default:                                                                          \
                    printf("Invalid operation (%d)\n", OP);                                       \
                    error_d = 1;                                                                  \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        if (*remote != ~(TYPE)((1LLU << npes) - 1LLU)) {                                          \
            printf("PE %i observed error with TEST_NVSHMEM_ATOMIC_XOR(%s, %s)\n", mype, #OP,      \
                   #TYPE);                                                                        \
            error_d = 1;                                                                          \
        }                                                                                         \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                             \
    }
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_XOR_KERNEL, ATOMIC_XOR)
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_XOR_KERNEL, ATOMIC_FETCH_XOR)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_XOR(OP, TYPE, TYPENAME)                                         \
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
            TEST_NVSHMEM_ATOMIC_XOR_CUBIN(TYPENAME, TYPE, OP);                              \
        } else {                                                                            \
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_nvshmem_##TYPENAME##_##OP##_kernel,      \
                                              _dynamic_smem_size);                          \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1, _dynamic_smem_size>>>(          \
                remote, _dynamic_smem_size);                                                \
        }                                                                                   \
        cudaDeviceSynchronize();                                                            \
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

    REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_XOR, ATOMIC_XOR)
    REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_XOR, ATOMIC_FETCH_XOR)

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
