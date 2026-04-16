/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_add.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>

#include "utils.h"

__device__ int error_d;

enum op {
    ATOMIC_ADD_UNSIGNED = 0,
    ATOMIC_FETCH_ADD_UNSIGNED,
    ATOMIC_ADD_SIGNED,
    ATOMIC_FETCH_ADD_SIGNED
};

#define REPT_MACRO_FOR_UNSIGNED_TYPES(MACRO_NAME, OP) \
    MACRO_NAME(OP, unsigned int, uint);               \
    MACRO_NAME(OP, unsigned long, ulong);             \
    MACRO_NAME(OP, unsigned long long, ulonglong);    \
    MACRO_NAME(OP, uint32_t, uint32);                 \
    MACRO_NAME(OP, uint64_t, uint64);                 \
    MACRO_NAME(OP, size_t, size);

#define REPT_MACRO_FOR_SIGNED_TYPES(MACRO_NAME, OP) \
    MACRO_NAME(OP, int, int);                       \
    MACRO_NAME(OP, long, long);                     \
    MACRO_NAME(OP, int32_t, int32);

#define TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP)                                      \
    void *args_##TYPENAME##_add_##OP[] = {(void *)&remote, (void *)&value, (void *)&expected}; \
    CUfunction test_##TYPENAME##_add_##OP##_cubin;                                             \
    init_test_case_kernel(&test_##TYPENAME##_add_##OP##_cubin,                                 \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_##TYPENAME##_##OP##_kernel));   \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_add_##OP##_cubin, 1, 1, 1, 1, 1, 1, 0, 0,        \
                            args_##TYPENAME##_add_##OP, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif
/* Each PE adds value to remote. In the fetch case, we compare old against (expected * npes) for
   Each PE. We have to use a range since we don't know in which order the PEs will be executed.
   At the end, we confirm that remote contains (expected * npes). */
#define TEST_NVSHMEM_ATOMIC_ADD_KERNEL(OP, TYPE, TYPENAME)                                         \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(TYPE *remote, TYPE value,              \
                                                            TYPE expected) {                       \
        TYPE old;                                                                                  \
        const int mype = nvshmem_my_pe();                                                          \
        const int npes = nvshmem_n_pes();                                                          \
        for (int i = 0; i < npes; i++) {                                                           \
            if (OP == ATOMIC_ADD_UNSIGNED || OP == ATOMIC_ADD_SIGNED) {                            \
                nvshmem_##TYPENAME##_atomic_add(remote, value, i);                                 \
            } else if (OP == ATOMIC_FETCH_ADD_UNSIGNED || OP == ATOMIC_FETCH_ADD_SIGNED) {         \
                old = nvshmem_##TYPENAME##_atomic_fetch_add(remote, value, i);                     \
                if ((value > 0) && (old >= (expected * npes))) {                                   \
                    printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, #TYPE);  \
                    printf("found = " NVSHPRI_##TYPENAME ", expected < " NVSHPRI_##TYPENAME "\n",  \
                           old, expected *npes);                                                   \
                    error_d++;                                                                     \
                } else if ((value <= 0) && (old < (expected * npes))) {                            \
                    printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, #TYPE);  \
                    printf("found = " NVSHPRI_##TYPENAME ", expected >= " NVSHPRI_##TYPENAME "\n", \
                           old, (expected * npes));                                                \
                    error_d++;                                                                     \
                }                                                                                  \
            } else {                                                                               \
                printf("Invalid operation (%d)\n", OP);                                            \
                assert(0);                                                                         \
            }                                                                                      \
        }                                                                                          \
        nvshmem_barrier_all();                                                                     \
        if (*remote != expected * npes) {                                                          \
            printf("PE %i observed error with TEST_NVSHMEM_ADD_KERNEL(%s, %s)\n", mype, #OP,       \
                   #TYPE);                                                                         \
            printf("found = " NVSHPRI_##TYPENAME ", expected = " NVSHPRI_##TYPENAME "\n", *remote, \
                   expected);                                                                      \
            error_d = 1;                                                                           \
        }                                                                                          \
    }
REPT_MACRO_FOR_UNSIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_KERNEL, ATOMIC_ADD_UNSIGNED)
REPT_MACRO_FOR_SIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_KERNEL, ATOMIC_ADD_SIGNED)
REPT_MACRO_FOR_UNSIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_KERNEL, ATOMIC_FETCH_ADD_UNSIGNED)
REPT_MACRO_FOR_SIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_KERNEL, ATOMIC_FETCH_ADD_SIGNED)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_ADD_UNSIGNED(OP, TYPE, TYPENAME)                            \
    do {                                                                                \
        TYPE value = 5128;                                                              \
        TYPE expected = 5128;                                                           \
        TYPE *remote = (TYPE *)nvshmem_calloc(1, sizeof(TYPE));                         \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
        value = 128;                                                                    \
        expected = 5256;                                                                \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
    } while (0)

#define TEST_NVSHMEM_ATOMIC_ADD_SIGNED(OP, TYPE, TYPENAME)                              \
    do {                                                                                \
        TYPE value = 5128;                                                              \
        TYPE expected = 5128;                                                           \
        TYPE *remote = (TYPE *)nvshmem_calloc(1, sizeof(TYPE));                         \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
        value = -10256;                                                                 \
        expected = -5128;                                                               \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
        value = 5128;                                                                   \
        expected = 0;                                                                   \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
        value = -512;                                                                   \
        expected = -512;                                                                \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
        value = -1024;                                                                  \
        expected = -1536;                                                               \
        nvshmem_barrier_all();                                                          \
        if (use_cubin) {                                                                \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                          \
        } else {                                                                        \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                               \
        cudaDeviceSynchronize();                                                        \
    } while (0)

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    REPT_MACRO_FOR_UNSIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_UNSIGNED, ATOMIC_ADD_UNSIGNED)
    REPT_MACRO_FOR_UNSIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_UNSIGNED, ATOMIC_FETCH_ADD_UNSIGNED)

    REPT_MACRO_FOR_SIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_SIGNED, ATOMIC_ADD_SIGNED)
    REPT_MACRO_FOR_SIGNED_TYPES(TEST_NVSHMEM_ATOMIC_ADD_SIGNED, ATOMIC_FETCH_ADD_SIGNED)

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}