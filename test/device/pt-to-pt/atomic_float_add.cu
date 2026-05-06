/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_float_add.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>

#include "utils.h"

__device__ int error_d;

enum op {
    ATOMIC_ADD_FLOAT = 0,
    ATOMIC_FETCH_ADD_FLOAT
};

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
/* Each PE adds value to remote. In the fetch case, we compare old against expected*npes.
   At the end, we confirm that remote contains expected*npes.
   Comparisons and arithmetic use (float) casts so the macro works with __half. */
#define TEST_NVSHMEM_ATOMIC_ADD_KERNEL(OP, TYPE, TYPENAME)                                        \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(TYPE *remote, TYPE value,             \
                                                            TYPE expected) {                      \
        TYPE old;                                                                                 \
        const int mype = nvshmem_my_pe();                                                         \
        const int npes = nvshmem_n_pes();                                                         \
        for (int i = 0; i < npes; i++) {                                                          \
            if (OP == ATOMIC_ADD_FLOAT) {                                                         \
                nvshmemx_##TYPENAME##_atomic_add(remote, value, i);                               \
            } else if (OP == ATOMIC_FETCH_ADD_FLOAT) {                                            \
                old = nvshmemx_##TYPENAME##_atomic_fetch_add(remote, value, i);                   \
                if (((float)value > 0) && ((float)old >= (float)expected * npes)) {               \
                    printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, #TYPE); \
                    printf("found = %f, expected < %f\n", (double)old,                            \
                           (double)expected * npes);                                               \
                    error_d++;                                                                    \
                } else if (((float)value <= 0) && ((float)old < (float)expected * npes)) {        \
                    printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, #TYPE); \
                    printf("found = %f, expected >= %f\n", (double)old,                           \
                           (double)expected * npes);                                               \
                    error_d++;                                                                    \
                }                                                                                 \
            } else {                                                                              \
                printf("Invalid operation (%d)\n", OP);                                           \
                assert(0);                                                                        \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        float tolerance = 1e-5f * (float)expected * npes;                                         \
        float diff = (float)*remote - (float)expected * npes;                                     \
        if (diff > tolerance || diff < -tolerance) {                                              \
            printf("PE %i observed error with TEST_NVSHMEM_ADD_KERNEL(%s, %s)\n", mype, #OP,       \
                   #TYPE);                                                                        \
            printf("found = %f, expected = %f\n", (double)*remote, (double)expected * npes);      \
            error_d = 1;                                                                          \
        }                                                                                         \
    }
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_ADD_FLOAT, __half, half)
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_FETCH_ADD_FLOAT, __half, half)
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_ADD_FLOAT, float, float)
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_FETCH_ADD_FLOAT, float, float)
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_ADD_FLOAT, double, double)
TEST_NVSHMEM_ATOMIC_ADD_KERNEL(ATOMIC_FETCH_ADD_FLOAT, double, double)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_ADD_FLOATING(OP, TYPE, TYPENAME)                             \
    do {                                                                                 \
        TYPE value = 1.25f;                                                              \
        TYPE expected = 1.25f;                                                           \
        TYPE *remote = (TYPE *)nvshmem_calloc(1, sizeof(TYPE));                          \
        nvshmem_barrier_all();                                                           \
        if (use_cubin) {                                                                 \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                           \
        } else {                                                                         \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                                \
        cudaDeviceSynchronize();                                                         \
        value = -0.5f;                                                                   \
        expected = 0.75f;                                                                \
        nvshmem_barrier_all();                                                           \
        if (use_cubin) {                                                                 \
            TEST_NVSHMEM_ATOMIC_ADD_CUBIN(TYPENAME, TYPE, OP);                           \
        } else {                                                                         \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1>>>(remote, value, expected); \
        }                                                                                \
        cudaDeviceSynchronize();                                                         \
    } while (0)

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_ADD_FLOAT, __half, half);
    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_FETCH_ADD_FLOAT, __half, half);

    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_ADD_FLOAT, float, float);
    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_FETCH_ADD_FLOAT, float, float);

    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_ADD_FLOAT, double, double);
    TEST_NVSHMEM_ATOMIC_ADD_FLOATING(ATOMIC_FETCH_ADD_FLOAT, double, double);

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
