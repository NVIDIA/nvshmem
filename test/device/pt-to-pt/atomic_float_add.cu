/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "atomic_float_add.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>

#include "utils.h"

__device__ int error_d;

enum op {
    ATOMIC_ADD_FLOAT = 0,
    ATOMIC_FETCH_ADD_FLOAT,
};

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

/* Each PE adds value to remote. In the fetch case, we compare old against (expected * npes) for
   each PE. We have to use a range since we don't know in which order the PEs will be executed.
   At the end, we confirm that remote contains (expected * npes). */
#define TEST_NVSHMEM_ATOMIC_FLOAT_ADD_KERNEL(OP)                                                    \
    __global__ void test_nvshmem_float_##OP##_kernel(float *remote, float value, float expected) { \
        float old;                                                                                   \
        const int mype = nvshmem_my_pe();                                                            \
        const int npes = nvshmem_n_pes();                                                            \
        for (int i = 0; i < npes; i++) {                                                             \
            if (OP == ATOMIC_ADD_FLOAT) {                                                            \
                nvshmemx_float_atomic_add(remote, value, i);                                         \
            } else if (OP == ATOMIC_FETCH_ADD_FLOAT) {                                              \
                old = nvshmemx_float_atomic_fetch_add(remote, value, i);                             \
                if ((value > 0.0f) && (old >= (expected * npes))) {                                  \
                    printf("PE %i error inconsistent value of old (%s)\n", mype, #OP);               \
                    printf("found = " NVSHPRI_float ", expected < " NVSHPRI_float "\n",              \
                           old, expected * npes);                                                    \
                    error_d++;                                                                        \
                } else if ((value <= 0.0f) && (old < (expected * npes))) {                           \
                    printf("PE %i error inconsistent value of old (%s)\n", mype, #OP);               \
                    printf("found = " NVSHPRI_float ", expected >= " NVSHPRI_float "\n",             \
                           old, expected * npes);                                                    \
                    error_d++;                                                                        \
                }                                                                                    \
            } else {                                                                                  \
                printf("Invalid operation (%d)\n", OP);                                              \
                assert(0);                                                                            \
            }                                                                                        \
        }                                                                                            \
        nvshmem_barrier_all();                                                                       \
        if (fabsf(*remote - expected * npes) > 0.01f) {                                              \
            printf("PE %i observed error with TEST_NVSHMEM_ATOMIC_FLOAT_ADD_KERNEL(%s)\n", mype,    \
                   #OP);                                                                              \
            printf("found = " NVSHPRI_float ", expected = " NVSHPRI_float "\n", *remote,             \
                   expected * npes);                                                                  \
            error_d = 1;                                                                              \
        }                                                                                             \
    }

TEST_NVSHMEM_ATOMIC_FLOAT_ADD_KERNEL(ATOMIC_ADD_FLOAT)
TEST_NVSHMEM_ATOMIC_FLOAT_ADD_KERNEL(ATOMIC_FETCH_ADD_FLOAT)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_FLOAT_ADD_CUBIN(OP)                                                   \
    void *args_float_add_##OP[] = {(void *)&remote, (void *)&value, (void *)&expected};           \
    CUfunction test_float_add_##OP##_cubin;                                                        \
    init_test_case_kernel(&test_float_add_##OP##_cubin,                                            \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_float_##OP##_kernel));              \
    CU_CHECK(cuLaunchKernel(test_float_add_##OP##_cubin, 1, 1, 1, 1, 1, 1, 0, 0,                  \
                            args_float_add_##OP, NULL));

#define TEST_NVSHMEM_ATOMIC_FLOAT_ADD(OP)                                               \
    do {                                                                                 \
        float value = 5128.0f;                                                           \
        float expected = 5128.0f;                                                        \
        float *remote = (float *)nvshmem_calloc(1, sizeof(float));                       \
        nvshmem_barrier_all();                                                           \
        if (use_cubin) {                                                                 \
            TEST_NVSHMEM_ATOMIC_FLOAT_ADD_CUBIN(OP);                                     \
        } else {                                                                         \
            test_nvshmem_float_##OP##_kernel<<<1, 1>>>(remote, value, expected);         \
        }                                                                                \
        cudaDeviceSynchronize();                                                         \
        value = 128.0f;                                                                  \
        expected = 5256.0f;                                                              \
        nvshmem_barrier_all();                                                           \
        if (use_cubin) {                                                                 \
            TEST_NVSHMEM_ATOMIC_FLOAT_ADD_CUBIN(OP);                                     \
        } else {                                                                         \
            test_nvshmem_float_##OP##_kernel<<<1, 1>>>(remote, value, expected);         \
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

    TEST_NVSHMEM_ATOMIC_FLOAT_ADD(ATOMIC_ADD_FLOAT);
    TEST_NVSHMEM_ATOMIC_FLOAT_ADD(ATOMIC_FETCH_ADD_FLOAT);

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
