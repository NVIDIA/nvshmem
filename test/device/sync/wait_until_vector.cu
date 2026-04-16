/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__device__ int error_d;

#define NVSHMEM_WAIT_UNTIL_ALL_VECTOR_TEST_KERNEL(TYPE, TYPENAME)                             \
    __global__ void nvshmem_##TYPENAME##_wait_until_all_vector_test_kernel(                   \
        TYPE *ivars, int *status, TYPE *cmp_values) {                                         \
        const int my_pe = nvshmem_my_pe();                                                    \
        const int npes = nvshmem_n_pes();                                                     \
                                                                                              \
        int i = 0;                                                                            \
        int expected_sum = 0;                                                                 \
        int total_sum = 0;                                                                    \
                                                                                              \
        for (i = 0; i < npes; i++) {                                                          \
            nvshmemx_signal_op(&ivars[my_pe], (TYPE)my_pe, NVSHMEM_SIGNAL_SET, i);            \
            cmp_values[i] = i;                                                                \
        }                                                                                     \
                                                                                              \
        expected_sum = (npes - 1) * npes / 2;                                                 \
        nvshmem_##TYPENAME##_wait_until_all_vector(ivars, npes, status, NVSHMEM_CMP_EQ,       \
                                                   cmp_values);                               \
                                                                                              \
        for (i = 0; i < npes; i++) {                                                          \
            total_sum += ivars[i];                                                            \
        }                                                                                     \
                                                                                              \
        if (expected_sum != total_sum) {                                                      \
            printf("Incorrect total_sum = %d, expected sum = %d\n", total_sum, expected_sum); \
            error_d = 1;                                                                      \
        }                                                                                     \
    }

NVSHMEM_WAIT_UNTIL_ALL_VECTOR_TEST_KERNEL(uint64_t, uint64)

#define TEST_NVSHMEM_WAIT_UNTIL_ALL_VECTOR(TYPE, TYPENAME)                              \
    do {                                                                                \
        const int npes = nvshmem_n_pes();                                               \
        TYPE *ivars = (TYPE *)nvshmem_calloc(npes, sizeof(TYPE));                       \
        int *status;                                                                    \
        cudaMalloc((void **)&status, npes * sizeof(int));                               \
        cudaMemset(status, 0, npes * sizeof(int));                                      \
        TYPE *cmp_values;                                                               \
        cudaMalloc((void **)&cmp_values, npes * sizeof(TYPE));                          \
        nvshmem_barrier_all();                                                          \
        nvshmem_##TYPENAME##_wait_until_all_vector_test_kernel<<<1, 1>>>(ivars, status, \
                                                                         cmp_values);   \
        cudaDeviceSynchronize();                                                        \
                                                                                        \
    } while (0)

#define NVSHMEM_WAIT_UNTIL_ANY_VECTOR_TEST_KERNEL(TYPE, TYPENAME)                                  \
    __global__ void nvshmem_##TYPENAME##_wait_until_any_vector_test_kernel(                        \
        TYPE *ivars, int *status, TYPE *cmp_values) {                                              \
        const int my_pe = nvshmem_my_pe();                                                         \
        const int npes = nvshmem_n_pes();                                                          \
                                                                                                   \
        int i = 0;                                                                                 \
        int total_sum = 0;                                                                         \
                                                                                                   \
        for (i = 0; i < npes; i++) {                                                               \
            nvshmemx_signal_op(&ivars[my_pe], (TYPE)my_pe, NVSHMEM_SIGNAL_SET, i);                 \
            cmp_values[i] = i;                                                                     \
        }                                                                                          \
                                                                                                   \
        int expected_sum = (npes - 1) * npes / 2;                                                  \
                                                                                                   \
        int ncompleted = 0;                                                                        \
        while (ncompleted < npes) {                                                                \
            size_t ndone = nvshmem_##TYPENAME##_wait_until_any_vector(ivars, npes, status,         \
                                                                      NVSHMEM_CMP_EQ, cmp_values); \
            status[ndone] = 1;                                                                     \
            total_sum += ivars[ndone];                                                             \
            ncompleted++;                                                                          \
        }                                                                                          \
                                                                                                   \
        if (expected_sum != total_sum) {                                                           \
            printf("Incorrect total_sum = %d, expected_sum = %d\n", total_sum, expected_sum);      \
            error_d = 1;                                                                           \
        }                                                                                          \
    }

NVSHMEM_WAIT_UNTIL_ANY_VECTOR_TEST_KERNEL(uint64_t, uint64)

#define TEST_NVSHMEM_WAIT_UNTIL_ANY_VECTOR(TYPE, TYPENAME)                              \
    do {                                                                                \
        const int npes = nvshmem_n_pes();                                               \
        TYPE *ivars = (TYPE *)nvshmem_calloc(npes, sizeof(TYPE));                       \
        int *status;                                                                    \
        cudaMalloc((void **)&status, npes * sizeof(int));                               \
        cudaMemset(status, 0, npes * sizeof(int));                                      \
        TYPE *cmp_values;                                                               \
        cudaMalloc((void **)&cmp_values, npes * sizeof(TYPE));                          \
        nvshmem_barrier_all();                                                          \
        nvshmem_##TYPENAME##_wait_until_any_vector_test_kernel<<<1, 1>>>(ivars, status, \
                                                                         cmp_values);   \
        cudaDeviceSynchronize();                                                        \
                                                                                        \
    } while (0)

#define NVSHMEM_WAIT_UNTIL_SOME_VECTOR_TEST_KERNEL(TYPE, TYPENAME)                            \
    __global__ void nvshmem_##TYPENAME##_wait_until_some_vector_test_kernel(                  \
        TYPE *ivars, size_t *indices, int *status, TYPE *cmp_values) {                        \
        const int my_pe = nvshmem_my_pe();                                                    \
        const int npes = nvshmem_n_pes();                                                     \
        int i = 0;                                                                            \
        int expected_sum = 0;                                                                 \
        int total_sum = 0;                                                                    \
                                                                                              \
        for (i = 0; i < npes; i++) {                                                          \
            nvshmemx_signal_op(&ivars[my_pe], (TYPE)my_pe, NVSHMEM_SIGNAL_SET, i);            \
            cmp_values[i] = i;                                                                \
        }                                                                                     \
                                                                                              \
        expected_sum = (npes - 1) * npes / 2;                                                 \
                                                                                              \
        int ncompleted = 0;                                                                   \
        while (ncompleted < npes) {                                                           \
            size_t ndone = nvshmem_##TYPENAME##_wait_until_some_vector(                       \
                ivars, npes, indices, status, NVSHMEM_CMP_EQ, cmp_values);                    \
            for (size_t j = 0; j < ndone; j++) {                                              \
                total_sum += ivars[indices[j]];                                               \
                status[indices[j]] = 1;                                                       \
            }                                                                                 \
            ncompleted += ndone;                                                              \
        }                                                                                     \
                                                                                              \
        if (expected_sum != total_sum) {                                                      \
            printf("Incorrect total_sum = %d, expected_sum = %d\n", total_sum, expected_sum); \
            error_d = 1;                                                                      \
        }                                                                                     \
    }

NVSHMEM_WAIT_UNTIL_SOME_VECTOR_TEST_KERNEL(uint64_t, uint64)

#define TEST_NVSHMEM_WAIT_UNTIL_SOME_VECTOR(TYPE, TYPENAME)                                       \
    do {                                                                                          \
        const int npes = nvshmem_n_pes();                                                         \
        TYPE *ivars = (TYPE *)nvshmem_calloc(npes, sizeof(TYPE));                                 \
        int *status;                                                                              \
        cudaMalloc((void **)&status, npes * sizeof(int));                                         \
        cudaMemset(status, 0, npes * sizeof(int));                                                \
        TYPE *cmp_values;                                                                         \
        cudaMalloc((void **)&cmp_values, npes * sizeof(TYPE));                                    \
                                                                                                  \
        size_t *indices;                                                                          \
        cudaMalloc((void **)&indices, npes * sizeof(size_t));                                     \
        nvshmem_barrier_all();                                                                    \
        nvshmem_##TYPENAME##_wait_until_some_vector_test_kernel<<<1, 1>>>(ivars, indices, status, \
                                                                          cmp_values);            \
        cudaDeviceSynchronize();                                                                  \
                                                                                                  \
    } while (0)

int main(int argc, char **argv) {
    int ret_val = 0;
    init_wrapper(&argc, &argv);
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    int zero = 0;
    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    TEST_NVSHMEM_WAIT_UNTIL_ALL_VECTOR(uint64_t, uint64);
    TEST_NVSHMEM_WAIT_UNTIL_ANY_VECTOR(uint64_t, uint64);
    TEST_NVSHMEM_WAIT_UNTIL_SOME_VECTOR(uint64_t, uint64);

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();

    return ret_val;
}
