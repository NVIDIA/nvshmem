/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_REDUCESCATTER_COMMON_CPU_H
#define NVSHMEMTEST_REDUCESCATTER_COMMON_CPU_H

#include <cuda_runtime.h>
#include "stdio.h"
#include "device_host/nvshmem_common.cuh"
#include "utils.h"
#include "reduce_common.h"

#define DECL_INIT_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __device__ void init_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *source, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_INIT_REDUCESCATTER_DATA)
#undef DECL_INIT_REDUCESCATTER_DATA

#define DECL_INIT_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)         \
    __global__ void init_##TYPENAME##_##OP##_reducescatter_data_kernel( \
        nvshmem_team_t team, TYPE *source, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_INIT_REDUCESCATTER_DATA_KERNEL)
#undef DECL_INIT_REDUCESCATTER_DATA_KERNEL

#define DECL_VALIDATE_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __device__ void validate_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_VALIDATE_REDUCESCATTER_DATA)
#undef DECL_VALIDATE_REDUCESCATTER_DATA

#define DECL_VALIDATE_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)         \
    __global__ void validate_##TYPENAME##_##OP##_reducescatter_data_kernel( \
        nvshmem_team_t team, TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_VALIDATE_REDUCESCATTER_DATA_KERNEL)
#undef DECL_VALIDATE_REDUCESCATTER_DATA_KERNEL

#define DECL_RESET_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __device__ void reset_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_RESET_REDUCESCATTER_DATA)
#undef DECL_RESET_REDUCESCATTER_DATA

#define DECL_RESET_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)         \
    __global__ void reset_##TYPENAME##_##OP##_reducescatter_data_kernel( \
        nvshmem_team_t team, TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_RESET_REDUCESCATTER_DATA_KERNEL)
#undef DECL_RESET_REDUCESCATTER_DATA_KERNEL

#define INIT_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)           \
    __device__ void init_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(             \
        nvshmem_team_t team, TYPE *source, size_t nelems) {                             \
        int my_pe = nvshmem_team_my_pe(team);                                           \
        int n_pes = nvshmem_team_n_pes(team);                                           \
        int myIdx = nvshmtest_thread_id_in_##SC();                                      \
        int groupSize = nvshmtest_##SC##_size();                                        \
        for (size_t i = myIdx; i < nelems * nvshmem_team_n_pes(team); i += groupSize) { \
            if (nvshmtest_are_strings_same(#OP, "prod"))                                \
                source[i] = assign<TYPE>((my_pe == n_pes - 1) ? n_pes : 1);             \
            else if (nvshmtest_are_strings_same(#TYPENAME, "half"))                     \
                source[i] = assign<TYPE>((my_pe >= 2048) ? 0 : 1);                      \
            else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                 \
                source[i] = assign<TYPE>((my_pe >= 256) ? 0 : 1);                       \
            else                                                                        \
                source[i] = assign<TYPE>(my_pe + i);                                    \
        }                                                                               \
        nvshmtest_##SC##_sync();                                                        \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(INIT_REDUCESCATTER_DATA)
#undef INIT_REDUCE_DATA

#define INIT_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)                       \
    __global__ void init_##TYPENAME##_##OP##_reducescatter_data_kernel(          \
        nvshmem_team_t team, TYPE *source, size_t nelems) {                      \
        init_##TYPENAME##_##OP##_reducescatter_data_block(team, source, nelems); \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(INIT_REDUCESCATTER_DATA_KERNEL)
#undef INIT_REDUCE_DATA_KERNEL

#define VALIDATE_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                \
    __device__ void validate_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(                  \
        nvshmem_team_t team, TYPE *dest, size_t nelems) {                                        \
        int n_pes = nvshmem_team_n_pes(team);                                                    \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
                                                                                                 \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                     \
            TYPE expected;                                                                       \
            if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                                 \
                nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                               \
                expected = assign<TYPE>(1);                                                      \
            else                                                                                 \
                expected = assign<TYPE>(i + nelems * my_pe);                                     \
            if (nvshmtest_are_strings_same(#OP, "prod"))                                         \
                expected = assign<TYPE>(n_pes);                                                  \
            else {                                                                               \
                for (size_t j = 1; j < n_pes; j++) {                                             \
                    if (nvshmtest_are_strings_same(#TYPENAME, "half"))                           \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 2048) ? 0 : 1)); \
                    else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                  \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 256) ? 0 : 1));  \
                    else                                                                         \
                        expected = reduce_##OP(expected, assign<TYPE>(j + nelems * my_pe + i));  \
                }                                                                                \
            }                                                                                    \
            if (dest[i] != expected) {                                                           \
                printf(NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC),                            \
                       to_printable(dest[i]), to_printable(expected), i, nelems, team);          \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(VALIDATE_REDUCESCATTER_DATA)
#undef VALIDATE_REDUCESCATTER_DATA

#define VALIDATE_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)               \
    __global__ void validate_##TYPENAME##_##OP##_reducescatter_data_kernel(  \
        nvshmem_team_t team, TYPE *dest, size_t nelems) {                    \
        validate_##TYPENAME##_##OP##_reducescatter_data(team, dest, nelems); \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(VALIDATE_REDUCESCATTER_DATA_KERNEL)
#undef VALIDATE_REDUCESCATTER_DATA_KERNEL

#define RESET_REDUCESCATTER_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __device__ void reset_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(   \
        nvshmem_team_t team, TYPE *dest, size_t nelems) {                      \
        int myIdx = nvshmtest_thread_id_in_##SC();                             \
        int groupSize = nvshmtest_##SC##_size();                               \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                   \
            dest[i] = assign<TYPE>(0);                                         \
        }                                                                      \
        nvshmtest_##SC##_sync();                                               \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(RESET_REDUCESCATTER_DATA)
#undef RESET_REDUCESCATTER_DATA

#define RESET_REDUCESCATTER_DATA_KERNEL(TYPENAME, TYPE, OP)               \
    __global__ void reset_##TYPENAME##_##OP##_reducescatter_data_kernel(  \
        nvshmem_team_t team, TYPE *dest, size_t nelems) {                 \
        reset_##TYPENAME##_##OP##_reducescatter_data(team, dest, nelems); \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(RESET_REDUCESCATTER_DATA_KERNEL)
#undef RESET_REDUCESCATTER_DATA_KERNEL

#define DO_RDST_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                         \
    void *args_##TYPENAME##_##SC_SUFFIX[] = {(void *)&team, (void *)&dest, (void *)&source,      \
                                             (void *)&nelems};                                   \
    CUfunction test_##TYPENAME##_rdst##SC_SUFFIX_cubin;                                          \
    init_test_case_kernel(                                                                       \
        &test_##TYPENAME##_rdst##SC_SUFFIX_cubin,                                                \
        NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_##OP##_reducescatter_kernel##SC_SUFFIX));      \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_rdst##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, 1, \
                            0, cstrm, args_##TYPENAME##_##SC_SUFFIX, NULL));

#define DECL_TYPENAME_OP_REDUCESCATTER(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __global__ void test_##TYPENAME##_##OP##_reducescatter_kernel##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems);

#define DEFN_TYPENAME_OP_REDUCESCATTER(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)              \
    __global__ void test_##TYPENAME##_##OP##_reducescatter_kernel##SC_SUFFIX(                     \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems) {                           \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
        init_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(team, source, nelems);             \
                                                                                                  \
        for (int j = 0; j < 1 /*MAX_ITER*/; j++) {                                                \
            reset_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(team, dest, nelems);          \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                        \
            nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reducescatter##SC_SUFFIX(team, dest, source, \
                                                                              nelems);            \
            validate_##TYPENAME##_##OP##_reducescatter_data##SC_SUFFIX(team, dest, nelems);       \
        }                                                                                         \
    }

#define DO_REDUCESCATTER_DEVICE_TEST(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)              \
    if (use_cubin) {                                                                            \
        init_cumodule(CUMODULE_NAME);                                                           \
        DO_RDST_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP);                       \
    } else {                                                                                    \
        test_##TYPENAME##_##OP##_reducescatter_kernel##SC_SUFFIX<<<1, num_threads, 0, cstrm>>>( \
            team, (TYPE *)dest, (TYPE *)source, nelems);                                        \
    }                                                                                           \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                                     \
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(cstrm));

#endif /* NVSHMEMTEST_REDUCESCATTER_COMMON_CPU_H */
