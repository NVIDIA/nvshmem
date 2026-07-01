/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_ALLTOALL_COMMON_H
#define NVSHMEMTEST_ALLTOALL_COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include "device_host/nvshmem_common.cuh"
#include "utils.h"

__device__ unsigned long long int errs_d;

#define DECL_INIT_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                         \
    __device__ void init_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                               size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_INIT_ALLTOALL_DATA)
#undef DECL_INIT_ALLTOALL_DATA

#define DECL_VALIDATE_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)           \
    __device__ void validate_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team, \
                                                                   TYPE *dest, size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_VALIDATE_ALLTOALL_DATA)
#undef DECL_VALIDATE_ALLTOALL_DATA

#define DECL_RESET_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                       \
    __device__ void reset_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_RESET_ALLTOALL_DATA)
#undef DECL_RESET_ALLTOALL_DATA

#define DECL_INIT_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                        \
    __global__ void init_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                           size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_INIT_ALLTOALL_DATA_KERNEL)
#undef DECL_INIT_ALLTOALL_DATA_KERNEL

#define DECL_VALIDATE_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void validate_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                               size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_VALIDATE_ALLTOALL_DATA_KERNEL)
#undef DECL_VALIDATE_ALLTOALL_DATA_KERNEL

#define DECL_RESET_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void reset_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                            size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_RESET_ALLTOALL_DATA_KERNEL)
#undef DECL_RESET_ALLTOALL_DATA_KERNEL

#define INIT_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                              \
    __device__ void init_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                               size_t nelems) {                   \
        int n_pes = nvshmem_team_n_pes(team);                                                     \
        int my_pe = nvshmem_team_my_pe(team);                                                     \
                                                                                                  \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
                                                                                                  \
        for (size_t i = 0; i < n_pes; i++) {                                                      \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                  \
                source[i * nelems + j] = assign<TYPE>(my_pe * n_pes * nelems + i * nelems + j);   \
            }                                                                                     \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(INIT_ALLTOALL_DATA)
#undef INIT_ALLTOALL_DATA

#define INIT_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                             \
    __global__ void init_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                           size_t nelems) {                   \
        init_##TYPENAME##_alltoall_data_block(team, source, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(INIT_ALLTOALL_DATA_KERNEL)
#undef INIT_ALLTOALL_DATA_KERNEL

#define VALIDATE_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                        \
    __device__ void validate_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team,         \
                                                                   TYPE *dest, size_t nelems) { \
        int n_pes = nvshmem_team_n_pes(team);                                                   \
        int my_pe = nvshmem_team_my_pe(team);                                                   \
        int myIdx = nvshmtest_thread_id_in_##SC();                                              \
        int groupSize = nvshmtest_##SC##_size();                                                \
                                                                                                \
        for (size_t i = 0; i < n_pes; i++) {                                                    \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                \
                TYPE expected = assign<TYPE>(i * n_pes * nelems + my_pe * nelems + j);          \
                if (dest[i * nelems + j] != expected) {                                         \
                    printf(NVSHMEMTEST_ERRSTR_FORMAT_1(TYPENAME, SC),                           \
                           to_printable(dest[i * nelems + j]), to_printable(expected),          \
                           i * nelems + j, nelems, team);                                       \
                    atomicAdd(&errs_d, 1);                                                      \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        nvshmtest_##SC##_sync();                                                                \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(VALIDATE_ALLTOALL_DATA)
#undef VALIDATE_ALLTOALL_DATA

#define VALIDATE_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void validate_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                               size_t nelems) {                 \
        validate_##TYPENAME##_alltoall_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(VALIDATE_ALLTOALL_DATA_KERNEL)
#undef VALIDATE_ALLTOALL_DATA_KERNEL

#define RESET_ALLTOALL_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                            \
    __device__ void reset_##TYPENAME##_alltoall_data##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                size_t nelems) {                 \
        int n_pes = nvshmem_team_n_pes(team);                                                    \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        for (size_t i = 0; i < n_pes; i++) {                                                     \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                 \
                dest[i * nelems + j] = assign<TYPE>(0);                                          \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(RESET_ALLTOALL_DATA)
#undef RESET_ALLTOALL_DATA

#define RESET_ALLTOALL_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void reset_##TYPENAME##_alltoall_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                            size_t nelems) {                 \
        reset_##TYPENAME##_alltoall_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(RESET_ALLTOALL_DATA_KERNEL)
#undef RESET_ALLTOALL_DATA_KERNEL

#endif /* NVSHMEMTEST_ALLTOALL_COMMON_H */
