/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_BROADCAST_COMMON_H
#define NVSHMEMTEST_BROADCAST_COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include "stdio.h"
#include "device_host/nvshmem_common.cuh"
#include "utils.h"
#include "cuda.h"

__device__ unsigned long long int errs_d;

#define DECL_INIT_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                         \
    __device__ void init_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                                size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_INIT_BROADCAST_DATA)
#undef DECL_INIT_BROADCAST_DATA

#define DECL_INIT_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                        \
    __global__ void init_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                            size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_INIT_BROADCAST_DATA_KERNEL)
#undef DECL_INIT_BROADCAST_DATA_KERNEL

#define DECL_VALIDATE_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)           \
    __device__ void validate_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team, \
                                                                    TYPE *dest, size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_VALIDATE_BROADCAST_DATA)
#undef DECL_VALIDATE_BROADCAST_DATA

#define DECL_VALIDATE_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void validate_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                                size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_VALIDATE_BROADCAST_DATA_KERNEL)
#undef DECL_VALIDATE_BROADCAST_DATA_KERNEL

#define DECL_RESET_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                       \
    __device__ void reset_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                 size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_RESET_BROADCAST_DATA)
#undef DECL_RESET_BROADCAST_DATA

#define DECL_RESET_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void reset_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                             size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_RESET_BROADCAST_DATA_KERNEL)
#undef DECL_RESET_BROADCAST_DATA_KERNEL

#define INIT_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                              \
    __device__ void init_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                                size_t nelems) {                   \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                 \
        int groupSize = nvshmtest_##SC##_size();                                                   \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                       \
            source[i] = assign<TYPE>(i);                                                           \
        }                                                                                          \
        nvshmtest_##SC##_sync();                                                                   \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(INIT_BROADCAST_DATA)
#undef INIT_BROADCAST_DATA

#define INIT_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                             \
    __global__ void init_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                            size_t nelems) {                   \
        init_##TYPENAME##_broadcast_data_block(team, source, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(INIT_BROADCAST_DATA_KERNEL)
#undef INIT_BROADCAST_DATA_KERNEL

#define VALIDATE_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                        \
    __device__ void validate_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team,         \
                                                                    TYPE *dest, size_t nelems) { \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                     \
            TYPE expected = assign<TYPE>(i);                                                     \
            if (dest[i] != expected) {                                                           \
                print_err<TYPE>(dest[i], expected, i, nelems, team,                              \
                                NVSHMEMTEST_ERRSTR_FORMAT_1(TYPENAME, SC));                      \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(VALIDATE_BROADCAST_DATA)
#undef VALIDATE_BROADCAST_DATA

#define VALIDATE_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void validate_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                                size_t nelems) {                 \
        validate_##TYPENAME##_broadcast_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(VALIDATE_BROADCAST_DATA_KERNEL)
#undef VALIDATE_BROADCAST_DATA_KERNEL

#define RESET_BROADCAST_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                            \
    __device__ void reset_##TYPENAME##_broadcast_data##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                 size_t nelems) {                 \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                      \
            dest[i] = assign<TYPE>(0);                                                            \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(RESET_BROADCAST_DATA)
#undef RESET_BROADCAST_DATA

#define RESET_BROADCAST_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void reset_##TYPENAME##_broadcast_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                             size_t nelems) {                 \
        reset_##TYPENAME##_broadcast_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(RESET_BROADCAST_DATA_KERNEL)
#undef RESET_BROADCAST_DATA_KERNEL

#endif /* NVSHMEMTEST_BROADCAST_COMMON_H */
