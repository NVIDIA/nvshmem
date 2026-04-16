/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_BARRIER_COMMON_H
#define NVSHMEMTEST_BARRIER_COMMON_H

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "utils.h"

__device__ unsigned long long int errs_d;

#define NVSHMEMI_REPT_FOR_SCOPES2(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(thread, , )                    \
    NVSHMEMI_FN_TEMPLATE(warp, _warp, x)                \
    NVSHMEMI_FN_TEMPLATE(block, _block, x)

#define NVSHMEMI_REPT_FOR_SCOPE(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(thread)                      \
    NVSHMEMI_FN_TEMPLATE(warp)                        \
    NVSHMEMI_FN_TEMPLATE(block)

#define DECL_INIT_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX) \
    __device__ void init_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf);
NVSHMEMI_REPT_FOR_SCOPES2(DECL_INIT_BARRIER_DATA)
#undef DECL_INIT_BARRIER_DATA

__global__ void init_barrier_data_kernel(nvshmem_team_t team, int *buf);

#define DECL_VALIDATE_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX) \
    __device__ void validate_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf);
NVSHMEMI_REPT_FOR_SCOPES2(DECL_VALIDATE_BARRIER_DATA)
#undef DECL_VALIDATE_BARRIER_DATA

__global__ void validate_barrier_data_kernel(nvshmem_team_t team, int *buf);

#define DECL_RESET_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX) \
    __device__ void reset_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf);
NVSHMEMI_REPT_FOR_SCOPES2(DECL_RESET_BARRIER_DATA)
#undef DECL_RESET_BARRIER_DATA

__global__ void reset_barrier_data_kernel(nvshmem_team_t team, int *buf);

#define INIT_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX)                                \
    __device__ void init_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf) {  \
        int my_pe = nvshmem_team_my_pe(team);                                      \
        int n_pes = nvshmem_team_n_pes(team);                                      \
        int myIdx = nvshmtest_thread_id_in_##SC();                                 \
        int groupSize = nvshmtest_##SC##_size();                                   \
        for (size_t i = myIdx; i < n_pes; i += groupSize) {                        \
            nvshmem_int_p(&buf[my_pe], my_pe,                                      \
                          nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD)); \
        }                                                                          \
        nvshmtest_##SC##_sync();                                                   \
    }
NVSHMEMI_REPT_FOR_SCOPES2(INIT_BARRIER_DATA)
#undef INIT_BARRIER_DATA

__global__ void init_barrier_data_kernel(nvshmem_team_t team, int *buf) {
    init_barrier_data_block(team, buf);
}

#define VALIDATE_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX)                                           \
    __device__ void validate_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf) {             \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
        int n_pes = nvshmem_team_n_pes(team);                                                     \
        for (int pe = myIdx; pe < n_pes; pe += groupSize) {                                       \
            if (buf[pe] != pe) {                                                                  \
                printf("error: buf_idx = %d, found = %d, expected = %d, n_pes = %d, team = %d\n", \
                       pe, buf[pe], pe, n_pes, team);                                             \
                atomicAdd(&errs_d, 1);                                                            \
            }                                                                                     \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }

NVSHMEMI_REPT_FOR_SCOPES2(VALIDATE_BARRIER_DATA)
#undef VALIDATE_BARRIER_DATA

__global__ void validate_barrier_data_kernel(nvshmem_team_t team, int *buf) {
    validate_barrier_data_block(team, buf);
}

#define RESET_BARRIER_DATA(SC, SC_SUFFIX, SC_PREFIX)                               \
    __device__ void reset_barrier_data##SC_SUFFIX(nvshmem_team_t team, int *buf) { \
        int myIdx = nvshmtest_thread_id_in_##SC();                                 \
        int groupSize = nvshmtest_##SC##_size();                                   \
        int n_pes = nvshmem_team_n_pes(team);                                      \
        for (size_t i = myIdx; i < n_pes; i += groupSize) {                        \
            buf[i] = 0;                                                            \
        }                                                                          \
        nvshmtest_##SC##_sync();                                                   \
    }

NVSHMEMI_REPT_FOR_SCOPES2(RESET_BARRIER_DATA)
#undef RESET_BARRIER_DATA

__global__ void reset_barrier_data_kernel(nvshmem_team_t team, int *buf) {
    reset_barrier_data_block(team, buf);
}

#endif /* NVSHMEMTEST_BARRIER_COMMON_H */
