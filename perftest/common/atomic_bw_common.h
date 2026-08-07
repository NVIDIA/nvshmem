/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _ATOMIC_BW_COMMON_H_
#define _ATOMIC_BW_COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

#define MAX_ITERS 10
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 4
#define MAX_MSG_SIZE 64 * 1024
#define ATOMIC_BW_TARGET_STRIDE 2
/* CFT assigns two handle-barrier slots to each warp. Keep every benchmark warp
 * within the fixed handle-barrier region so the measurement does not mix CFT
 * atomics with fallback atomics.
 */
#define CFT_ATOMIC_HANDLE_SLOTS_PER_WARP 2
#define MAX_CFT_ATOMIC_THREADS \
    ((NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS / CFT_ATOMIC_HANDLE_SLOTS_PER_WARP) * NVSHMEMI_WARP_SIZE)

#define DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                \
    void test_atomic_##AMO##_bw_cubin(int num_blocks, int num_tpb, void **arglist,       \
                                      size_t dynamic_smem_size) {                        \
        CUfunction test_cubin;                                                           \
        init_test_case_kernel(&test_cubin, NVSHMEMI_TEST_STRINGIFY(atomic_##AMO##_bw));  \
        if (dynamic_smem_size > 48 * 1024) {                                             \
            CU_CHECK(cuFuncSetAttribute(test_cubin,                                      \
                                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, \
                                        (int)dynamic_smem_size));                        \
        }                                                                                \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, num_blocks, 1, 1, num_tpb, 1, 1,  \
                                           dynamic_smem_size, 0, arglist));              \
    }

#define DEFINE_ATOMIC_BW_FN_NO_ARG(AMO)                                                            \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter, size_t dynamic_smem_size) {                \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, peer);                                         \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) {                                                                       \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, peer);                                         \
            }                                                                                      \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1);                                                 \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1);                                                     \
        }                                                                                          \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

#define DEFINE_ATOMIC_BW_FN_ONE_ARG(AMO, SET_EXPR)                                                 \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter, size_t dynamic_smem_size) {                \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, SET_EXPR, peer);                               \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) {                                                                       \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, SET_EXPR, peer);                               \
            }                                                                                      \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1);                                                 \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1);                                                     \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

#define DEFINE_ATOMIC_BW_FN_TWO_ARG(AMO, COMPARE_EXPR, SET_EXPR)                                   \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter, size_t dynamic_smem_size) {                \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, COMPARE_EXPR, SET_EXPR, peer);                 \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) {                                                                       \
                uint64_t *target = data_d + idx * ATOMIC_BW_TARGET_STRIDE;                         \
                nvshmem_uint64_atomic_##AMO(target, COMPARE_EXPR, SET_EXPR, peer);                 \
            }                                                                                      \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1);                                                 \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1);                                                     \
        }                                                                                          \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

#define CALL_ATOMIC_BW_KERNEL(AMO, BLOCKS, THREADS, DATA, COUNTER, SIZE, PE, ITER, ARGS)         \
    if (use_cubin) {                                                                             \
        test_atomic_##AMO##_bw_cubin(BLOCKS, THREADS, ARGS, dynamic_smem_size);                  \
    } else {                                                                                     \
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(atomic_##AMO##_bw, dynamic_smem_size);                 \
        atomic_##AMO##_bw<<<BLOCKS, THREADS, dynamic_smem_size>>>(DATA, COUNTER, SIZE, PE, ITER, \
                                                                  dynamic_smem_size);            \
    }

#endif /* _ATOMIC_BW_COMMON_H_ */
