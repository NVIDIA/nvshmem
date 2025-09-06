/*
 * Copyright (c) 2021, NVIDIA CORPORATION   All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto   Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
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

#define DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                     \
    void test_atomic_##AMO##_bw_cubin(int num_blocks, int num_tpb, void **arglist) {          \
        CUfunction test_cubin;                                                                \
        init_test_case_kernel(&test_cubin, NVSHMEMI_TEST_STRINGIFY(atomic_##AMO##_bw));       \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, num_blocks, 1, 1, num_tpb, 1, 1, 0, 0, \
                                           arglist));                                         \
    }

#define DEFINE_ATOMIC_BW_FN_NO_ARG(AMO)                                                            \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, peer);                                   \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) nvshmem_uint64_atomic_##AMO(data_d + idx, peer);                        \
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
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
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
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
    }

#define DEFINE_ATOMIC_BW_FN_ONE_ARG(AMO, SET_EXPR)                                                 \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, SET_EXPR, peer);                         \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) nvshmem_uint64_atomic_##AMO(data_d + idx, SET_EXPR, peer);              \
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
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
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
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
    }

#define DEFINE_ATOMIC_BW_FN_TWO_ARG(AMO, COMPARE_EXPR, SET_EXPR)                                   \
    DEFINE_ATOMIC_BW_CALL_KERNEL(AMO)                                                              \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, COMPARE_EXPR, SET_EXPR, peer);           \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) {                                                                       \
                nvshmem_uint64_atomic_##AMO(data_d + idx, COMPARE_EXPR, SET_EXPR, peer);           \
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
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
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
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
    }

#define CALL_ATOMIC_BW_KERNEL(AMO, BLOCKS, THREADS, DATA, COUNTER, SIZE, PE, ITER, ARGS) \
    if (use_cubin) {                                                                     \
        test_atomic_##AMO##_bw_cubin(BLOCKS, THREADS, ARGS);                             \
    } else {                                                                             \
        atomic_##AMO##_bw<<<BLOCKS, THREADS>>>(DATA, COUNTER, SIZE, PE, ITER);           \
    }

#endif /* _ATOMIC_BW_COMMON_H_ */
