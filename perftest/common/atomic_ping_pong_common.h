/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#ifndef _ATOMIC_PING_PONG_COMMON_H_
#define _ATOMIC_PING_PONG_COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define DEFINE_ATOMIC_LATENCY_PING_PONG_CALL_KERNEL(AMO, TYPE_NAME)                            \
    void test_ping_pong_##TYPE_NAME##_##AMO##_cubin(cudaStream_t stream, void **arglist) {     \
        CUfunction test_cubin;                                                                 \
        init_test_case_kernel(&test_cubin,                                                     \
                              NVSHMEMI_TEST_STRINGIFY(ping_pong_##TYPE_NAME##_##AMO));         \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, 1, 1, 1, 1, 1, 1, 0, stream, arglist)); \
    }

#define DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR)               \
    DEFINE_ATOMIC_LATENCY_PING_PONG_CALL_KERNEL(AMO, TYPE_NAME)                                \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter) {            \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                              \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                              \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
    DEFINE_ATOMIC_LATENCY_PING_PONG_CALL_KERNEL(AMO, TYPE_NAME)                                \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, TYPE value,  \
                                                  TYPE cmp) {                                  \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                    \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                    \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
    DEFINE_ATOMIC_LATENCY_PING_PONG_CALL_KERNEL(AMO, TYPE_NAME)                                \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, TYPE value,  \
                                                  TYPE cmp) {                                  \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);            \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);      \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);      \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);            \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat, atomic_op) \
    do {                                                                                     \
        init_wrapper(&c, &v);                                                                \
                                                                                             \
        if (use_cubin) {                                                                     \
            init_cumodule(CUMODULE_NAME);                                                    \
        }                                                                                    \
                                                                                             \
        mype = nvshmem_my_pe();                                                              \
        npes = nvshmem_n_pes();                                                              \
                                                                                             \
        if (npes != 2) {                                                                     \
            fprintf(stderr, "This test requires exactly two processes \n");                  \
            finalize_wrapper();                                                              \
            exit(-1);                                                                        \
        }                                                                                    \
                                                                                             \
        alloc_tables(&h_tables, 2, 1);                                                       \
        h_size_arr = (uint64_t *)h_tables[0];                                                \
        h_lat = (double *)h_tables[1];                                                       \
                                                                                             \
        if (use_mmap) {                                                                      \
            flag_d = allocate_mmap_buffer(sizeof(uint64_t), mem_handle_type, use_egm);       \
        } else {                                                                             \
            flag_d = nvshmem_malloc(sizeof(uint64_t));                                       \
        }                                                                                    \
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));                                 \
                                                                                             \
        CUDA_CHECK(cudaStreamCreate(&stream));                                               \
                                                                                             \
        nvshmem_barrier_all();                                                               \
                                                                                             \
        CUDA_CHECK(cudaDeviceSynchronize());                                                 \
                                                                                             \
        if (mype == 0) {                                                                     \
            printf("Note: This test measures full round-trip latency\n");                    \
        }                                                                                    \
    } while (0)

#define LAUNCH_KERNEL(TYPE_NAME, AMO, ARGLIST, STREAM)                                         \
    if (use_cubin) {                                                                           \
        test_ping_pong_##TYPE_NAME##_##AMO##_cubin(STREAM, ARGLIST);                           \
    } else {                                                                                   \
        status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO, 1, 1, \
                                            ARGLIST, 0, STREAM);                               \
        if (status != NVSHMEMX_SUCCESS) {                                                      \
            fprintf(stderr, "shmemx_collective_launch failed %d  \n", status);                 \
            exit(-1);                                                                          \
        }                                                                                      \
    }

#define RUN_TEST_WITHOUT_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip, h_lat, h_size_arr, \
                             flag_init)                                                         \
    do {                                                                                        \
        int size = sizeof(TYPE);                                                                \
                                                                                                \
        int status = 0;                                                                         \
        h_size_arr[0] = size;                                                                   \
        void *args_1[] = {&flag_d, &mype, &skip};                                               \
        void *args_2[] = {&flag_d, &mype, &iter};                                               \
                                                                                                \
        float milliseconds;                                                                     \
        cudaEvent_t start, stop;                                                                \
        cudaEventCreate(&start);                                                                \
        cudaEventCreate(&stop);                                                                 \
        TYPE flag_init_var = flag_init;                                                         \
                                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                    \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));   \
        nvshmem_barrier_all();                                                                  \
                                                                                                \
        cudaEventRecord(start, stream);                                                         \
        LAUNCH_KERNEL(TYPE_NAME, AMO, args_1, stream);                                          \
        cudaEventRecord(stop, stream);                                                          \
                                                                                                \
        cudaStreamSynchronize(stream);                                                          \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));   \
        cudaEventRecord(start, stream);                                                         \
        LAUNCH_KERNEL(TYPE_NAME, AMO, args_2, stream);                                          \
        cudaEventRecord(stop, stream);                                                          \
        CUDA_CHECK(cudaStreamSynchronize(stream));                                              \
        /* give latency in us */                                                                \
        cudaEventElapsedTime(&milliseconds, start, stop);                                       \
        h_lat[0] = (milliseconds * 1000) / iter;                                                \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
                                                                                                \
        if (mype == 0) {                                                                        \
            print_table_v1("shmem_at_" #TYPE "_" #AMO "_ping_lat", "None", "size (Bytes)",      \
                           "latency", "us", '-', h_size_arr, h_lat, 1);                         \
        }                                                                                       \
                                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                    \
                                                                                                \
    } while (0)

#define RUN_TEST_WITH_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip, h_lat, h_size_arr, val, \
                          cmp, flag_init)                                                         \
    do {                                                                                          \
        int size = sizeof(TYPE);                                                                  \
        TYPE compare, value, flag_init_var;                                                       \
                                                                                                  \
        int status = 0;                                                                           \
        h_size_arr[0] = size;                                                                     \
        void *args_1[] = {&flag_d, &mype, &skip, &value, &compare};                               \
        void *args_2[] = {&flag_d, &mype, &iter, &value, &compare};                               \
                                                                                                  \
        float milliseconds;                                                                       \
        cudaEvent_t start, stop;                                                                  \
        cudaEventCreate(&start);                                                                  \
        cudaEventCreate(&stop);                                                                   \
                                                                                                  \
        compare = cmp;                                                                            \
        value = val;                                                                              \
        flag_init_var = flag_init;                                                                \
                                                                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                                      \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));     \
        nvshmem_barrier_all();                                                                    \
                                                                                                  \
        LAUNCH_KERNEL(TYPE_NAME, AMO, args_1, stream);                                            \
                                                                                                  \
        cudaStreamSynchronize(stream);                                                            \
                                                                                                  \
        nvshmem_barrier_all();                                                                    \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));     \
        cudaEventRecord(start, stream);                                                           \
                                                                                                  \
        LAUNCH_KERNEL(TYPE_NAME, AMO, args_2, stream);                                            \
                                                                                                  \
        cudaEventRecord(stop, stream);                                                            \
        cudaStreamSynchronize(stream);                                                            \
        /* give latency in us */                                                                  \
        cudaEventElapsedTime(&milliseconds, start, stop);                                         \
        h_lat[0] = (milliseconds * 1000) / iter;                                                  \
                                                                                                  \
        nvshmem_barrier_all();                                                                    \
                                                                                                  \
        if (mype == 0) {                                                                          \
            print_table_v1("shmem_at_" #TYPE "_" #AMO "_lat", "None", "size (Bytes)", "latency",  \
                           "us", '-', h_size_arr, h_lat, 1);                                      \
        }                                                                                         \
                                                                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                                      \
                                                                                                  \
    } while (0)

#define MAIN_CLEANUP(flag_d, stream, h_tables, num_entries) \
    do {                                                    \
        if (flag_d) nvshmem_free(flag_d);                   \
        cudaStreamDestroy(stream);                          \
        free_tables(h_tables, 2);                           \
        finalize_wrapper();                                 \
    } while (0);

#endif /* _ATOMIC_PING_PONG_COMMON_H_ */
