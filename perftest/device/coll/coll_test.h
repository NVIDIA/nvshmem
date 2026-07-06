/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COLL_TEST_H
#define COLL_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>

using namespace std;

#define MAX_ITERS 100
#define MAX_SKIP 10
#define BARRIER_MAX_ITERS 1000
#define BARRIER_MAX_SKIP 10
#define TEST_NUM_TPB_BLOCK 256

#define NVSHMEM_PERF_COLL_DYNAMIC_SMEM_SIZE() \
    ((use_smem && !use_cubin) ? NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED : 0)

#define NVSHMEM_PERF_CU_LAUNCH_COOP(kernel, num_blocks, num_tpb, stream, arglist,                  \
                                    dynamic_smem_size)                                             \
    do {                                                                                           \
        if ((dynamic_smem_size) > 48 * 1024) {                                                     \
            CU_CHECK(cuFuncSetAttribute((kernel), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, \
                                        (int)(dynamic_smem_size)));                                \
        }                                                                                          \
        CU_CHECK(cuLaunchCooperativeKernel((kernel), (num_blocks), 1, 1, (num_tpb), 1, 1,          \
                                           (unsigned int)(dynamic_smem_size), (stream),            \
                                           (arglist)));                                            \
    } while (0)

#define NVSHMEM_PERF_COLLECTIVE_LAUNCH(status, kernel, blocks, threads, arglist,           \
                                       dynamic_smem_size, stream)                          \
    do {                                                                                   \
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((kernel), (dynamic_smem_size));                  \
        (status) = nvshmemx_collective_launch((const void *)(kernel), (blocks), (threads), \
                                              (arglist), (dynamic_smem_size), (stream));   \
        if ((status) != NVSHMEMX_SUCCESS) {                                                \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", (status));            \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

typedef struct run_opt {
    int run_thread;
    int run_warp;
    int run_block;
} run_opt_t;

template <typename WarmupFn, typename TimedFn>
void measure_device_latency_batches(WarmupFn warmup, TimedFn timed, cudaStream_t stream, int mype,
                                    size_t iter, double *last_value, perf_stats_t *stats) {
    float milliseconds;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    warmup();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        timed();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            *last_value = (milliseconds * 1000.0) / (double)iter;
            perf_stats_add(*stats, *last_value);
        }
        nvshmem_barrier_all();
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

#define cuda_check_error()                                                                   \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (cudaSuccess != e) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(-1);                                                                        \
        }                                                                                    \
    }

#define PROCESS_OPTS(run_options)                          \
    do {                                                   \
        int opt;                                           \
        run_options.run_thread = 1;                        \
        run_options.run_warp = 1;                          \
        run_options.run_block = 1;                         \
                                                           \
        while ((opt = getopt(argc, argv, "twba")) != -1) { \
            switch (opt) {                                 \
                case 't':                                  \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 0;              \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'w':                                  \
                    run_options.run_warp = 1;              \
                    run_options.run_thread = 0;            \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'b':                                  \
                    run_options.run_block = 1;             \
                    run_options.run_thread = 0;            \
                    run_options.run_warp = 0;              \
                    break;                                 \
                case 'a':                                  \
                default:                                   \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 1;              \
                    run_options.run_block = 1;             \
                    break;                                 \
            }                                              \
        }                                                  \
    } while (0)

int page_size_roundoff(int value) {
    int page_sz = getpagesize();
    return ((value + page_sz - 1) / page_sz) * page_sz;
}

#endif /*COLL_TEST_H*/
