/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "shmem_get_latency.cubin"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define THREADS_PER_WARP 32

template <bool USE_ITERATION_BARRIER>
__device__ __forceinline__ void latency_thread_iteration_sync() {
    if constexpr (USE_ITERATION_BARRIER) nvshmem_quiet();
}

template <bool USE_ITERATION_BARRIER>
__device__ __forceinline__ void latency_threadgroup_iteration_sync(int tid) {
    if constexpr (USE_ITERATION_BARRIER) {
        __syncthreads();
        if (!tid) nvshmem_quiet();
        __syncthreads();
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define LATENCY_THREAD(NAME, USE_ITERATION_BARRIER)                                          \
    __global__ void NAME(int *data_d, int len, int pe, int iter, size_t dynamic_smem_size) { \
        int i, peer;                                                                         \
                                                                                             \
        peer = !pe;                                                                          \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                           \
                                                                                             \
        for (i = 0; i < iter; i++) {                                                         \
            nvshmem_int_get_nbi(data_d, data_d, len, peer);                                  \
            latency_thread_iteration_sync<USE_ITERATION_BARRIER>();                          \
        }                                                                                    \
                                                                                             \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                        \
    }

LATENCY_THREAD(latency_kern, true)
LATENCY_THREAD(latency_kern_no_iteration_barrier, false)

#define LATENCY_THREADGROUP(group, NAME, USE_ITERATION_BARRIER)                              \
    __global__ void NAME(int *data_d, int len, int pe, int iter, size_t dynamic_smem_size) { \
        int i, tid, peer;                                                                    \
                                                                                             \
        peer = !pe;                                                                          \
        tid = threadIdx.x;                                                                   \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                           \
                                                                                             \
        for (i = 0; i < iter; i++) {                                                         \
            nvshmemx_int_get_nbi_##group(data_d, data_d, len, peer);                         \
            latency_threadgroup_iteration_sync<USE_ITERATION_BARRIER>(tid);                  \
        }                                                                                    \
                                                                                             \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                        \
    }

LATENCY_THREADGROUP(warp, latency_kern_warp, true)
LATENCY_THREADGROUP(warp, latency_kern_warp_no_iteration_barrier, false)
LATENCY_THREADGROUP(block, latency_kern_block, true)
LATENCY_THREADGROUP(block, latency_kern_block_no_iteration_barrier, false)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DEFINE_TEST_LATENCY(TG)                                                                   \
                                                                                                  \
    void test_latency##TG(int *data_d, int len, int pe, int iter, CUfunction kernel, int threads, \
                          size_t dynamic_smem_size) {                                             \
        if (use_cubin) {                                                                          \
            void *arglist[] = {(void *)&data_d, (void *)&len, (void *)&pe, (void *)&iter,         \
                               (void *)&dynamic_smem_size};                                       \
            if (dynamic_smem_size > 48 * 1024) {                                                  \
                CU_CHECK(cuFuncSetAttribute(kernel,                                               \
                                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,      \
                                            (int)dynamic_smem_size));                             \
            }                                                                                     \
            CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, threads, 1, 1,                               \
                                    (unsigned int)dynamic_smem_size, NULL, arglist, NULL));       \
        } else {                                                                                  \
            if (use_iteration_barrier) {                                                          \
                CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(latency_kern##TG, dynamic_smem_size);           \
                latency_kern##TG<<<1, threads, dynamic_smem_size>>>(data_d, len, pe, iter,        \
                                                                    dynamic_smem_size);           \
            } else {                                                                              \
                CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(latency_kern##TG##_no_iteration_barrier,        \
                                                  dynamic_smem_size);                             \
                latency_kern##TG##_no_iteration_barrier<<<1, threads, dynamic_smem_size>>>(       \
                    data_d, len, pe, iter, dynamic_smem_size);                                    \
            }                                                                                     \
        }                                                                                         \
    }

DEFINE_TEST_LATENCY()
DEFINE_TEST_LATENCY(_warp)
DEFINE_TEST_LATENCY(_block)

int main(int argc, char *argv[]) {
    int mype, npes, size;
    int *data_d = NULL;

    read_args(argc, argv);

    int iter = iters;
    int skip = warmup_iters;
    size_t dynamic_smem_size = 0;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;
    perf_stats_t *h_lat_stats = NULL;

    float milliseconds;
    cudaEvent_t start, stop;
    CUfunction test_cubin = NULL;
    CUfunction test_cubin_warp = NULL;
    CUfunction test_cubin_block = NULL;

    init_wrapper(&argc, &argv);
    if (use_smem && !use_cubin) {
        dynamic_smem_size = NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED;
    }

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
        init_test_case_kernel(&test_cubin, use_iteration_barrier
                                               ? "latency_kern"
                                               : "latency_kern_no_iteration_barrier");
        init_test_case_kernel(&test_cubin_warp, use_iteration_barrier
                                                    ? "latency_kern_warp"
                                                    : "latency_kern_warp_no_iteration_barrier");
        init_test_case_kernel(&test_cubin_block, use_iteration_barrier
                                                     ? "latency_kern_block"
                                                     : "latency_kern_block_no_iteration_barrier");
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (int *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];
    h_lat_stats = (perf_stats_t *)calloc(array_size, sizeof(perf_stats_t));
    if (!h_lat_stats) goto finalize;

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency(data_d, nelems, mype, skip, test_cubin, 1, dynamic_smem_size);
            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                cudaEventRecord(start);
                test_latency(data_d, nelems, mype, iter, test_cubin, 1, dynamic_smem_size);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                h_lat[i] = (milliseconds * 1000) / iter;
                perf_stats_add(h_lat_stats[i], h_lat[i]);
            }
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_basic_table("shmem_g_latency", "Thread", "latency", "us", '-', h_size_arr, h_lat, i,
                          h_lat_stats);
    }
    memset(h_lat_stats, 0, array_size * sizeof(perf_stats_t));

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency_warp(data_d, nelems, mype, skip, test_cubin_warp, THREADS_PER_WARP,
                              dynamic_smem_size);
            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                cudaEventRecord(start);
                test_latency_warp(data_d, nelems, mype, iter, test_cubin_warp, THREADS_PER_WARP,
                                  dynamic_smem_size);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                h_lat[i] = (milliseconds * 1000) / iter;
                perf_stats_add(h_lat_stats[i], h_lat[i]);
            }
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_basic_table("shmem_get_latency", "Warp", "latency", "us", '-', h_size_arr, h_lat, i,
                          h_lat_stats);
    }
    memset(h_lat_stats, 0, array_size * sizeof(perf_stats_t));

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency_block(data_d, nelems, mype, skip, test_cubin_block, threads_per_block,
                               dynamic_smem_size);
            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                cudaEventRecord(start);
                test_latency_block(data_d, nelems, mype, iter, test_cubin_block, threads_per_block,
                                   dynamic_smem_size);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                h_lat[i] = (milliseconds * 1000) / iter;
                perf_stats_add(h_lat_stats[i], h_lat[i]);
            }
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_basic_table("shmem_get_latency", "Block", "latency", "us", '-', h_size_arr, h_lat, i,
                          h_lat_stats);
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    free_tables(h_tables, 2);
    free(h_lat_stats);
    finalize_wrapper();

    return 0;
}
