/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "sync_latency.cubin"

#include "coll_test.h"

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define SYNC_KERNEL(TG_PRE, THREADGROUP, THREAD_COMP, VARIANT, VARIANT_API, TEAM, TEAM_DELIM)  \
    void test_sync##TEAM_DELIM##TEAM##VARIANT##call_kernel##THREADGROUP##_cubin(               \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist,                      \
        size_t dynamic_smem_size) {                                                            \
        CUfunction test_cubin;                                                                 \
                                                                                               \
        init_test_case_kernel(                                                                 \
            &test_cubin, NVSHMEMI_TEST_STRINGIFY(                                              \
                             test_sync##TEAM_DELIM##TEAM##VARIANT##call_kernel##THREADGROUP)); \
        NVSHMEM_PERF_CU_LAUNCH_COOP(test_cubin, num_blocks, num_tpb, stream, arglist,          \
                                    dynamic_smem_size);                                        \
    }                                                                                          \
                                                                                               \
    __global__ void test_sync##TEAM_DELIM##TEAM##VARIANT##call_kernel##THREADGROUP(            \
        int iter, nvshmem_team_t team, size_t dynamic_smem_size) {                             \
        int i;                                                                                 \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                             \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP)) {                                      \
            for (i = 0; i < iter; i++) {                                                       \
                nvshmem##TG_PRE##TEAM_DELIM##TEAM##_sync##VARIANT_API##THREADGROUP(TEAM);      \
            }                                                                                  \
        }                                                                                      \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                          \
    }

#define CALL_SYNC_KERNEL(THREADGROUP, BLOCKS, THREADS, ARG_LIST, STREAM, VARIANT)               \
    if (use_cubin) {                                                                            \
        test_sync##VARIANT##call_kernel##THREADGROUP##_cubin(BLOCKS, THREADS, STREAM, ARG_LIST, \
                                                             dynamic_smem_size);                \
    } else {                                                                                    \
        NVSHMEM_PERF_COLLECTIVE_LAUNCH(status, test_sync##VARIANT##call_kernel##THREADGROUP,    \
                                       BLOCKS, THREADS, ARG_LIST, dynamic_smem_size, STREAM);   \
    }

SYNC_KERNEL(, , 1, _, , team, _);
SYNC_KERNEL(x, _warp, warpSize, _, , team, _);
SYNC_KERNEL(x, _block, INT_MAX, _, , team, _);

SYNC_KERNEL(, , 1, _all_, _all, , );
SYNC_KERNEL(x, _warp, warpSize, _all_, _all, , );
SYNC_KERNEL(x, _block, INT_MAX, _all_, _all, , );

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int sync_calling_kernel(nvshmem_team_t team, cudaStream_t stream, int mype, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int skip = warmup_iters;
    int iter = iters;
    size_t dynamic_smem_size = NVSHMEM_PERF_COLL_DYNAMIC_SMEM_SIZE();
    int num_blocks = 1;
    double *h_thread_lat = (double *)h_tables[0];
    double *h_warp_lat = (double *)h_tables[1];
    double *h_block_lat = (double *)h_tables[2];
    perf_stats_t thread_stats = {}, warp_stats = {}, block_stats = {};
    perf_stats_t all_thread_stats = {}, all_warp_stats = {}, all_block_stats = {};

    uint64_t tpb_size = (uint64_t)nvshm_test_num_tpb;

    void *sync_args_1[] = {&skip, &team, &dynamic_smem_size};
    void *sync_args_2[] = {&iter, &team, &dynamic_smem_size};
    void *sync_all_args_1[] = {&skip, &team, &dynamic_smem_size};
    void *sync_all_args_2[] = {&iter, &team, &dynamic_smem_size};
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(, num_blocks, nvshm_test_num_tpb, sync_args_1, stream, _team_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(, num_blocks, nvshm_test_num_tpb, sync_args_2, stream, _team_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(thread_stats, h_thread_lat[0]);
        }
        nvshmem_barrier_all();
    }

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, sync_args_1, stream, _team_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, sync_args_2, stream, _team_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(warp_stats, h_warp_lat[0]);
        }
        nvshmem_barrier_all();
    }

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(_block, num_blocks, nvshm_test_num_tpb, sync_args_1, stream, _team_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(_block, num_blocks, nvshm_test_num_tpb, sync_args_2, stream, _team_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(block_stats, h_block_lat[0]);
        }
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_basic_table("sync_device", "thread", "latency", "us", '-', &tpb_size, h_thread_lat, 1,
                          &thread_stats);
        print_basic_table("sync_device", "warp", "latency", "us", '-', &tpb_size, h_warp_lat, 1,
                          &warp_stats);
        print_basic_table("sync_device", "block", "latency", "us", '-', &tpb_size, h_block_lat, 1,
                          &block_stats);
    }

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(, num_blocks, nvshm_test_num_tpb, sync_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();
    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(, num_blocks, nvshm_test_num_tpb, sync_all_args_2, stream, _all_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(all_thread_stats, h_thread_lat[0]);
        }
        nvshmem_barrier_all();
    }

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, sync_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, sync_all_args_2, stream, _all_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(all_warp_stats, h_warp_lat[0]);
        }
        nvshmem_barrier_all();
    }

    nvshmem_barrier_all();
    CALL_SYNC_KERNEL(_block, num_blocks, nvshm_test_num_tpb, sync_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        cudaEventRecord(start, stream);
        CALL_SYNC_KERNEL(_block, num_blocks, nvshm_test_num_tpb, sync_all_args_2, stream, _all_)
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
            perf_stats_add(all_block_stats, h_block_lat[0]);
        }
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_basic_table("sync_all_device", "thread", "latency", "us", '-', &tpb_size,
                          h_thread_lat, 1, &all_thread_stats);
        print_basic_table("sync_all_device", "warp", "latency", "us", '-', &tpb_size, h_warp_lat, 1,
                          &all_warp_stats);
        print_basic_table("sync_all_device", "block", "latency", "us", '-', &tpb_size, h_block_lat,
                          1, &all_block_stats);
    }

    return status;
}

int main(int argc, char **argv) {
    int mype;
    cudaStream_t cstrm;
    void **h_tables;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 3, 1);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    sync_calling_kernel(NVSHMEM_TEAM_WORLD, cstrm, mype, h_tables);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
