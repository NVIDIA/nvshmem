/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "bcast_latency.cubin"

#include "coll_test.h"
#define DATATYPE int64_t

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define CALL_BCAST(TYPENAME, TYPE, TG_PRE, THREADGROUP, THREAD_COMP, ELEM_COMP)                    \
    __global__ void test_##TYPENAME##_bcast_call_kern##THREADGROUP(                                \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int mype, int PE_root,    \
        int iter) {                                                                                \
        int i;                                                                                     \
                                                                                                   \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                  \
            for (i = 0; i < iter; i++) {                                                           \
                nvshmem##TG_PRE##_##TYPENAME##_broadcast##THREADGROUP(team, dest, source, nelems,  \
                                                                      PE_root);                    \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    void test_##TYPENAME##_bcast_call_kern##THREADGROUP##_cubin(                                   \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist) {                        \
        CUfunction test_cubin;                                                                     \
                                                                                                   \
        init_test_case_kernel(                                                                     \
            &test_cubin, NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_bcast_call_kern##THREADGROUP)); \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, num_blocks, 1, 1, num_tpb, 1, 1, 0, stream, \
                                           arglist));                                              \
    }

#define CALL_BCAST_KERNEL(TYPENAME, THREADGROUP, BLOCKS, THREADS, ARG_LIST, STREAM)                \
    if (use_cubin) {                                                                               \
        test_##TYPENAME##_bcast_call_kern##THREADGROUP##_cubin(BLOCKS, THREADS, STREAM, ARG_LIST); \
    } else {                                                                                       \
        status = nvshmemx_collective_launch(                                                       \
            (const void *)test_##TYPENAME##_bcast_call_kern##THREADGROUP, BLOCKS, THREADS,         \
            ARG_LIST, 0, STREAM);                                                                  \
        if (status != NVSHMEMX_SUCCESS) {                                                          \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                      \
            exit(-1);                                                                              \
        }                                                                                          \
    }

CALL_BCAST(int32, int32_t, , , 1, 512);
CALL_BCAST(int64, int64_t, , , 1, 512);
CALL_BCAST(int32, int32_t, x, _warp, warpSize, 4096);
CALL_BCAST(int64, int64_t, x, _warp, warpSize, 4096);
CALL_BCAST(int32, int32_t, x, _block, INT_MAX, INT_MAX);
CALL_BCAST(int64, int64_t, x, _block, INT_MAX, INT_MAX);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int broadcast_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                             int PE_root, cudaStream_t stream, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int num_blocks = 1;
    size_t num_elems = 1;
    size_t min_elems, max_elems;
    int thread_points, warp_points, block_points;
    int skip = warmup_iters;
    int iter = iters;
    int npes = nvshmem_n_pes();
    uint64_t *h_size_array = (uint64_t *)h_tables[0];
    double *h_thread_lat = (double *)h_tables[1];
    double *h_warp_lat = (double *)h_tables[2];
    double *h_block_lat = (double *)h_tables[3];
    std::vector<perf_stats_t> h_thread_stats(max_size_log);
    std::vector<perf_stats_t> h_warp_stats(max_size_log);
    std::vector<perf_stats_t> h_block_stats(max_size_log);
    float milliseconds;
    void *args_1[] = {&team, &dest, &source, &num_elems, &mype, &PE_root, &skip};
    void *args_2[] = {&team, &dest, &source, &num_elems, &mype, &PE_root, &iter};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *ms_d = (float *)nvshmem_malloc(sizeof(float));
    float *ms_sum_d = (float *)nvshmem_malloc(sizeof(float));

    nvshmem_barrier_all();
    min_elems = max(static_cast<size_t>(1), min_size / sizeof(int32_t));
    max_elems = max(static_cast<size_t>(1), max_size / sizeof(int32_t));
    thread_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems && num_elems < 512;
         num_elems *= step_factor) {
        measure_device_latency_batches(
            [&] { CALL_BCAST_KERNEL(int32, , num_blocks, nvshm_test_num_tpb, args_1, stream); },
            [&] { CALL_BCAST_KERNEL(int32, , num_blocks, nvshm_test_num_tpb, args_2, stream); },
            stream, mype, iter, &h_thread_lat[thread_points], &h_thread_stats[thread_points]);
        thread_points++;
    }

    warp_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems && num_elems < 4096;
         num_elems *= step_factor) {
        measure_device_latency_batches(
            [&] {
                CALL_BCAST_KERNEL(int32, _warp, num_blocks, nvshm_test_num_tpb, args_1, stream);
            },
            [&] {
                CALL_BCAST_KERNEL(int32, _warp, num_blocks, nvshm_test_num_tpb, args_2, stream);
            },
            stream, mype, iter, &h_warp_lat[warp_points], &h_warp_stats[warp_points]);
        warp_points++;
    }

    block_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {
        h_size_array[block_points] =
            calculate_collective_size("bcast", num_elems, sizeof(int32_t), npes);
        measure_device_latency_batches(
            [&] {
                CALL_BCAST_KERNEL(int32, _block, num_blocks, nvshm_test_num_tpb, args_1, stream);
            },
            [&] {
                CALL_BCAST_KERNEL(int32, _block, num_blocks, nvshm_test_num_tpb, args_2, stream);
            },
            stream, mype, iter, &h_block_lat[block_points], &h_block_stats[block_points]);
        block_points++;
    }

    if (!mype) {
        print_device_collective_table("bcast_device", "32-bit-thread", "latency", "us", '-',
                                      h_size_array, h_thread_lat, thread_points,
                                      h_thread_stats.data());
        print_device_collective_table("bcast_device", "32-bit-warp", "latency", "us", '-',
                                      h_size_array, h_warp_lat, warp_points, h_warp_stats.data());
        print_device_collective_table("bcast_device", "32-bit-block", "latency", "us", '-',
                                      h_size_array, h_block_lat, block_points,
                                      h_block_stats.data());
    }

    std::fill(h_thread_stats.begin(), h_thread_stats.end(), perf_stats_t{});
    std::fill(h_warp_stats.begin(), h_warp_stats.end(), perf_stats_t{});
    std::fill(h_block_stats.begin(), h_block_stats.end(), perf_stats_t{});

    min_elems = max(static_cast<size_t>(1), min_size / sizeof(int64_t));
    max_elems = max(static_cast<size_t>(1), max_size / sizeof(int64_t));
    thread_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems && num_elems < 512;
         num_elems *= step_factor) {
        measure_device_latency_batches(
            [&] { CALL_BCAST_KERNEL(int64, , num_blocks, nvshm_test_num_tpb, args_1, stream); },
            [&] { CALL_BCAST_KERNEL(int64, , num_blocks, nvshm_test_num_tpb, args_2, stream); },
            stream, mype, iter, &h_thread_lat[thread_points], &h_thread_stats[thread_points]);
        thread_points++;
    }

    warp_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems && num_elems < 4096;
         num_elems *= step_factor) {
        measure_device_latency_batches(
            [&] {
                CALL_BCAST_KERNEL(int64, _warp, num_blocks, nvshm_test_num_tpb, args_1, stream);
            },
            [&] {
                CALL_BCAST_KERNEL(int64, _warp, num_blocks, nvshm_test_num_tpb, args_2, stream);
            },
            stream, mype, iter, &h_warp_lat[warp_points], &h_warp_stats[warp_points]);
        warp_points++;
    }

    block_points = 0;
    for (num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {
        h_size_array[block_points] =
            calculate_collective_size("bcast", num_elems, sizeof(int64_t), npes);
        CALL_BCAST_KERNEL(int64, _block, num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        for (size_t repetition = 0; repetition < repetitions; repetition++) {
            cudaEventRecord(start, stream);
            CALL_BCAST_KERNEL(int64, _block, num_blocks, nvshm_test_num_tpb, args_2, stream);
            cudaEventRecord(stop, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaMemcpy(ms_d, &milliseconds, sizeof(float), cudaMemcpyHostToDevice);
            nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, ms_sum_d, ms_d, 1);
            cudaMemcpy(&milliseconds, ms_sum_d, sizeof(float), cudaMemcpyDeviceToHost);
            if (!mype) {
                h_block_lat[block_points] = (milliseconds * 1000.0) /
                                            ((float)iter * nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD));
                perf_stats_add(h_block_stats[block_points], h_block_lat[block_points]);
            }
            nvshmem_barrier_all();
        }
        block_points++;
    }

    if (!mype) {
        print_device_collective_table("bcast_device", "64-bit-thread", "latency", "us", '-',
                                      h_size_array, h_thread_lat, thread_points,
                                      h_thread_stats.data());
        print_device_collective_table("bcast_device", "64-bit-warp", "latency", "us", '-',
                                      h_size_array, h_warp_lat, warp_points, h_warp_stats.data());
        print_device_collective_table("bcast_device", "64-bit-block", "latency", "us", '-',
                                      h_size_array, h_block_lat, block_points,
                                      h_block_stats.data());
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;

    read_args(argc, argv);
    size_t size = max_size * 2;
    size_t alloc_size;
    DATATYPE *buffer = NULL;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    int root = 0;
    char size_string[100];
    cudaStream_t cstrm;
    void **h_tables;

    array_size = max_size_log;
    if (use_mmap) {
        size = pad_up(size) * 2;
    }
    DEBUG_PRINT("symmetric size %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 4, array_size);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype,
                nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD));

    alloc_size = max_size * 2;

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[max_size / sizeof(DATATYPE)];

    if (use_mmap) {
        buffer = (DATATYPE *)allocate_mmap_buffer(alloc_size, mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmap buffer of size %zu\n", alloc_size);
    } else {
        buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
        DEBUG_PRINT("Allocating nvshmem malloc buffer of size %zu\n", alloc_size);
    }

    if (!buffer) {
        fprintf(stderr, "buffer allocation failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)buffer;
    d_dest = (DATATYPE *)&d_source[max_size / sizeof(DATATYPE)];

    for (int i = 0; i < max_size / sizeof(DATATYPE); i++) {
        h_source[i] = i;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, max_size, cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, max_size, cudaMemcpyHostToDevice, cstrm));

    broadcast_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, root, cstrm, h_tables);

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, max_size, cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, max_size, cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    if (use_mmap) {
        free_mmap_buffer(buffer);
    } else {
        nvshmem_free(buffer);
    }

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 4);
    finalize_wrapper();

out:
    return 0;
}
