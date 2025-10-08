/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#define CUMODULE_NAME "barrier_latency.cubin"

#include "coll_test.h"

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define BARRIER_KERNEL_WRAPPER(TG_PRE, THREADGROUP, THREAD_COMP, VARIANT, VARIANT_API)             \
    void test_barrier##VARIANT##call_kernel##VARIANT_API##THREADGROUP##_cubin(                     \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist) {                        \
        CUfunction test_cubin;                                                                     \
                                                                                                   \
        init_test_case_kernel(&test_cubin,                                                         \
                              NVSHMEMI_TEST_STRINGIFY(                                             \
                                  test_barrier##VARIANT##call_kernel##VARIANT_API##THREADGROUP));  \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, num_blocks, 1, 1, num_tpb, 1, 1, 0, stream, \
                                           arglist));                                              \
    }

#define BARRIER_KERNEL(TG_PRE, THREADGROUP, THREAD_COMP)                                   \
    __global__ void test_barrier_call_kernel##THREADGROUP(nvshmem_team_t team, int iter) { \
        int i;                                                                             \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP)) {                                  \
            for (i = 0; i < iter; i++) {                                                   \
                nvshmem##TG_PRE##_barrier##THREADGROUP(team);                              \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    __global__ void test_barrier_all_call_kernel##THREADGROUP(int iter) {                  \
        int i;                                                                             \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP)) {                                  \
            for (i = 0; i < iter; i++) {                                                   \
                nvshmem##TG_PRE##_barrier_all##THREADGROUP();                              \
            }                                                                              \
        }                                                                                  \
    }

#define CALL_BARRIER_KERNEL(THREADGROUP, BLOCKS, THREADS, ARG_LIST, STREAM, VARIANT)        \
    if (use_cubin) {                                                                        \
        test_barrier##VARIANT##call_kernel##THREADGROUP##_cubin(BLOCKS, THREADS, STREAM,    \
                                                                ARG_LIST);                  \
    } else {                                                                                \
        status = nvshmemx_collective_launch(                                                \
            (const void *)test_barrier##VARIANT##call_kernel##THREADGROUP, BLOCKS, THREADS, \
            ARG_LIST, 0, STREAM);                                                           \
        if (status != NVSHMEMX_SUCCESS) {                                                   \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);               \
            exit(-1);                                                                       \
        }                                                                                   \
    }

BARRIER_KERNEL(, , 1);
BARRIER_KERNEL(x, _warp, warpSize);
BARRIER_KERNEL(x, _block, INT_MAX);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

BARRIER_KERNEL_WRAPPER(, , 1, _, );
BARRIER_KERNEL_WRAPPER(x, warp, warpSize, _, _);
BARRIER_KERNEL_WRAPPER(x, block, INT_MAX, _, _);
BARRIER_KERNEL_WRAPPER(, , 1, _all_, );
BARRIER_KERNEL_WRAPPER(x, warp, warpSize, _all_, _);
BARRIER_KERNEL_WRAPPER(x, block, INT_MAX, _all_, _);

int barrier_calling_kernel(nvshmem_team_t team, cudaStream_t stream, int mype, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    size_t skip = warmup_iters;
    size_t iter = iters;
    int num_blocks = 1;
    int npes = nvshmem_n_pes();
    double *h_thread_lat = (double *)h_tables[0];
    double *h_warp_lat = (double *)h_tables[1];
    double *h_block_lat = (double *)h_tables[2];
    uint64_t tpb_size = nvshm_test_num_tpb;
    void *barrier_args_1[] = {&team, &skip};
    void *barrier_args_2[] = {&team, &iter};
    void *barrier_all_args_1[] = {&skip};
    void *barrier_all_args_2[] = {&iter};
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(, num_blocks, nvshm_test_num_tpb, barrier_args_1, stream, _)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(, num_blocks, nvshm_test_num_tpb, barrier_args_2, stream, _)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, barrier_args_1, stream, _)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, barrier_args_2, stream, _)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(_block, num_blocks, nvshm_test_num_tpb, barrier_args_1, stream, _)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(_block, num_blocks, nvshm_test_num_tpb, barrier_args_2, stream, _)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    if (!mype) {
        print_table_basic("barrier_device", "thread", "threads per block", "latency", "us", '-',
                          &tpb_size, h_thread_lat, 1);
        print_table_basic("barrier_device", "warp", "threads per block", "latency", "us", '-',
                          &tpb_size, h_warp_lat, 1);
        print_table_basic("barrier_device", "block", "threads per block", "latency", "us", '-',
                          &tpb_size, h_block_lat, 1);
    }

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(, num_blocks, nvshm_test_num_tpb, barrier_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(, num_blocks, nvshm_test_num_tpb, barrier_all_args_2, stream, _all_)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, barrier_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(_warp, num_blocks, nvshm_test_num_tpb, barrier_all_args_2, stream, _all_)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    CALL_BARRIER_KERNEL(_block, num_blocks, nvshm_test_num_tpb, barrier_all_args_1, stream, _all_)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    CALL_BARRIER_KERNEL(_block, num_blocks, nvshm_test_num_tpb, barrier_all_args_1, stream, _all_)

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    if (!mype) {
        print_table_basic("barrier_all_device", "thread", "threads per block", "latency", "us", '-',
                          &tpb_size, h_thread_lat, 1);
        print_table_basic("barrier_all_device", "warp", "threads per block", "latency", "us", '-',
                          &tpb_size, h_warp_lat, 1);
        print_table_basic("barrier_all_device", "block", "threads per block", "latency", "us", '-',
                          &tpb_size, h_block_lat, 1);
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

    barrier_calling_kernel(NVSHMEM_TEAM_WORLD, cstrm, mype, h_tables);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
