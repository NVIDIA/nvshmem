/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "shmem_put_atomic_ping_pong_latency.cubin"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define UNROLL 8

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void ping_pong(int *data_d, uint64_t *flag_d, int len, int pe, int iter,
                          size_t dynamic_smem_size) {
    int i, peer;

    peer = !pe;
    NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);

    for (i = 0; i < iter; i++) {
        if (pe) {
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));

            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmem_uint64_atomic_inc(flag_d, peer);
        } else {
            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmem_uint64_atomic_inc(flag_d, peer);

            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
        }
    }
    nvshmem_quiet();
    NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

void test_ping_pong(void **arglist, CUfunction kernel, cudaStream_t stream,
                    size_t dynamic_smem_size) {
    int status;
    if (use_cubin) {
        if (dynamic_smem_size > 48 * 1024) {
            CU_CHECK(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        (int)dynamic_smem_size));
        }
        CU_CHECK(cuLaunchCooperativeKernel(kernel, 1, 1, 1, 1, 1, 1, dynamic_smem_size, stream,
                                           arglist));
    } else {
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(ping_pong, dynamic_smem_size);
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, arglist,
                                            dynamic_smem_size, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    uint64_t *flag_d = NULL;
    int *data_d = NULL;
    cudaStream_t stream;

    read_args(c, v);
    int iter = iters;
    int skip = warmup_iters;
    size_t max_msg_size = max_size;
    size_t dynamic_smem_size = 0;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;
    perf_stats_t *h_lat_stats = NULL;

    float milliseconds;
    cudaEvent_t start, stop;
    CUfunction test_cubin = NULL;

    init_wrapper(&c, &v);
    if (use_smem && !use_cubin) {
        dynamic_smem_size = NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED;
    }

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
        init_test_case_kernel(&test_cubin, "ping_pong");
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];
    h_lat_stats = (perf_stats_t *)calloc(array_size, sizeof(perf_stats_t));
    if (!h_lat_stats) {
        fprintf(stderr, "Failed to allocate latency statistics\n");
        nvshmem_global_exit(EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_msg_size, mem_handle_type, use_egm, true);
        flag_d = (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t), mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (int *)nvshmem_malloc(max_msg_size);
        flag_d = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems = 0;
        h_size_arr[i] = size;
        nelems = size / sizeof(int);
        void *args_1[] = {&data_d, &flag_d, &nelems, &mype, &skip, &dynamic_smem_size};
        void *args_2[] = {&data_d, &flag_d, &nelems, &mype, &iter, &dynamic_smem_size};

        if (use_egm) {
            memset(flag_d, 0, sizeof(uint64_t));
        } else {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        test_ping_pong(args_1, test_cubin, stream, dynamic_smem_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (size_t repetition = 0; repetition < repetitions; repetition++) {
            if (use_egm) {
                memset(flag_d, 0, sizeof(uint64_t));
            } else {
                CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            cudaEventRecord(start, stream);
            test_ping_pong(args_2, test_cubin, stream, dynamic_smem_size);
            cudaEventRecord(stop, stream);

            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            if (mype == 0) {
                perf_stats_add(h_lat_stats[i], h_lat[i]);
            }
            nvshmem_barrier_all();
        }
        i++;
    }

    if (mype == 0) {
        print_basic_table("shmem_at_ping_lat", "None", "latency", "us", '-', h_size_arr, h_lat, i,
                          h_lat_stats);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    if (flag_d) {
        if (use_mmap) {
            free_mmap_buffer(flag_d);
        } else {
            nvshmem_free(flag_d);
        }
    }
    free(h_lat_stats);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
