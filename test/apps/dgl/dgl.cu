/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"
#include "perf_utils.h"

#define MAX_MSG_SIZE (64 * 1024)

#define MAX_ITERS 1
#define MAX_SKIP 1
#define BLOCKS 1
#define THREADS_PER_BLOCK 1

__global__ void bw(double *data_d, double *ldata_d, volatile unsigned int *counter_d, int len,
                   int pe, int iter) {
    int peer = (pe + 1) % nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    for (int i = 0; i < iter; i++) {
        nvshmem_getmem(ldata_d, data_d, len * sizeof(double), peer);
    }
}

__global__ void bw_nbi(double *data_d, double *ldata_d, volatile unsigned int *counter_d, int len,
                       int pe, int iter) {
    unsigned int counter;
    int tid = threadIdx.x;

    int peer = (pe + 1) % nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    for (int i = 0; i < iter; i++) {
        nvshmem_getmem_nbi(ldata_d, data_d, len * sizeof(double), peer);
    }

    // synchronizing across blocks
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != 1)
            ;
    }
}

int main(int argc, char *argv[]) {
    int mype;
    double *data_d = NULL, *ldata_d = NULL;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS_PER_BLOCK;
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw, *h_bw_nbi;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();

    while (1) {
        int c;
        c = getopt(argc, argv, "c:t:h");
        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-c [CTAs] -t [THREADS] \n");
                goto finalize;
        }
    }

    data_d = (double *)nvshmem_malloc(MAX_MSG_SIZE);
    CUDA_CHECK(cudaMemset(data_d, 0, MAX_MSG_SIZE));

    CUDA_CHECK(cudaMalloc((void **)&ldata_d, MAX_MSG_SIZE));
    nvshmemx_buffer_register(ldata_d, MAX_MSG_SIZE);

    array_size = floor(std::log2((float)MAX_MSG_SIZE)) + 1;
    alloc_tables(&h_tables, 3, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];
    h_bw_nbi = (double *)h_tables[2];

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        i = 0;
        CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
        bw<<<max_blocks, max_threads>>>(data_d, ldata_d, counter_d, MAX_MSG_SIZE / sizeof(double),
                                        mype, skip);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (size_t size = 512; size <= MAX_MSG_SIZE; size *= 2) {
            h_size_arr[i] = size;

            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            cudaEventRecord(start);
            bw<<<max_blocks, max_threads>>>(data_d, ldata_d, counter_d, size / sizeof(double), mype,
                                            iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] =
                (size * max_blocks * max_threads) / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));

            nvshmem_barrier_all();

            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            cudaEventRecord(start);
            bw_nbi<<<max_blocks, max_threads>>>(data_d, ldata_d, counter_d, size / sizeof(double),
                                                mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw_nbi[i] =
                (size * max_blocks * max_threads) / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));

            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size_t size = 512; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_v1("shmem_get_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                       h_bw, i);
        print_table_v1("shmem_get_bw_nbi", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                       h_bw_nbi, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
