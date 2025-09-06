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

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

__global__ void bw(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_put_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }
        __syncthreads();
    }

    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
    __syncthreads();
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL;
    unsigned int *counter_d;

    read_args(argc, argv);
    int max_blocks = num_blocks, max_threads = threads_per_block;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw = NULL, *h_bw_total = NULL;
    double *d_bw = NULL, *d_bw_sum = NULL;

    int iter = iters;
    int skip = warmup_iters;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    if (bidirectional) {
        h_bw_total = (double *)malloc(sizeof(double) * array_size);

        if (!h_bw_total) {
            fprintf(stderr, "Error: Unable to malloc on the host.\n");
            exit(1);
        }

        memset(h_bw_total, 0, sizeof(double) * array_size);

        d_bw = (double *)nvshmem_malloc(sizeof(double));
        d_bw_sum = (double *)nvshmem_malloc(sizeof(double));
    }

    if (use_mmap) {
        data_d = (double *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (double *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (bidirectional || mype == 0) {
        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            bw<<<max_blocks, max_threads>>>(data_d, counter_d, size / sizeof(double), mype, skip);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            bw<<<max_blocks, max_threads>>>(data_d, counter_d, size / sizeof(double), mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            nvshmem_barrier_all();

            /* Sum all h_bw of each PE for bidirectional mode. */
            if (bidirectional) {
                CUDA_CHECK(cudaMemcpy(d_bw, &h_bw[i], sizeof(double), cudaMemcpyDefault));
                nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, d_bw_sum, d_bw, 1);
                CUDA_CHECK(cudaMemcpy(&h_bw_total[i], d_bw_sum, sizeof(double), cudaMemcpyDefault));
            }

            i++;
        }
    } else {
        for (int size = min_size; size <= max_size; size *= step_factor) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        double *p_h_bw_tmp = bidirectional ? h_bw_total : h_bw;
        const char *const test_name = bidirectional ? "shmem_put_bw_bidi" : "shmem_put_bw_uni";
        print_table_basic(test_name, "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                          p_h_bw_tmp, i);
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }

    if (d_bw) nvshmem_free(d_bw);
    if (d_bw_sum) nvshmem_free(d_bw_sum);

    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
