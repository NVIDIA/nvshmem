/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
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

#define UNROLL 2

template <typename T>
__device__ inline void call_nvshmem_p(T *rptr, T val, int peer) {
    switch (sizeof(T)) {
        case 1:
            nvshmem_uint8_p((uint8_t *)rptr, val, peer);
            break;
        case 2:
            nvshmem_uint16_p((uint16_t *)rptr, val, peer);
            break;
        case 4:
            nvshmem_uint32_p((uint32_t *)rptr, val, peer);
            break;
        case 8:
            nvshmem_double_p((double *)rptr, val, peer);
            break;
        default:
            assert(0);
    }
}

template <typename T>
__global__ void bw(T *data_d, volatile unsigned int *counter_d, int len, int pe, int iter,
                   int stride) {
    int u, i, j, peer, tid, slice;
    unsigned int counter;
    int threads = gridDim.x * blockDim.x;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    peer = !pe;
    slice = UNROLL * threads * stride;

    // When stride > 1, each iteration sends less than len elements.
    // We increase the number of iterations to make up for that.
    for (i = 0; i < iter * stride; i++) {
        for (j = 0; j < len - slice; j += slice) {
            for (u = 0; u < UNROLL; ++u) {
                int idx = j + u * threads + tid * stride;
                call_nvshmem_p<T>(data_d + idx, *(data_d + idx), peer);
            }
            __syncthreads();
        }

        for (u = 0; u < UNROLL; ++u) {
            int idx = j + u * threads + tid * stride;
            if (idx >= 0 && idx < len) call_nvshmem_p<T>(data_d + idx, *(data_d + idx), peer);
        }

        // synchronizing across blocks
        __syncthreads();

        if (!threadIdx.x) {
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

    // synchronizing across blocks
    __syncthreads();

    if (!threadIdx.x) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
}

void call_bw(int blocks, int threads, void *data_d, unsigned int *counter_d, size_t size,
             int element_size, int mype, int iter, int stride) {
    switch (element_size) {
        case 1:
            bw<uint8_t><<<blocks, threads>>>((uint8_t *)data_d, counter_d, size / sizeof(uint8_t),
                                             mype, iter, stride);
            break;
        case 2:
            bw<uint16_t><<<blocks, threads>>>((uint16_t *)data_d, counter_d,
                                              size / sizeof(uint16_t), mype, iter, stride);
            break;
        case 4:
            bw<uint32_t><<<blocks, threads>>>((uint32_t *)data_d, counter_d,
                                              size / sizeof(uint32_t), mype, iter, stride);
            break;
        case 8:
            bw<double><<<blocks, threads>>>((double *)data_d, counter_d, size / sizeof(double),
                                            mype, iter, stride);
            break;
        default:
            fprintf(stderr, "element_size=%d is not supported \n", element_size);
            exit(-EINVAL);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    void *data_d = NULL;
    unsigned int *counter_d;

    read_args(argc, argv);
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;
    double *h_msgrate;
    bool report_msgrate = false;

    int iter = iters;
    int skip = warmup_iters;
    int element_size = datatype.size;

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
    alloc_tables(&h_tables, 3, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];
    h_msgrate = (double *)h_tables[2];

    if (use_mmap) {
        data_d = (void *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (void *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t size;
    i = 0;
    if (mype == 0) {
        for (size = min_size; size <= max_size; size *= step_factor) {
            int blocks = num_blocks, threads = threads_per_block;
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            call_bw(blocks, threads, data_d, counter_d, size, element_size, mype, skip, stride);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            call_bw(blocks, threads, data_d, counter_d, size, element_size, mype, iter, stride);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            h_msgrate[i] = (double)(size / element_size) * iter / (milliseconds * MS_TO_S);
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size = min_size; size <= max_size; size *= step_factor) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_basic("shmem_p_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                          h_bw, i);
        if (report_msgrate)
            print_table_basic("shmem_p_bw", "None", "size (Bytes)", "msgrate", "MMPS", '+',
                              h_size_arr, h_msgrate, i);
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
