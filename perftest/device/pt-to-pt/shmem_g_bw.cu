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
__device__ inline T call_nvshmem_g(T *rptr, int peer) {
    switch (sizeof(T)) {
        case 1:
            return nvshmem_uint8_g((uint8_t *)rptr, peer);
            break;
        case 2:
            return nvshmem_uint16_g((uint16_t *)rptr, peer);
            break;
        case 4:
            return nvshmem_uint32_g((uint32_t *)rptr, peer);
            break;
        case 8:
            return nvshmem_double_g((double *)rptr, peer);
            break;
        default:
            assert(0);
    }
    return (T)0;
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

    // When stride > 1, each iteration requests less than len elements.
    // We increase the number of iterations to make up for that.
    for (i = 0; i < iter * stride; i++) {
        for (j = 0; j < len - slice; j += slice) {
            for (u = 0; u < UNROLL; ++u) {
                int idx = j + u * threads + tid * stride;
                *(data_d + idx) = call_nvshmem_g<T>(data_d + idx, peer);
            }
            __syncthreads(); /* This is required for performance over PCIe. PCIe has a P2P mailbox
                                protocol that has a window of 64KB for device BAR addresses. Not
                                synchronizing
                                across threads will lead to jumping in and out of the 64K window */
        }

        for (u = 0; u < UNROLL; ++u) {
            int idx = j + u * threads + tid * stride;
            if (idx < len) *(data_d + idx) = call_nvshmem_g<T>(data_d + idx, peer);
        }

        // synchronizing across blocks
        __syncthreads();

        if (!threadIdx.x) {
            __threadfence(); /* To ensure that the data received through shmem_g is
                                visible across the gpu */
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
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
}

void call_bw(int blocks, int threads, void *data_d, unsigned int *counter_d, size_t size,
             NVSHMEM_DATATYPE_T dt, int mype, int iter, int stride) {
    switch (dt) {
        case NVSHMEM_INT:
            bw<int><<<blocks, threads>>>((int *)data_d, counter_d, size / sizeof(uint8_t), mype,
                                         iter, stride);
            break;
        case NVSHMEM_LONG:
            bw<long><<<blocks, threads>>>((long *)data_d, counter_d, size / sizeof(uint16_t), mype,
                                          iter, stride);
            break;
        case NVSHMEM_LONGLONG:
            bw<long long><<<blocks, threads>>>((long long *)data_d, counter_d,
                                               size / sizeof(uint32_t), mype, iter, stride);
            break;
        case NVSHMEM_ULONGLONG:
            bw<unsigned long long><<<blocks, threads>>>((unsigned long long *)data_d, counter_d,
                                                        size / sizeof(double), mype, iter, stride);
            break;
        case NVSHMEM_FLOAT:
            bw<float><<<blocks, threads>>>((float *)data_d, counter_d, size / sizeof(double), mype,
                                           iter, stride);
            break;
        case NVSHMEM_DOUBLE:
            bw<double><<<blocks, threads>>>((double *)data_d, counter_d, size / sizeof(double),
                                            mype, iter, stride);
            break;
        case NVSHMEM_UINT:
            bw<unsigned int><<<blocks, threads>>>((unsigned int *)data_d, counter_d,
                                                  size / sizeof(double), mype, iter, stride);
            break;
        case NVSHMEM_INT32:
            bw<int32_t><<<blocks, threads>>>((int32_t *)data_d, counter_d, size / sizeof(double),
                                             mype, iter, stride);
            break;
        case NVSHMEM_UINT32:
            bw<uint32_t><<<blocks, threads>>>((uint32_t *)data_d, counter_d, size / sizeof(double),
                                              mype, iter, stride);
            break;
        case NVSHMEM_INT64:
            bw<int64_t><<<blocks, threads>>>((int64_t *)data_d, counter_d, size / sizeof(double),
                                             mype, iter, stride);
            break;
        case NVSHMEM_UINT64:
            bw<uint64_t><<<blocks, threads>>>((uint64_t *)data_d, counter_d, size / sizeof(double),
                                              mype, iter, stride);
            break;
        case NVSHMEM_FP16:
            bw<half><<<blocks, threads>>>((half *)data_d, counter_d, size / sizeof(double), mype,
                                          iter, stride);
            break;
#if CUDA_VERSION >= 12020
        case NVSHMEM_BF16:
            bw<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16 *)data_d, counter_d,
                                                   size / sizeof(double), mype, iter, stride);
            break;
#endif
        default:
            fprintf(stderr, "element=%d is not supported \n", dt);
            exit(-EINVAL);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    void *data_d = NULL;
    unsigned int *counter_d;

    read_args(argc, argv);
    int max_blocks = num_blocks, max_threads = threads_per_block;
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
            int blocks = max_blocks, threads = max_threads;
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            call_bw(blocks, threads, data_d, counter_d, size, datatype.type, mype, skip, stride);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            call_bw(blocks, threads, data_d, counter_d, size, datatype.type, mype, iter, stride);
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
        print_table_basic("shmem_g_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                          h_bw, i);
        if (report_msgrate)
            print_table_basic("shmem_g_bw", "None", "size (Bytes)", "msgrate", "MMPS", '+',
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
