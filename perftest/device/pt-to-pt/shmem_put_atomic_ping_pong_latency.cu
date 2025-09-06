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

#define CUMODULE_NAME "shmem_put_atomic_ping_pong_latency.cubin"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define MAX_MSG_SIZE 1 * 1024 * 1024
#define UNROLL 8

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void ping_pong(int *data_d, uint64_t *flag_d, int len, int pe, int iter) {
    int i, peer;

    peer = !pe;

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
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

void test_ping_pong(void **arglist, CUfunction kernel, cudaStream_t stream) {
    int status;
    if (use_cubin) {
        CU_CHECK(cuLaunchCooperativeKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, arglist));
    } else {
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, arglist, 0, stream);
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

    int iter = 500;
    int skip = 50;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    float milliseconds;
    cudaEvent_t start, stop;
    CUfunction test_cubin = NULL;

    init_wrapper(&c, &v);

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
        void *args_1[] = {&data_d, &flag_d, &nelems, &mype, &skip};
        void *args_2[] = {&data_d, &flag_d, &nelems, &mype, &iter};

        if (use_egm) {
            memset(flag_d, 0, sizeof(uint64_t));
        } else {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        test_ping_pong(args_1, test_cubin, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (use_egm) {
            memset(flag_d, 0, sizeof(uint64_t));
        } else {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        }
        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        test_ping_pong(args_2, test_cubin, stream);
        cudaEventRecord(stop, stream);

        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_lat[i] = (milliseconds * 1000) / iter;
        nvshmem_barrier_all();
        i++;
    }

    if (mype == 0) {
        print_table_basic("shmem_at_ping_lat", "None", "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);
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
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
