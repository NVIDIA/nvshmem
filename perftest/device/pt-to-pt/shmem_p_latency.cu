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

#define CUMODULE_NAME "shmem_p_latency.cubin"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void p_latency(int *data_d, int len, int pe, int iter) {
    int i, j, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < iter; i++) {
        for (j = tid; j < len; j += blockDim.x) {
            nvshmem_int_p(data_d + j, *(data_d + j), peer);
        }
        __syncthreads();
        if (!tid) {
            nvshmem_quiet();
        }
        __syncthreads();
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

void test_p(int *data_d, int len, int pe, int iter, CUfunction kernel) {
    if (use_cubin) {
        void *arglist[] = {(void *)&data_d, (void *)&len, (void *)&pe, (void *)&iter};
        CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, threads_per_block, 1, 1, 0, NULL, arglist, NULL));
    } else {
        p_latency<<<1, threads_per_block>>>(data_d, len, pe, iter);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    int *data_d = NULL;

    read_args(argc, argv);

    int iter = iters;
    int skip = warmup_iters;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    float milliseconds;
    cudaEvent_t start, stop;
    CUfunction test_cubin = NULL;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
        init_test_case_kernel(&test_cubin, "p_latency");
    }

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
    h_lat = (double *)h_tables[1];

    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
        if (!data_d) {
            fprintf(stderr, "buffer allocation failed \n");
            goto finalize;
        }
    } else {
        data_d = (int *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        if (!data_d) {
            fprintf(stderr, "buffer allocation failed \n");
            goto finalize;
        }
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    i = 0;
    for (size_t size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_p(data_d, nelems, mype, skip, test_cubin);
            cudaEventRecord(start);
            test_p(data_d, nelems, mype, iter, test_cubin);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            /* give latency in us */
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_basic("shmem_p_latency", "None", "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);
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
    finalize_wrapper();

    return 0;
}
