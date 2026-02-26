/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "shmem_ll_ping_pong_latency.cubin"
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"
#include "perf_utils.h"

#define MAX_MSG_SIZE 1 * 1024 * 1024
#define UNROLL 8

#define TEST_NVSHMEM_PP_CUBIN()                                                       \
    void *args_pp[] = {                                                               \
        (void *)&data_d, (void *)&pack_buffer_d, (void *)&nelems,  (void *)&mype,     \
        (void *)&iter,   (void *)&skip,          (void *)&hflag_d, (void *)&cur_lat}; \
    CUfunction test_pp_cubin;                                                         \
    init_test_case_kernel(&test_pp_cubin, NVSHMEMI_TEST_STRINGIFY(ping_pong));        \
    status = cuLaunchKernel(test_pp_cubin, 1, 1, 1, 1, 1, 1, 0, stream, args_pp, NULL);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void ping_pong(int *data_d, uint64_t *pack_buffer_d, int len, int pe, int iter, int skip,
                          int *hflag, double *lat_result) {
    long long int start, stop;
    double time;
    int i, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        /*if (pe) {
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_WARP>(data_d, pack_buffer_d, len, 2 * i + 1);
            __syncwarp();
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_WARP>(pack_buffer_d, data_d, len, 2 * i + 2);
            nvshmemi_uint64_put_nbi_warp(pack_buffer_d, pack_buffer_d, len, peer);
        } else {
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_WARP>(pack_buffer_d, data_d, len, 2 * i + 1);
            nvshmemi_uint64_put_nbi_warp(pack_buffer_d, pack_buffer_d, len, peer);
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_WARP>(data_d, pack_buffer_d, len, 2 * i + 2);
            __syncwarp();
        }*/
        /*if (pe) {
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_BLOCK>(data_d, pack_buffer_d, len, 2 * i + 1);
            __syncthreads();
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_BLOCK>(pack_buffer_d, data_d, len, 2 * i + 2);
            nvshmemi_uint64_put_nbi_block(pack_buffer_d, pack_buffer_d, len, peer);
        } else {
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_BLOCK>(pack_buffer_d, data_d, len, 2 * i + 1);
            nvshmemi_uint64_put_nbi_block(pack_buffer_d, pack_buffer_d, len, peer);
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_BLOCK>(data_d, pack_buffer_d, len, 2 * i + 2);
            __syncthreads();
        } */
        if (pe) {
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_THREAD>(data_d, pack_buffer_d, len,
                                                              2 * i + 1);
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_THREAD>(pack_buffer_d, data_d, len,
                                                              2 * i + 2);
            nvshmem_uint64_put_nbi(pack_buffer_d, pack_buffer_d, len, peer);
        } else {
            nvshmemi_packLL<int, NVSHMEMI_THREADGROUP_THREAD>(pack_buffer_d, data_d, len,
                                                              2 * i + 1);
            nvshmem_uint64_put_nbi(pack_buffer_d, pack_buffer_d, len, peer);
            nvshmemi_recvLL<int, NVSHMEMI_THREADGROUP_THREAD>(data_d, pack_buffer_d, len,
                                                              2 * i + 2);
        }
    }
    if (!tid) {
        stop = clock64();
        nvshmem_quiet();
        *hflag = 1;
    }

    if ((pe == 0) && !tid) {
        time = (stop - start) / iter;
        *lat_result = time * 1000 / clockrate;
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int c, char *v[]) {
    int mype, npes, size;
    uint64_t *pack_buffer_d = NULL;
    int *data_d = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;
    double *cur_lat;
    read_args(c, v);
    init_wrapper(&c, &v);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_msg_size, _mem_handle_type, use_egm, true);
        pack_buffer_d =
            (uint64_t *)allocate_mmap_buffer(2 * max_msg_size, _mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        data_d = (int *)nvshmem_malloc(max_msg_size);
        pack_buffer_d = (uint64_t *)nvshmem_malloc(2 * max_msg_size);
        CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
        CUDA_CHECK(cudaMemset(pack_buffer_d, 0, sizeof(uint64_t)));
    }

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    int *hflag, *hflag_d;
    CUDA_CHECK(cudaHostAlloc((void **)&hflag, sizeof(int), 0));
    *hflag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&hflag_d, hflag, 0));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    i = 0;
    for (size = 2 * sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems, status = 0;
        nelems = size / sizeof(int);
        h_size_arr[i] = size;
        cur_lat = &h_lat[i];
        void *args[] = {&data_d, &pack_buffer_d, &nelems, &mype, &iter, &skip, &hflag_d, &cur_lat};
        if (use_egm) {
            memset(pack_buffer_d, 0, sizeof(uint64_t));
        } else {
            CUDA_CHECK(cudaMemset(pack_buffer_d, 0, sizeof(uint64_t)));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        *hflag = 0;
        if (use_cubin) {
            TEST_NVSHMEM_PP_CUBIN();
        } else {
            status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, args, 0, stream);
        }
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }

        while (*((volatile int *)hflag) != 1)
            ;

        nvshmem_barrier_all();
        i++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        print_table_v1("shmem_put_ping_lat", "None", "size (Bytes)", "latency", "us", '-',
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
    if (pack_buffer_d) {
        if (use_mmap) {
            free_mmap_buffer(pack_buffer_d);
        } else {
            nvshmem_free(pack_buffer_d);
        }
    }
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
