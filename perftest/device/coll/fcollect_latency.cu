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
#define CUMODULE_NAME "fcollect_latency.cubin"

#include "coll_test.h"
#define DATATYPE int64_t

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define CALL_FCOLLECT(TYPENAME, TYPE, TG_PRE, THREADGROUP, THREAD_COMP, ELEM_COMP)                 \
    __global__ void test_##TYPENAME##_fcollect_call_kern##THREADGROUP(                             \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int mype, int iter) {     \
        int i;                                                                                     \
                                                                                                   \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                  \
            for (i = 0; i < iter; i++) {                                                           \
                nvshmem##TG_PRE##_##TYPENAME##_fcollect##THREADGROUP(team, dest, source, nelems);  \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    void test_##TYPENAME##_fcollect_call_kern##THREADGROUP##_cubin(                                \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist) {                        \
        CUfunction test_cubin;                                                                     \
                                                                                                   \
        init_test_case_kernel(                                                                     \
            &test_cubin,                                                                           \
            NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_fcollect_call_kern##THREADGROUP));           \
        CU_CHECK(cuLaunchCooperativeKernel(test_cubin, num_blocks, 1, 1, num_tpb, 1, 1, 0, stream, \
                                           arglist));                                              \
    }

#define CALL_FCOLLECT_KERNEL(TYPENAME, THREADGROUP, BLOCKS, THREADS, ARG_LIST, STREAM)        \
    if (use_cubin) {                                                                          \
        test_##TYPENAME##_fcollect_call_kern##THREADGROUP##_cubin(BLOCKS, THREADS, STREAM,    \
                                                                  ARG_LIST);                  \
    } else {                                                                                  \
        status = nvshmemx_collective_launch(                                                  \
            (const void *)test_##TYPENAME##_fcollect_call_kern##THREADGROUP, BLOCKS, THREADS, \
            ARG_LIST, 0, STREAM);                                                             \
        if (status != NVSHMEMX_SUCCESS) {                                                     \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                 \
            exit(-1);                                                                         \
        }                                                                                     \
    }

CALL_FCOLLECT(int32, int32_t, , , 1, 512);
CALL_FCOLLECT(int64, int64_t, , , 1, 512);
CALL_FCOLLECT(int32, int32_t, x, _warp, warpSize, 4096);
CALL_FCOLLECT(int64, int64_t, x, _warp, warpSize, 4096);
CALL_FCOLLECT(int32, int32_t, x, _block, INT_MAX, INT_MAX);
CALL_FCOLLECT(int64, int64_t, x, _block, INT_MAX, INT_MAX);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int fcollect_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                            cudaStream_t stream, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int num_blocks = 1;
    size_t num_elems = 1, min_elems, max_elems;
    int npes = nvshmem_n_pes();
    int i;
    int skip = warmup_iters;
    int iter = iters;
    uint64_t *h_size_array = (uint64_t *)h_tables[0];
    double *h_thread_lat = (double *)h_tables[1];
    double *h_warp_lat = (double *)h_tables[2];
    double *h_block_lat = (double *)h_tables[3];
    float milliseconds;
    void *args_1[] = {&team, &dest, &source, &num_elems, &mype, &skip};
    void *args_2[] = {&team, &dest, &source, &num_elems, &mype, &iter};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvshmem_barrier_all();
    min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(int32_t)));
    max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(int32_t)));
    i = 0;
    for (num_elems = min_elems; num_elems < 512; num_elems *= step_factor) {
        CALL_FCOLLECT_KERNEL(int32, , num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int32, , num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = min_elems; num_elems < 4096; num_elems *= step_factor) {
        CALL_FCOLLECT_KERNEL(int32, _warp, num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int32, _warp, num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems <= max_elems; num_elems *= step_factor) {
        h_size_array[i] = calculate_collective_size("fcollect", num_elems, sizeof(int32_t), npes);
        CALL_FCOLLECT_KERNEL(int32, _block, num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int32, _block, num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_block_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_table_v1("fcollect_device", "32-bit-thread", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_thread_lat, i);
        print_table_v1("fcollect_device", "32-bit-warp", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_warp_lat, i);
        print_table_v1("fcollect_device", "32-bit-block", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_block_lat, i);
    }

    min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(int64_t)));
    max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(int64_t)));
    i = 0;
    for (num_elems = 1; num_elems < 512; num_elems *= step_factor) {
        CALL_FCOLLECT_KERNEL(int64, , num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int64, , num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < 4096; num_elems *= step_factor) {
        CALL_FCOLLECT_KERNEL(int64, _warp, num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int64, _warp, num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < max_elems; num_elems *= step_factor) {
        h_size_array[i] = calculate_collective_size("fcollect", num_elems, sizeof(int64_t), npes);
        CALL_FCOLLECT_KERNEL(int64, _block, num_blocks, nvshm_test_num_tpb, args_1, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        CALL_FCOLLECT_KERNEL(int64, _block, num_blocks, nvshm_test_num_tpb, args_2, stream);

        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_block_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_table_v1("fcollect_device", "64-bit-thread", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_thread_lat, i);
        print_table_v1("fcollect_device", "64-bit-warp", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_warp_lat, i);
        print_table_v1("fcollect_device", "64-bit-block", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_block_lat, i);
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;

    read_args(argc, argv);

    size_t size = max_size * 2;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    char size_string[100];
    cudaStream_t cstrm;
    void **h_tables;

    array_size = max_size_log;

    if (use_mmap) {
        size = pad_up(size);
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

    CUDA_CHECK(cudaHostAlloc(&h_buffer, max_size * 2, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[max_size / sizeof(DATATYPE)];

    if (use_mmap) {
        d_buffer = (DATATYPE *)allocate_mmap_buffer(max_size * 2, mem_handle_type, use_egm);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        d_buffer = (DATATYPE *)nvshmem_malloc(max_size * 2);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
    }
    if (!d_buffer) {
        fprintf(stderr, "buffer allocation failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)d_buffer;
    d_dest = (DATATYPE *)&d_source[max_size / sizeof(DATATYPE)];

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, max_size, cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, max_size, cudaMemcpyHostToDevice, cstrm));

    fcollect_calling_kernel(NVSHMEM_TEAM_WORLD, (void *)d_dest, (void *)d_source, mype, cstrm,
                            h_tables);

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, max_size, cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, max_size, cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 4);
    finalize_wrapper();

out:
    return 0;
}
