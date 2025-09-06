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

#include "coll_test.h"
#define LARGEST_DT uint64_t

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    read_args(argc, argv);
    size_t alloc_size;
    size_t min_elems, max_elems;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    char size_string[100];
    uint64_t *size_array = (uint64_t *)calloc(max_size_log, sizeof(uint64_t));
    double **latency_array = (double **)malloc(max_size_log * sizeof(double *));
    cudaStream_t stream;

    for (int i = 0; i < max_size_log; i++) {
        latency_array[i] = (double *)calloc(iters, sizeof(double));
    }

    DEBUG_PRINT("symmetric size requested %lu\n", max_size * 2);
    sprintf(size_string, "%lu", max_size * 2);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreate(&stream));

    alloc_size = max_size * 2;

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (LARGEST_DT *)h_buffer;
    h_dest = (LARGEST_DT *)&h_source[max_size / sizeof(LARGEST_DT)];
    if (use_mmap) {
        d_buffer = (LARGEST_DT *)allocate_mmap_buffer(alloc_size, mem_handle_type, use_egm);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
    }
    if (!d_buffer) {
        fprintf(stderr, "Buffer allocation failed \n");
        status = -1;
        goto out;
    }

    d_source = (LARGEST_DT *)d_buffer;
    d_dest = (LARGEST_DT *)&d_source[max_size / sizeof(LARGEST_DT)];

#define CALL_RUN_RDXN_DATATYPE(TYPENAME, TYPE)                                      \
    RUN_RDXN_DATATYPE(reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, (TYPE *)d_source, \
                      (TYPE *)d_dest, num_elems, stream, size_array, latency_array);
#define CALL_RUN_RDXN_BITWISE_DATATYPE(TYPENAME, TYPE)                                      \
    RUN_RDXN_BITWISE_DATATYPE(reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, (TYPE *)d_source, \
                              (TYPE *)d_dest, num_elems, stream, size_array, latency_array);

    switch (datatype.type) {
        case NVSHMEM_INT:
            CALL_RUN_RDXN_DATATYPE(int, int);
            break;
        case NVSHMEM_LONG:
            CALL_RUN_RDXN_DATATYPE(long, long);
            break;
        case NVSHMEM_LONGLONG:
            CALL_RUN_RDXN_DATATYPE(longlong, long long);
            break;
        case NVSHMEM_ULONGLONG:
            CALL_RUN_RDXN_DATATYPE(ulonglong, unsigned long long);
            break;
        case NVSHMEM_SIZE:
            CALL_RUN_RDXN_BITWISE_DATATYPE(size, size_t);
            break;
        case NVSHMEM_FLOAT:
            CALL_RUN_RDXN_DATATYPE(float, float);
            break;
        case NVSHMEM_DOUBLE:
            CALL_RUN_RDXN_DATATYPE(double, double);
            break;
        case NVSHMEM_UINT:
            CALL_RUN_RDXN_BITWISE_DATATYPE(uint, unsigned int);
            break;
        case NVSHMEM_INT32:
            CALL_RUN_RDXN_BITWISE_DATATYPE(int32, int32_t);
            break;
        case NVSHMEM_INT64:
            CALL_RUN_RDXN_BITWISE_DATATYPE(int64, int64_t);
            break;
        case NVSHMEM_UINT32:
            CALL_RUN_RDXN_BITWISE_DATATYPE(uint32, uint32_t);
            break;
        case NVSHMEM_UINT64:
            CALL_RUN_RDXN_BITWISE_DATATYPE(uint64, uint64_t);
            break;
        case NVSHMEM_FP16:
            CALL_RUN_RDXN_DATATYPE(half, half);
            break;
        case NVSHMEM_BF16:
            CALL_RUN_RDXN_DATATYPE(bfloat16, __nv_bfloat16);
            break;
        default:
            printf("Incorrect datatype specified\n");
            exit(1);
            break;
    }
    if (!mype) {
        print_table_v2("reduction_on_stream", (datatype.name + "-" + reduce_op.name).c_str(),
                       "size (Bytes)", "latency", "us", '-', size_array, latency_array,
                       max_size_log, iters);
    }

    nvshmem_barrier_all();
    CUDA_CHECK(cudaFreeHost(h_buffer));
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
