/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "reducescatter_latency.cubin"

#include "coll_test.h"
#define LARGEST_DT int64_t

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define CALL_RDXN(TG_PRE, TG, TYPENAME, TYPE, OP, THREAD_COMP, ELEM_COMP)                     \
                                                                                              \
    void call_test_##TYPENAME##_##OP##_reducescatter_kern##TG##_cubin(                        \
        int num_blocks, int num_tpb, cudaStream_t stream, void **arglist) {                   \
        CUfunction test_##TYPENAME##_##OP##_reducescatter_kern##TG_cubin;                     \
                                                                                              \
        init_test_case_kernel(                                                                \
            &test_##TYPENAME##_##OP##_reducescatter_kern##TG_cubin,                           \
            NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_##OP##_reducescatter_kern##TG));        \
        size_t dynamic_smem_size = *reinterpret_cast<size_t *>(arglist[5]);                   \
        NVSHMEM_PERF_CU_LAUNCH_COOP(test_##TYPENAME##_##OP##_reducescatter_kern##TG_cubin,    \
                                    num_blocks, num_tpb, stream, arglist, dynamic_smem_size); \
    }                                                                                         \
                                                                                              \
    __global__ void test_##TYPENAME##_##OP##_reducescatter_kern##TG(                          \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int iter,            \
        size_t dynamic_smem_size) {                                                           \
        int i;                                                                                \
                                                                                              \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                            \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {             \
            for (i = 0; i < iter; i++) {                                                      \
                nvshmem##TG_PRE##_##TYPENAME##_##OP##_reducescatter##TG(team, dest, source,   \
                                                                        nelems);              \
            }                                                                                 \
        }                                                                                     \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                         \
    }

#define CALL_RDXN_KERNEL(TYPENAME, OP, TG, BLOCKS, THREADS, ARG_LIST, STREAM)                   \
    if (use_cubin) {                                                                            \
        call_test_##TYPENAME##_##OP##_reducescatter_kern##TG##_cubin(BLOCKS, THREADS, STREAM,   \
                                                                     ARG_LIST);                 \
    } else {                                                                                    \
        size_t dynamic_smem_size = *reinterpret_cast<size_t *>((ARG_LIST)[5]);                  \
        NVSHMEM_PERF_COLLECTIVE_LAUNCH(status, test_##TYPENAME##_##OP##_reducescatter_kern##TG, \
                                       BLOCKS, THREADS, ARG_LIST, dynamic_smem_size, STREAM);   \
    }

#define CALL_RDXN_OPS_ALL_TG(TYPENAME, TYPE)                     \
    CALL_RDXN(x, _block, TYPENAME, TYPE, sum, INT_MAX, INT_MAX)  \
    CALL_RDXN(x, _block, TYPENAME, TYPE, prod, INT_MAX, INT_MAX) \
    CALL_RDXN(x, _block, TYPENAME, TYPE, and, INT_MAX, INT_MAX)  \
    CALL_RDXN(x, _block, TYPENAME, TYPE, or, INT_MAX, INT_MAX)   \
    CALL_RDXN(x, _block, TYPENAME, TYPE, xor, INT_MAX, INT_MAX)  \
    CALL_RDXN(x, _block, TYPENAME, TYPE, min, INT_MAX, INT_MAX)  \
    CALL_RDXN(x, _block, TYPENAME, TYPE, max, INT_MAX, INT_MAX)  \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, sum, warpSize, 4096)     \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, prod, warpSize, 4096)    \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, and, warpSize, 4096)     \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, or, warpSize, 4096)      \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, xor, warpSize, 4096)     \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, min, warpSize, 4096)     \
    CALL_RDXN(x, _warp, TYPENAME, TYPE, max, warpSize, 4096)     \
    CALL_RDXN(, , TYPENAME, TYPE, sum, 1, 512)                   \
    CALL_RDXN(, , TYPENAME, TYPE, prod, 1, 512)                  \
    CALL_RDXN(, , TYPENAME, TYPE, and, 1, 512)                   \
    CALL_RDXN(, , TYPENAME, TYPE, or, 1, 512)                    \
    CALL_RDXN(, , TYPENAME, TYPE, xor, 1, 512)                   \
    CALL_RDXN(, , TYPENAME, TYPE, min, 1, 512)                   \
    CALL_RDXN(, , TYPENAME, TYPE, max, 1, 512)

CALL_RDXN_OPS_ALL_TG(int32, int32_t)
CALL_RDXN_OPS_ALL_TG(int64, int64_t)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define SET_SIZE_ARR(TYPE, ELEM_COMP)                                                          \
    do {                                                                                       \
        j = 0;                                                                                 \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) { \
            if (num_elems < ELEM_COMP) {                                                       \
                size_arr[j] =                                                                  \
                    calculate_collective_size("reducescatter", num_elems, sizeof(TYPE), npes); \
            } else {                                                                           \
                size_arr[j] = 0;                                                               \
            }                                                                                  \
            j++;                                                                               \
        }                                                                                      \
    } while (0)

#define RUN_ITERS_OP(TYPENAME, TYPE, GROUP, OP, ELEM_COMP)                                       \
    do {                                                                                         \
        void *skip_arg_list[] = {&team, &dest, &source, &num_elems, &skip, &dynamic_smem_size};  \
        void *time_arg_list[] = {&team, &dest, &source, &num_elems, &iter, &dynamic_smem_size};  \
        float milliseconds;                                                                      \
        cudaEvent_t start, stop;                                                                 \
        cudaEventCreate(&start);                                                                 \
        cudaEventCreate(&stop);                                                                  \
        SET_SIZE_ARR(TYPE, ELEM_COMP);                                                           \
                                                                                                 \
        nvshmem_barrier_all();                                                                   \
        j = 0;                                                                                   \
        for (num_elems = min_elems; num_elems < ELEM_COMP; num_elems *= step_factor) {           \
            CALL_RDXN_KERNEL(TYPENAME, OP, GROUP, num_blocks, nvshm_test_num_tpb, skip_arg_list, \
                             stream)                                                             \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            nvshmem_barrier_all();                                                               \
                                                                                                 \
            cudaEventRecord(start, stream);                                                      \
            CALL_RDXN_KERNEL(TYPENAME, OP, GROUP, num_blocks, nvshm_test_num_tpb, time_arg_list, \
                             stream)                                                             \
            cudaEventRecord(stop, stream);                                                       \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
                                                                                                 \
            if (!mype) {                                                                         \
                cudaEventElapsedTime(&milliseconds, start, stop);                                \
                h_##OP##_lat[j] = (milliseconds * 1000.0) / (float)iter;                         \
            }                                                                                    \
            nvshmem_barrier_all();                                                               \
            j++;                                                                                 \
        }                                                                                        \
    } while (0)

#define RUN_ITERS(TYPENAME, TYPE, GROUP, ELEM_COMP)       \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, sum, ELEM_COMP);  \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, prod, ELEM_COMP); \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, and, ELEM_COMP);  \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, or, ELEM_COMP);   \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, xor, ELEM_COMP);  \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, min, ELEM_COMP);  \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, max, ELEM_COMP);

int rdxn_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                        cudaStream_t stream, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    size_t num_elems = 1, min_elems, max_elems;
    int iter = iters;
    int skip = warmup_iters;
    size_t dynamic_smem_size = NVSHMEM_PERF_COLL_DYNAMIC_SMEM_SIZE();
    int j;
    int npes = nvshmem_n_pes();
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_sum_lat = (double *)h_tables[1];
    double *h_prod_lat = (double *)h_tables[2];
    double *h_and_lat = (double *)h_tables[3];
    double *h_or_lat = (double *)h_tables[4];
    double *h_xor_lat = (double *)h_tables[5];
    double *h_min_lat = (double *)h_tables[6];
    double *h_max_lat = (double *)h_tables[7];

    // if (!mype) printf("Transfer size in bytes and latency of thread/warp/block variants of all
    // operations of reduction API in us\n");
    if (threadgroup_scope.type == NVSHMEM_THREAD || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int32_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int32_t)));
        RUN_ITERS(int32, int32_t, , 512);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int32-sum-t", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int32-prod-t", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int32-and-t", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int32-or-t", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int32-xor-t", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int32-min-t", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int32-max-t", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }

        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int64_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int64_t)));
        RUN_ITERS(int64, int64_t, , 512);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int64-sum-t", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int64-prod-t", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int64-and-t", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int64-or-t", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int64-xor-t", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int64-min-t", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int64-max-t", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }
    }

    if (threadgroup_scope.type == NVSHMEM_WARP || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int32_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int32_t)));
        RUN_ITERS(int32, int32_t, _warp, 4096);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int32-sum-w", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int32-prod-w", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int32-and-w", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int32-or-w", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int32-xor-w", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int32-min-w", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int32-max-w", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }

        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int64_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int64_t)));
        RUN_ITERS(int64, int64_t, _warp, 4096);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int64-sum-w", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int64-prod-w", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int64-and-w", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int64-or-w", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int64-xor-w", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int64-min-w", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int64-max-w", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }
    }

    if (threadgroup_scope.type == NVSHMEM_BLOCK || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int32_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int32_t)));
        RUN_ITERS(int32, int32_t, _block, max_elems);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int32-sum-b", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int32-prod-b", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int32-and-b", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int32-or-b", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int32-xor-b", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int32-min-b", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int32-max-b", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }

        min_elems = max(static_cast<size_t>(1), min_size / (nvshmem_n_pes() * sizeof(int64_t)));
        max_elems = max(static_cast<size_t>(1), max_size / (nvshmem_n_pes() * sizeof(int64_t)));
        RUN_ITERS(int64, int64_t, _block, max_elems);
        if (!mype) {
            print_device_collective_table("device_reducescatter", "int64-sum-b", "latency", "us",
                                          '-', size_arr, h_sum_lat, j);
            print_device_collective_table("device_reducescatter", "int64-prod-b", "latency", "us",
                                          '-', size_arr, h_prod_lat, j);
            print_device_collective_table("device_reducescatter", "int64-and-b", "latency", "us",
                                          '-', size_arr, h_and_lat, j);
            print_device_collective_table("device_reducescatter", "int64-or-b", "latency", "us",
                                          '-', size_arr, h_or_lat, j);
            print_device_collective_table("device_reducescatter", "int64-xor-b", "latency", "us",
                                          '-', size_arr, h_xor_lat, j);
            print_device_collective_table("device_reducescatter", "int64-min-b", "latency", "us",
                                          '-', size_arr, h_min_lat, j);
            print_device_collective_table("device_reducescatter", "int64-max-b", "latency", "us",
                                          '-', size_arr, h_max_lat, j);
        }
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;
    size_t size = 0;

    read_args(argc, argv);

    int *d_source, *d_dest;
    char size_string[100];
    cudaStream_t cstrm;
    void **h_tables;

    size = page_size_roundoff(max_size);   // send buf
    size += page_size_roundoff(max_size);  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    array_size = max_size_log;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 8, array_size);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_source = (int32_t *)nvshmem_align(getpagesize(), max_size);
    d_dest = (int32_t *)nvshmem_align(getpagesize(), max_size / nvshmem_n_pes());

    rdxn_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, cstrm, h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    nvshmem_barrier_all();

    nvshmem_free(d_source);
    nvshmem_free(d_dest);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
