/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "coll_test.h"
#define LARGEST_DT double2

constexpr int REDMAXLOC_NUM_ELEMS = 1;
constexpr int REDMAXLOC_MAX_ELEMS_EXCLUSIVE = REDMAXLOC_NUM_ELEMS + 1;

#define CALL_RDXN(TG_PRE, TG, TYPENAME, TYPE, OP, THREAD_COMP, ELEM_COMP)                          \
    __global__ void test_##TYPENAME##_##OP##_reduce_kern##TG(nvshmem_team_t team, TYPE *dest,      \
                                                             const TYPE *source, int nelems,       \
                                                             int iter, size_t dynamic_smem_size) { \
        int i;                                                                                     \
                                                                                                   \
        NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);                                                 \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                  \
            for (i = 0; i < iter; i++) {                                                           \
                nvshmem##TG_PRE##_##TYPENAME##_##OP##_reduce##TG(team, dest, source, nelems);      \
            }                                                                                      \
        }                                                                                          \
        NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

#define CALL_RDXN_OPS_ALL_TG(TYPENAME, TYPE) \
    CALL_RDXN(x, _block, TYPENAME, TYPE, maxloc, INT_MAX, REDMAXLOC_MAX_ELEMS_EXCLUSIVE)

CALL_RDXN_OPS_ALL_TG(double2, double2)

#define SET_SIZE_ARR(TYPE, ELEM_COMP)                                                      \
    do {                                                                                   \
        j = 0;                                                                             \
        for (num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {    \
            if (num_elems < ELEM_COMP) {                                                   \
                size_arr[j] =                                                              \
                    calculate_collective_size("redmaxloc", num_elems, sizeof(TYPE), npes); \
            } else {                                                                       \
                size_arr[j] = 0;                                                           \
            }                                                                              \
            j++;                                                                           \
        }                                                                                  \
    } while (0)

#define RUN_ITERS_OP(TYPENAME, TYPE, GROUP, OP, ELEM_COMP)                                      \
    do {                                                                                        \
        void *skip_arg_list[] = {&team, &dest, &source, &num_elems, &skip, &dynamic_smem_size}; \
        void *time_arg_list[] = {&team, &dest, &source, &num_elems, &iter, &dynamic_smem_size}; \
        float milliseconds;                                                                     \
        cudaEvent_t start, stop;                                                                \
        cudaEventCreate(&start);                                                                \
        cudaEventCreate(&stop);                                                                 \
        SET_SIZE_ARR(TYPE, ELEM_COMP);                                                          \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
        j = 0;                                                                                  \
        for (num_elems = min_elems; num_elems < ELEM_COMP; num_elems *= step_factor) {          \
            NVSHMEM_PERF_COLLECTIVE_LAUNCH(status, test_##TYPENAME##_##OP##_reduce_kern##GROUP, \
                                           num_blocks, nvshm_test_num_tpb, skip_arg_list,       \
                                           dynamic_smem_size, stream);                          \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                          \
            nvshmem_barrier_all();                                                              \
                                                                                                \
            cudaEventRecord(start, stream);                                                     \
            NVSHMEM_PERF_COLLECTIVE_LAUNCH(status, test_##TYPENAME##_##OP##_reduce_kern##GROUP, \
                                           num_blocks, nvshm_test_num_tpb, time_arg_list,       \
                                           dynamic_smem_size, stream);                          \
            cudaEventRecord(stop, stream);                                                      \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                          \
                                                                                                \
            if (!mype) {                                                                        \
                cudaEventElapsedTime(&milliseconds, start, stop);                               \
                h_##OP##_lat[j] = (milliseconds * 1000.0) / (float)iter;                        \
            }                                                                                   \
            nvshmem_barrier_all();                                                              \
            j++;                                                                                \
        }                                                                                       \
    } while (0)

#define RUN_ITERS(TYPENAME, TYPE, GROUP, ELEM_COMP) \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, maxloc, ELEM_COMP);

int rdxn_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                        cudaStream_t stream, run_opt_t run_options, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int num_blocks = 1;
    int num_elems = REDMAXLOC_NUM_ELEMS;
    int min_elems = REDMAXLOC_NUM_ELEMS;
    int max_elems = REDMAXLOC_MAX_ELEMS_EXCLUSIVE;
    int iter = iters;
    int skip = warmup_iters;
    size_t dynamic_smem_size = NVSHMEM_PERF_COLL_DYNAMIC_SMEM_SIZE();
    int j;
    int npes = nvshmem_n_pes();
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_maxloc_lat = (double *)h_tables[1];

    if (run_options.run_block) {
        RUN_ITERS(double2, double2, _block, max_elems);
        if (!mype) {
            print_device_collective_table("device_reduction", "double2-maxloc-b", "latency", "us",
                                          '-', size_arr, h_maxloc_lat, j);
        }
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;
    size_t size = 0;
    size_t alloc_size;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    char size_string[100];
    cudaStream_t cstrm;
    run_opt_t run_options;
    void **h_tables;

    PROCESS_OPTS(run_options);

    size = page_size_roundoff(REDMAXLOC_NUM_ELEMS * sizeof(LARGEST_DT));   // send buf
    size += page_size_roundoff(REDMAXLOC_NUM_ELEMS * sizeof(LARGEST_DT));  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    array_size = floor(std::log2((float)REDMAXLOC_MAX_ELEMS_EXCLUSIVE)) + 1;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 8, array_size);

    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    alloc_size = (REDMAXLOC_NUM_ELEMS * 2) * sizeof(LARGEST_DT);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = h_buffer;
    h_dest = &h_source[REDMAXLOC_NUM_ELEMS];

    d_source = (LARGEST_DT *)nvshmem_align(getpagesize(), REDMAXLOC_NUM_ELEMS * sizeof(LARGEST_DT));
    d_dest = (LARGEST_DT *)nvshmem_align(getpagesize(), REDMAXLOC_NUM_ELEMS * sizeof(LARGEST_DT));

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(LARGEST_DT) * REDMAXLOC_NUM_ELEMS),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(LARGEST_DT) * REDMAXLOC_NUM_ELEMS),
                               cudaMemcpyHostToDevice, cstrm));

    rdxn_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, cstrm, run_options, h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(LARGEST_DT) * REDMAXLOC_NUM_ELEMS),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(LARGEST_DT) * REDMAXLOC_NUM_ELEMS),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
