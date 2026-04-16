/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "barrier.cubin"

#include "utils.h"
#include "coll_test.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "barrier_common.h"
#include "test_teams.h"

#undef MAX_ITER
#define MAX_ITER 10

#define DO_BARRIER_ALL_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX)                                    \
    void *args_all_##SC_SUFFIX[] = {(void *)&buf};                                             \
    CUfunction test_barrier_all_##SC_SUFFIX_cubin;                                             \
    init_test_case_kernel(&test_barrier_all_##SC_SUFFIX_cubin,                                 \
                          NVSHMEMI_TEST_STRINGIFY(test_barrier_all##SC_SUFFIX##_kernel));      \
    CU_CHECK(cuLaunchKernel(test_barrier_all_##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, 1, 0, \
                            cstrm, args_all_##SC_SUFFIX, NULL));

#define DO_BARRIER_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX)                                           \
    void *args_##SC_SUFFIX[] = {(void *)&team, (void *)&buf};                                     \
    CUfunction test_barrier_##SC_SUFFIX_cubin;                                                    \
    init_test_case_kernel(&test_barrier_##SC_SUFFIX_cubin,                                        \
                          NVSHMEMI_TEST_STRINGIFY(test_barrier##SC_SUFFIX##_kernel));             \
    CU_CHECK(cuLaunchKernel(test_barrier_##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, 1, 0, cstrm, \
                            args_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DECL_TEST_BARRIER_ALL_KERNEL(SC, SC_SUFFIX, SC_PREFIX) \
    __global__ void test_barrier_all##SC_SUFFIX##_kernel(int *buf);
NVSHMEMI_REPT_FOR_SCOPES2(DECL_TEST_BARRIER_ALL_KERNEL)
#undef DECL_TEST_BARRIER_ALL_KERNEL

#define DECL_TEST_BARRIER_KERNEL(SC, SC_SUFFIX, SC_PREFIX) \
    __global__ void test_barrier##SC_SUFFIX##_kernel(nvshmem_team_t team, int *buf);
NVSHMEMI_REPT_FOR_SCOPES2(DECL_TEST_BARRIER_KERNEL)
#undef DECL_TEST_BARRIER_KERNEL

#define TEST_BARRIER_ALL_KERNEL(SC, SC_SUFFIX, SC_PREFIX)              \
    __global__ void test_barrier_all##SC_SUFFIX##_kernel(int *buf) {   \
        for (int iters = 0; iters < MAX_ITER; iters++) {               \
            init_barrier_data##SC_SUFFIX(NVSHMEM_TEAM_WORLD, buf);     \
            nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();             \
            validate_barrier_data##SC_SUFFIX(NVSHMEM_TEAM_WORLD, buf); \
            reset_barrier_data##SC_SUFFIX(NVSHMEM_TEAM_WORLD, buf);    \
            nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();             \
        }                                                              \
    }
NVSHMEMI_REPT_FOR_SCOPES2(TEST_BARRIER_ALL_KERNEL)

#define TEST_BARRIER_KERNEL(SC, SC_SUFFIX, SC_PREFIX)                                 \
    __global__ void test_barrier##SC_SUFFIX##_kernel(nvshmem_team_t team, int *buf) { \
        for (int iters = 0; iters < MAX_ITER; iters++) {                              \
            init_barrier_data##SC_SUFFIX(team, buf);                                  \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                            \
            validate_barrier_data##SC_SUFFIX(team, buf);                              \
            reset_barrier_data##SC_SUFFIX(team, buf);                                 \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                            \
        }                                                                             \
    }
NVSHMEMI_REPT_FOR_SCOPES2(TEST_BARRIER_KERNEL)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_BARRIER_ALL(SC, SC_SUFFIX, SC_PREFIX)                               \
    if (use_cubin) {                                                             \
        DO_BARRIER_ALL_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX);                     \
    } else {                                                                     \
        test_barrier_all##SC_SUFFIX##_kernel<<<1, num_threads, 0, cstrm>>>(buf); \
    }                                                                            \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                      \
    cudaStreamSynchronize(cstrm);

#define TEST_BARRIER(SC, SC_SUFFIX, SC_PREFIX)                                     \
    if (use_cubin) {                                                               \
        DO_BARRIER_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX);                           \
    } else {                                                                       \
        test_barrier##SC_SUFFIX##_kernel<<<1, num_threads, 0, cstrm>>>(team, buf); \
    }                                                                              \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                        \
    cudaStreamSynchronize(cstrm);

int main(int argc, char **argv) {
    int status = 0;
    int n_pes, num_threads;
    int i = 0;
    cudaStream_t cstrm;
    int *buf;
    unsigned long long int errs;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

#ifdef _NVSHMEM_DEBUG
    int mype = nvshmem_my_pe();
#endif
    n_pes = nvshmem_n_pes();
    if (use_mmap) {
        buf = (int *)allocate_mmap_buffer(sizeof(int) * n_pes, _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        buf = (int *)nvshmem_malloc(sizeof(int) * n_pes);
    }

    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(cstrm));

    init_test_teams();
    nvshmem_barrier_all();

    DEBUG_PRINT("barrier/barrier_all tests\n");
    // sleep(20);
    /* threaded */
    num_threads = 1;
    for (i = 0; i < MAX_ITER; i++) {
        TEST_BARRIER_ALL(thread, , );
    }
    DEBUG_PRINT("%d: barrier_all thread passed\n", mype);

    /* warp */
    num_threads = 32;
    for (i = 0; i < MAX_ITER; i++) {
        TEST_BARRIER_ALL(warp, _warp, x);
    }
    DEBUG_PRINT("%d: barrier_all warp passed\n", mype);

    /* block */
    num_threads = NVSHM_TEST_NUM_TPB;
    for (i = 0; i < MAX_ITER; i++) {
        TEST_BARRIER_ALL(block, _block, x);
    }
    DEBUG_PRINT("%d: barrier_all block passed\n", mype);
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (team_id = %d, my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), team, nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));

            /* thread */
            num_threads = 1;
            for (i = 0; i < MAX_ITER; i++) {
                TEST_BARRIER(thread, , );
            }
            DEBUG_PRINT("%d: barrier thread passed\n", mype);

            /* warp */
            num_threads = 32;
            for (i = 0; i < MAX_ITER; i++) {
                TEST_BARRIER(warp, _warp, x);
            }
            DEBUG_PRINT("%d: barrier warp passed\n", mype);

            /* block */
            num_threads = NVSHM_TEST_NUM_TPB;
            for (i = 0; i < MAX_ITER; i++) {
                TEST_BARRIER(block, _block, x);
            }
            DEBUG_PRINT("%d: barrier block passed\n", mype);

            COLL_CHECK_ERRS_D();
            nvshmem_barrier_all();
        } else {
            nvshmem_barrier_all();
        }
    }

    DEBUG_PRINT("%d: passed\n", mype);

    finalize_test_teams();
    CUDA_RUNTIME_CHECK(cudaStreamDestroy(cstrm));
    nvshmem_barrier_all();
    if (use_mmap) {
        free_mmap_buffer(buf);
    } else {
        nvshmem_free(buf);
    }
    finalize_wrapper();

out:
    return (status) ? status : errs;
}
