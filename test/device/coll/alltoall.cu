/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "alltoall.cubin"

#include "utils.h"
#include "coll_test.h"
#include "test_teams.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "alltoall_common.h"
#include <inttypes.h>

#define DO_ALLTOALL_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                          \
    void *args_##TYPENAME##_##SC_SUFFIX[] = {(void *)&team, (void *)&d_dest, (void *)&d_source,   \
                                             (void *)&nelems, (void *)&iters,                     \
                                             (void *)&_dynamic_smem_size};                        \
    CUfunction test_##TYPENAME##_alltoall##SC_SUFFIX_cubin;                                       \
    init_test_case_kernel(&test_##TYPENAME##_alltoall##SC_SUFFIX_cubin,                           \
                          NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_alltoall##SC_SUFFIX));        \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_alltoall##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, \
                            1, _dynamic_smem_size, cstrm, args_##TYPENAME##_##SC_SUFFIX, NULL));

#define DO_ALLTOALLMEM_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX)                                      \
    void *args_allmem_##SC_SUFFIX[] = {(void *)&team, (void *)&d_dest, (void *)&d_source,        \
                                       (void *)&nelems, (void *)&iters,                          \
                                       (void *)&_dynamic_smem_size};                             \
    CUfunction test_allmem_##SC_SUFFIX_cubin;                                                    \
    init_test_case_kernel(&test_allmem_##SC_SUFFIX_cubin,                                        \
                          NVSHMEMI_TEST_STRINGIFY(test_alltoallmem##SC_SUFFIX));                 \
    CU_CHECK(cuLaunchKernel(test_allmem_##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, 1,           \
                            _dynamic_smem_size, cstrm, args_allmem_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DECL_ALLTOALL_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                \
    __global__ void test_##TYPENAME##_alltoall##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                          TYPE *source, size_t nelems, int iters, \
                                                          size_t dynamic_smem_size);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DECL_ALLTOALL_TEST_KERNEL, thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DECL_ALLTOALL_TEST_KERNEL, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DECL_ALLTOALL_TEST_KERNEL, block, _block, x)
#undef DECL_ALLTOALL_TEST_KERNEL

#define DECL_ALLTOALLMEM_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX)                                 \
    __global__ void test_alltoallmem##SC_SUFFIX(nvshmem_team_t team, void *dest, void *source, \
                                                size_t nelems, int iters,                      \
                                                size_t dynamic_smem_size);
DECL_ALLTOALLMEM_TEST_KERNEL(thread, , )
DECL_ALLTOALLMEM_TEST_KERNEL(warp, _warp, x)
DECL_ALLTOALLMEM_TEST_KERNEL(block, _block, x)
#undef DECL_ALLTOALLMEM_TEST_KERNEL

#define DEFN_ALLTOALL_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                        \
    __global__ void test_##TYPENAME##_alltoall##SC_SUFFIX(                                         \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems, int iters,                   \
        size_t dynamic_smem_size) {                                                                \
        int iter;                                                                                  \
        int PE_size = nvshmem_team_n_pes(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                 \
        int groupSize = nvshmtest_##SC##_size();                                                   \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        init_##TYPENAME##_alltoall_data##SC_SUFFIX(team, source, nelems);                          \
                                                                                                   \
        for (iter = 0; iter < iters; iter++) {                                                     \
            reset_##TYPENAME##_alltoall_data##SC_SUFFIX(team, dest, nelems);                       \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team); /* To ensure dest is not overwritten */ \
            nvshmem##SC_PREFIX##_##TYPENAME##_alltoall##SC_SUFFIX(team, dest, source, nelems);     \
            nvshmtest_##SC##_sync();                                                               \
            validate_##TYPENAME##_alltoall_data##SC_SUFFIX(team, dest, nelems);                    \
        }                                                                                          \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_ALLTOALL_TEST_KERNEL, thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_ALLTOALL_TEST_KERNEL, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_ALLTOALL_TEST_KERNEL, block, _block, x)
#undef DEFN_ALLTOALL_TEST_KERNEL

#define DEFN_ALLTOALLMEM_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX)                                     \
    __global__ void test_alltoallmem##SC_SUFFIX(nvshmem_team_t team, void *dest, void *source,     \
                                                size_t nelems, int iters,                          \
                                                size_t dynamic_smem_size) {                        \
        int iter;                                                                                  \
        int PE_size = nvshmem_team_n_pes(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                 \
        int groupSize = nvshmtest_##SC##_size();                                                   \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        init_char_alltoall_data##SC_SUFFIX(team, (char *)source, nelems);                          \
                                                                                                   \
        for (iter = 0; iter < iters; iter++) {                                                     \
            reset_char_alltoall_data##SC_SUFFIX(team, (char *)dest, nelems);                       \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team); /* To ensure dest is not overwritten */ \
            nvshmem##SC_PREFIX##_alltoallmem##SC_SUFFIX(team, (char *)dest, (char *)source,        \
                                                        nelems);                                   \
            nvshmtest_##SC##_sync();                                                               \
            validate_char_alltoall_data##SC_SUFFIX(team, (char *)dest, nelems);                    \
        }                                                                                          \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

DEFN_ALLTOALLMEM_TEST_KERNEL(thread, , )
DEFN_ALLTOALLMEM_TEST_KERNEL(warp, _warp, x)
DEFN_ALLTOALLMEM_TEST_KERNEL(block, _block, x)
#undef DEFN_ALLTOALLMEM_TEST_KERNEL

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DO_ALLTOALL_TEST(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)           \
    if (use_cubin) {                                                         \
        DO_ALLTOALL_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE);    \
    } else {                                                                 \
        test_##TYPENAME##_alltoall##SC_SUFFIX<<<1, num_threads, _dynamic_smem_size, cstrm>>>( \
            team, (TYPE *)d_dest, (TYPE *)d_source, nelems, iters, _dynamic_smem_size);      \
    }                                                                        \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                  \
    cudaStreamSynchronize(cstrm);

#define DO_ALLTOALLMEM_TEST(SC, SC_SUFFIX, SC_PREFIX)                                             \
    if (use_cubin) {                                                                              \
        DO_ALLTOALLMEM_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX);                                      \
    } else {                                                                                      \
        test_alltoallmem##SC_SUFFIX<<<1, num_threads, _dynamic_smem_size, cstrm>>>(               \
            team, d_dest, d_source, nelems, iters, _dynamic_smem_size);                           \
    }                                                                                             \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                                       \
    cudaStreamSynchronize(cstrm);

int main(int argc, char **argv) {
    int status = 0;
    int num_threads;
    size_t alloc_size = 0;
    LARGEST_DT *d_buffer, *d_source, *d_dest;
    size_t nelems;
    char size_string[100];
    cudaStream_t cstrm;
    unsigned long long int errs;
    int iters;
#ifdef _NVSHMEM_DEBUG
    int mype;
#endif
    read_args(argc, argv);
    alloc_size = MAX_ELEMS * MAX_NPES * 2 * sizeof(LARGEST_DT);
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

#ifdef _NVSHMEM_DEBUG
    int npes;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
#endif

    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    alloc_size = MAX_ELEMS * MAX_NPES * 2 * sizeof(LARGEST_DT);
    if (use_mmap) {
        d_buffer = (LARGEST_DT *)allocate_mmap_buffer(alloc_size, _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    }
    if (!d_buffer) {
        ERROR_PRINT("nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = d_buffer;
    d_dest = (LARGEST_DT *)d_source + (MAX_ELEMS * MAX_NPES);

    init_test_teams();
    nvshmem_team_t team;
    iters = 1;
    /* Test all API for all teams but with single iteration and message size */
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));
            /* threaded */
            num_threads = 1;
            nelems = 1024;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_ALLTOALL_TEST, thread, , )
            DO_ALLTOALLMEM_TEST(thread, , )

            DEBUG_PRINT("%d: team: %s alltoall thread passed\n", mype,
                        map_team_to_string[team].c_str());

            /* warp */
            num_threads = 32;
            nelems = ELEMS_PER_THREAD * num_threads;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_ALLTOALL_TEST, warp, _warp, x)
            DO_ALLTOALLMEM_TEST(warp, _warp, x)

            DEBUG_PRINT("%d: team: %s alltoall warp passed\n", mype,
                        map_team_to_string[team].c_str());

            /* block */
            num_threads = NVSHM_TEST_NUM_TPB;
            nelems = ELEMS_PER_THREAD * num_threads;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_ALLTOALL_TEST, block, _block, x)
            DO_ALLTOALLMEM_TEST(block, _block, x)

            DEBUG_PRINT("%d: team: %s alltoall block passed\n", mype,
                        map_team_to_string[team].c_str());

            COLL_CHECK_ERRS_D();
            nvshmem_barrier_all();
        } else {
            nvshmem_barrier_all();
        }
    }

    /* Test different message sizes and iterations for a single API */
    iters = 50;
    num_threads = 1;
    team = NVSHMEM_TEAM_WORLD;
    for (nelems = 1; nelems < MAX_ELEMS; nelems *= 2) {
        DO_ALLTOALL_TEST(thread, , , float, float)
    }
    COLL_CHECK_ERRS_D();

    finalize_test_teams();

    nvshmem_barrier_all();
    if (use_mmap) {
        free_mmap_buffer((void *)d_buffer);
    } else {
        nvshmem_free((void *)d_buffer);
    }
    CUDA_RUNTIME_CHECK(cudaStreamDestroy(cstrm));
    finalize_wrapper();

out:
    return (status) ? status : errs;
}
