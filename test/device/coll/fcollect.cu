/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUMODULE_NAME "fcollect.cubin"

#include "utils.h"
#include "coll_test.h"
#include "test_teams.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "fcollect_common.h"

#define DO_FCOLLECT_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                          \
    void *args_##TYPENAME##_##SC_SUFFIX[] = {(void *)&team, (void *)&dest, (void *)&source,       \
                                             (void *)&nelems, (void *)&iters};                    \
    CUfunction test_##TYPENAME##_fcollect##SC_SUFFIX_cubin;                                       \
    init_test_case_kernel(&test_##TYPENAME##_fcollect##SC_SUFFIX_cubin,                           \
                          NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_fcollect##SC_SUFFIX));        \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_fcollect##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, \
                            1, 0, cstrm, args_##TYPENAME##_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DECL_FCOLLECT_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                \
    __global__ void test_##TYPENAME##_fcollect##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                          TYPE *source, size_t nelems, int iters);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_FCOLLECT_TEST_KERNEL)
#undef DECL_FCOLLECT_TEST_KERNEL

#define DEFN_FCOLLECT_TEST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                    \
    __global__ void test_##TYPENAME##_fcollect##SC_SUFFIX(                                     \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems, int iters) {             \
        int iter;                                                                              \
        int PE_size = nvshmem_team_n_pes(team);                                                \
        int myIdx = nvshmtest_thread_id_in_##SC();                                             \
        int groupSize = nvshmtest_##SC##_size();                                               \
                                                                                               \
        init_##TYPENAME##_fcollect_data##SC_SUFFIX(team, source, nelems);                      \
        for (iter = 0; iter < iters; iter++) {                                                 \
            reset_##TYPENAME##_fcollect_data##SC_SUFFIX(team, dest, nelems);                   \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                     \
            nvshmem##SC_PREFIX##_##TYPENAME##_fcollect##SC_SUFFIX(team, dest, source, nelems); \
            validate_##TYPENAME##_fcollect_data##SC_SUFFIX(team, dest, nelems);                \
        }                                                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DEFN_FCOLLECT_TEST_KERNEL)
#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif
#undef DEFN_FCOLLECT_TEST_KERNEL

#define DO_FCOLLECT_TEST(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)           \
    if (use_cubin) {                                                         \
        DO_FCOLLECT_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE);    \
    } else {                                                                 \
        test_##TYPENAME##_fcollect##SC_SUFFIX<<<1, num_threads, 0, cstrm>>>( \
            team, (TYPE *)dest, (TYPE *)source, nelems, iters);              \
    }                                                                        \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                  \
    cudaStreamSynchronize(cstrm);

int main(int argc, char **argv) {
    int status = 0;
    int num_threads;
    size_t alloc_size = 0;
    LARGEST_DT *d_buffer, *source, *dest;
    size_t nelems;
    char size_string[100];
    cudaStream_t cstrm;
    unsigned long long int errs;
    int iters;
#ifdef _NVSHMEM_DEBUG
    int mype;
#endif
    read_args(argc, argv);
    alloc_size = MAX_ELEMS * (MAX_NPES + 1) * sizeof(LARGEST_DT);

    sprintf(size_string, "%zu", alloc_size);
    // status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
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

    if (use_mmap) {
        d_buffer = (LARGEST_DT *)allocate_mmap_buffer(alloc_size, _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    }
    if (!d_buffer) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }

    source = d_buffer;
    dest = (LARGEST_DT *)source + MAX_ELEMS;

    init_test_teams();
    nvshmem_team_t team;
    iters = 1;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));

            /* threaded */
            num_threads = 1;
            nelems = 1024;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_FCOLLECT_TEST, thread, , )

            DEBUG_PRINT("%d: collect thread passed\n", mype);

            /* warp */
            num_threads = 32;
            nelems = ELEMS_PER_THREAD * num_threads;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_FCOLLECT_TEST, warp, _warp, x)

            DEBUG_PRINT("%d: collect warp passed\n", mype);

            /* block */
            num_threads = NVSHM_TEST_NUM_TPB;
            nelems = ELEMS_PER_THREAD * num_threads;
            NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_FCOLLECT_TEST, block, _block, x)

            DEBUG_PRINT("%d: fcollect block passed\n", mype);
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
        DO_FCOLLECT_TEST(thread, , , float, float)
    }
    COLL_CHECK_ERRS_D();

    finalize_test_teams();
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }
    CUDA_RUNTIME_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return (status) ? status : errs;
}
