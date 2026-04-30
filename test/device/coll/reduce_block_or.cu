/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "reduce_block_or.cubin"

#include "utils.h"
#include "coll_test.h"
#include <string>
#include <iostream>
#include <sstream>
#include "test_teams.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "reduce_common.h"

using namespace std;

#define DO_RDXN_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                         \
    void *args_##TYPENAME##_##SC_SUFFIX[] = {(void *)&team, (void *)&dest, (void *)&source,      \
                                             (void *)&nelems, (void *)&_dynamic_smem_size};      \
    CUfunction test_##TYPENAME##_rdxn##SC_SUFFIX_cubin;                                          \
    init_test_case_kernel(                                                                       \
        &test_##TYPENAME##_rdxn##SC_SUFFIX_cubin,                                                \
        NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_##OP##_reduce_kernel##SC_SUFFIX));             \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_rdxn##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, 1, \
                            _dynamic_smem_size, cstrm, args_##TYPENAME##_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DECL_TYPENAME_OP_REDUCE(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __global__ void test_##TYPENAME##_##OP##_reduce_kernel##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems, size_t dynamic_smem_size);
NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DECL_TYPENAME_OP_REDUCE, block, _block, x, or)
#undef DECL_TYPENAME_OP_REDUCE

#define DEFN_TYPENAME_OP_REDUCE(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)              \
    __global__ void test_##TYPENAME##_##OP##_reduce_kernel##SC_SUFFIX(                     \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems,                      \
        size_t dynamic_smem_size) {                                                        \
        int myIdx = nvshmtest_thread_id_in_##SC();                                         \
        int groupSize = nvshmtest_##SC##_size();                                           \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                         \
        init_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(team, source, nelems);             \
                                                                                           \
        for (int j = 0; j < 1 /*MAX_ITER*/; j++) {                                         \
            reset_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(team, dest, nelems);          \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                 \
            nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reduce##SC_SUFFIX(team, dest, source, \
                                                                       nelems);            \
            validate_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(team, dest, nelems);       \
        }                                                                                  \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                      \
    }

NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DEFN_TYPENAME_OP_REDUCE, block, _block, x, or)
#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DO_RDXN_TEST(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                                \
    if (use_cubin) {                                                                                      \
        DO_RDXN_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP);                                 \
    } else {                                                                                              \
        test_##TYPENAME##_##OP##_reduce_kernel##SC_SUFFIX<<<1, num_threads, _dynamic_smem_size, cstrm>>>( \
            team, (TYPE *)dest, (TYPE *)source, nelems, _dynamic_smem_size);                              \
    }                                                                                                     \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                                               \
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(cstrm));

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    unsigned num_threads;
    size_t alloc_size = MAX_ELEMS * 2 * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    cudaStream_t cstrm;
    size_t nelems;
    char size_string[100];
    unsigned long long int errs;

    read_args(argc, argv);
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }

    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();
    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
    if (use_mmap) {
        d_buffer =
            (LARGEST_DT *)allocate_mmap_buffer(alloc_size * 1, _mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_calloc(alloc_size, 1);
    }
    if (!d_buffer) {
        ERROR_PRINT("nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    source = d_buffer;
    dest = &source[MAX_ELEMS];

    nvshmem_barrier_all();

    init_test_teams();
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            printf("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                   map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                   nvshmem_team_n_pes(team));
            num_threads = NVSHM_TEST_NUM_TPB;
            for (nelems = 1; nelems <= ELEMS_PER_THREAD * num_threads; nelems *= 2) {
                DEBUG_PRINT("thread version, num elements = %ld\n", nelems);
                NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DO_RDXN_TEST, block, _block,
                                                                      x, or)
            }

            COLL_CHECK_ERRS_D();
            nvshmem_barrier_all();
        } else {
            nvshmem_barrier_all();
        }
    }

    finalize_test_teams();
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }
    CUDA_RUNTIME_CHECK(cudaStreamDestroy(cstrm));
    finalize_wrapper();

out:
    return errs;
}
