/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "broadcast.cubin"

#include "utils.h"
#include "coll_test.h"
#include "test_teams.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "broadcast_common.h"

#define DO_BROADCAST_TEST_CUBIN(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                          \
    void *args_##TYPENAME##_##SC_SUFFIX[] = {(void *)&team, (void *)&dest, (void *)&source,        \
                                             (void *)&nelems, (void *)&_dynamic_smem_size};        \
    CUfunction test_##TYPENAME##_broadcast##SC_SUFFIX_cubin;                                       \
    init_test_case_kernel(&test_##TYPENAME##_broadcast##SC_SUFFIX_cubin,                           \
                          NVSHMEMI_TEST_STRINGIFY(test_##TYPENAME##_broadcast##SC_SUFFIX));        \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_broadcast##SC_SUFFIX_cubin, 1, 1, 1, num_threads, 1, \
                            1, _dynamic_smem_size, cstrm, args_##TYPENAME##_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DECL_TEST_BROADCAST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __global__ void test_##TYPENAME##_broadcast##SC_SUFFIX(                  \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems, size_t dynamic_smem_size);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_TEST_BROADCAST_KERNEL)

#define DEFN_TEST_BROADCAST_KERNEL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                       \
    __global__ void test_##TYPENAME##_broadcast##SC_SUFFIX(                                        \
        nvshmem_team_t team, TYPE *dest, TYPE *source, size_t nelems, size_t dynamic_smem_size) {  \
        int iters;                                                                                 \
        int PE_size = nvshmem_team_n_pes(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                 \
        int groupSize = nvshmtest_##SC##_size();                                                   \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                 \
        init_##TYPENAME##_broadcast_data##SC_SUFFIX(team, source, nelems);                         \
        for (iters = 0; iters < MAX_ITER; iters++) {                                               \
            reset_##TYPENAME##_broadcast_data##SC_SUFFIX(team, dest, nelems);                      \
            nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                         \
            nvshmem##SC_PREFIX##_##TYPENAME##_broadcast##SC_SUFFIX(team, dest, source, nelems, 0); \
            validate_##TYPENAME##_broadcast_data##SC_SUFFIX(team, dest, nelems);                   \
        }                                                                                          \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DEFN_TEST_BROADCAST_KERNEL)
#undef DEFN_TEST_BROADCAST_KERNEL

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DO_BROADCAST_TEST(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                        \
    test_##TYPENAME##_broadcast##SC_SUFFIX<<<1, num_threads, _dynamic_smem_size, cstrm>>>( \
        team, (TYPE *)dest, (TYPE *)source, nelems, _dynamic_smem_size);                   \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                                \
    cudaStreamSynchronize(cstrm);

int main(int argc, char **argv) {
    int status = 0;
    unsigned num_threads;
    size_t alloc_size = 0;
    LARGEST_DT *d_buffer, *source, *dest;
    size_t nelems;
    cudaStream_t cstrm;
    unsigned long long int errs;

    read_args(argc, argv);
    alloc_size = MAX_ELEMS * 2 * sizeof(LARGEST_DT);

    init_wrapper(&argc, &argv);

#ifdef _NVSHMEM_DEBUG
    int npes;
    int mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
#endif

    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    if (use_mmap) {
        d_buffer = (LARGEST_DT *)allocate_mmap_buffer(alloc_size * 1, _mem_handle_type, use_egm);
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

    nvshmem_barrier_all();

    init_test_teams();
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));

            /* threaded */
            num_threads = 1;
            for (nelems = 1; nelems <= ELEMS_PER_THREAD; nelems *= 2) {
                NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_BROADCAST_TEST, thread, , )
            }
            DEBUG_PRINT("%d: broadcast thread passed\n", mype);
            /* warp */
            num_threads = 32;
            for (nelems = 1; nelems <= ELEMS_PER_THREAD * num_threads; nelems *= 2) {
                NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_BROADCAST_TEST, warp, _warp, x)
            }
            DEBUG_PRINT("%d: broadcast warp passed\n", mype);
            /* block */
            num_threads = NVSHM_TEST_NUM_TPB;
            for (nelems = 1; nelems <= ELEMS_PER_THREAD * num_threads; nelems *= 2) {
                NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DO_BROADCAST_TEST, block, _block,
                                                                 x)
            }
            DEBUG_PRINT("%d: broadcast block passed\n", mype);
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
    nvshmem_barrier_all();

    finalize_wrapper();

out:
    return (status) ? status : errs;
}
