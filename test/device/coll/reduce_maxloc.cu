/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "reduce_maxloc.cubin"

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

#define DO_TEST_CUBIN()                                                                          \
    void *args[] = {(void *)&team, (void *)&dest, (void *)&source, (void *)&nelems,              \
                    (void *)&_dynamic_smem_size};                                                \
    CUfunction test_double2_maxloc_reduce_kernel_block_cubin;                                    \
    init_test_case_kernel(&test_double2_maxloc_reduce_kernel_block_cubin,                        \
                          "test_double2_maxloc_reduce_kernel_block");                            \
    CU_CHECK(cuLaunchKernel(test_double2_maxloc_reduce_kernel_block_cubin, 1, 1, 1, num_threads, \
                            1, 1, _dynamic_smem_size, cstrm, args, NULL));

__device__ void init_double2_maxloc_reduce_data_block(nvshmem_team_t team, double2 *source) {
    int team_my_pe = nvshmem_team_my_pe(team);
    source[0].x = team_my_pe;
    source[0].y = team_my_pe;
}

__device__ void reset_double2_maxloc_reduce_data_block(nvshmem_team_t team, double2 *dest) {
    dest[0].x = 0;
    dest[0].y = 0;
}

__device__ void validate_double2_maxloc_reduce_data_block(nvshmem_team_t team, double2 *dest) {
    int team_n_pes = nvshmem_team_n_pes(team);
    int myIdx = nvshmtest_thread_id_in_block();

    if (!myIdx) {
        if (dest[0].x != team_n_pes - 1 || dest[0].y != team_n_pes - 1) {
            printf("found: (%lf, %d), expected: (%lf, %d)\n", dest[0].x, (int)dest[0].y,
                   (double)(team_n_pes - 1), team_n_pes - 1);
            atomicAdd(&errs_d, 1);
        }
    }
    __syncthreads();
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void test_double2_maxloc_reduce_kernel_block(nvshmem_team_t team, double2 *dest,
                                                        double2 *source, size_t nelems,
                                                        size_t dynamic_smem_size) {
    int myIdx = nvshmtest_thread_id_in_block();
    int groupSize = nvshmtest_block_size();
    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    init_double2_maxloc_reduce_data_block(team, source);
    for (int j = 0; j < MAX_ITER; j++) {
        reset_double2_maxloc_reduce_data_block(team, dest);
        init_double2_maxloc_reduce_data_block(team, source);
        nvshmemx_barrier_block(team);
        nvshmemx_double2_maxloc_reduce_block(team, dest, source, nelems);
        validate_double2_maxloc_reduce_data_block(team, dest);
        // nvshmemx_double_broadcast_block(team, (double *)source, (const double *)source, nelems,
        // 0);
    }
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DO_TEST()                                                                               \
    if (use_cubin) {                                                                            \
        DO_TEST_CUBIN();                                                                        \
    } else {                                                                                    \
        test_double2_maxloc_reduce_kernel_block<<<1, num_threads, _dynamic_smem_size, cstrm>>>( \
            team, (double2 *)dest, (double2 *)source, nelems, _dynamic_smem_size);              \
    }                                                                                           \
    CUDA_RUNTIME_CHECK(cudaGetLastError());                                                     \
    cudaStreamSynchronize(cstrm);

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    unsigned num_threads;
#undef LARGEST_DT
#define LARGEST_DT double2
    size_t alloc_size = MAX_ELEMS * 2 * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    cudaStream_t cstrm;
    size_t nelems;
    char size_string[100];
    unsigned long long int errs;
    read_args(argc, argv);
    // Maxloc reduction does not support the CFT path yet.
    _dynamic_smem_size = SMEM_SIZE_DISABLE;
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }
    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    /*status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failted \n");
        status = -1;
        goto out;
    }*/

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
            printf("%d: Testing team: %s (my_pe = %d, n_pes = %d), team_id: %d\n", mype,
                   map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                   nvshmem_team_n_pes(team), team);

            num_threads = NVSHM_TEST_NUM_TPB;
            nelems = 1;
            DO_TEST();
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(cstrm));
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
    if (status) {
        return status;
    }
    return errs;
}
