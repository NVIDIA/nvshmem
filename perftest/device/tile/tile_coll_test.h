/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TILE_COLL_TEST_H
#define TILE_COLL_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "utils.h"
#include "non_abi/device/common/nvshmem_common.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

using namespace std;

#define MAX_ITERS 100
#define MAX_SKIP 10
#define BARRIER_MAX_ITERS 1000
#define BARRIER_MAX_SKIP 10
#define TEST_NUM_TPB_BLOCK 256

typedef struct run_opt {
    int run_thread;
    int run_warp;
    int run_warpgroup;
    int run_block;
} run_opt_t;

#define cuda_check_error()                                                                   \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (cudaSuccess != e) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(-1);                                                                        \
        }                                                                                    \
    }

#define PROCESS_OPTS(run_options)                          \
    do {                                                   \
        int opt;                                           \
        run_options.run_thread = 1;                        \
        run_options.run_warp = 1;                          \
        run_options.run_warpgroup = 1;                     \
        run_options.run_block = 1;                         \
                                                           \
        while ((opt = getopt(argc, argv, "twba")) != -1) { \
            switch (opt) {                                 \
                case 't':                                  \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 0;              \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'w':                                  \
                    run_options.run_warp = 1;              \
                    run_options.run_thread = 0;            \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'b':                                  \
                    run_options.run_block = 1;             \
                    run_options.run_thread = 0;            \
                    run_options.run_warp = 0;              \
                    break;                                 \
                case 'a':                                  \
                default:                                   \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 1;              \
                    run_options.run_block = 1;             \
                    break;                                 \
            }                                              \
        }                                                  \
    } while (0)

int page_size_roundoff(int value) {
    int page_sz = getpagesize();
    return ((value + page_sz - 1) / page_sz) * page_sz;
}

// Callers have no multicast pointer: probe CFT capability independently of routing priority.
__global__ void check_tile_cft_support(nvshmem_team_t team, bool *supported,
                                       size_t dynamic_smem_size, bool reduction) {
    NVSHMEM_PERF_GIVE_SMEM(dynamic_smem_size);
#ifdef __CUDA_ARCH__
    if (!threadIdx.x) {
        auto *teami = nvshmemi_device_state_d.team_pool[team];
        *supported = reduction
                         ? nvshmemi_is_multicast_reduce_le_implemented<NVSHMEMI_THREADGROUP_WARP>(
                               teami->mc_leid_with_flag, 16)
                         : nvshmemi_is_multicast_le_implemented(teami->mc_leid_with_flag, 16,
                                                                NVSHMEMI_THREADGROUP_WARP);
    }
#endif
    NVSHMEM_PERF_RELEASE_SMEM(dynamic_smem_size);
}

#endif /*TILE_COLL_TEST_H*/
