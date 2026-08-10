/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include <mpi.h>
#endif
#include <cassert>
#include <vector>
#include "utils.h"

#define INIT_DEFAULT_ITERS 150
#define INIT_DEFAULT_ITERS_LARGE_NPES 20
#define INIT_DEFAULT_ITERS_CFT_LARGE_NPES 10
#define INIT_LARGE_NPES_THRESHOLD 8

int main(int argc, char *argv[]) {
    int devices, npes, num_iters = 0;
    const char *test_num_iter = getenv("NVSHMEMTEST_INIT_NUM_ITERS");
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    npes = nvshmem_n_pes();
    nvshmem_barrier_all();
    nvshmem_finalize();

    if (test_num_iter) {
        num_iters = atoi(test_num_iter);
    }

    if (num_iters <= 0) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
        const bool reduce_cft_iters = npes >= INIT_LARGE_NPES_THRESHOLD;
#else
        constexpr bool reduce_cft_iters = false;
#endif
        if (reduce_cft_iters) {
            num_iters = INIT_DEFAULT_ITERS_CFT_LARGE_NPES;
        } else {
            num_iters = (use_mmap && !test_num_iter && npes > INIT_LARGE_NPES_THRESHOLD)
                            ? INIT_DEFAULT_ITERS_LARGE_NPES
                            : INIT_DEFAULT_ITERS;
        }
    }

    for (int i = 0; i < num_iters; i++) {
        printf("Step %d\n", i);
        nvshmem_init();
        int *destination = NULL;
        if (use_mmap) {
            destination = (int *)allocate_mmap_buffer(sizeof(int), _mem_handle_type, use_egm);
            free_mmap_buffer(destination);
        } else {
            destination = (int *)nvshmem_malloc(sizeof(int));
            nvshmem_free(destination);
        }
        nvshmem_finalize();
        printf("Step %d done\n", i);
    }
    nvshmem_init();     /* finalize_wrapper will call nvshmem_finalize();
                           this is the corresponding init for it */
    finalize_wrapper(); /* should finalize boostrap stuff as well */
}
