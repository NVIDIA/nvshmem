/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COLL_TEST_H
#define COLL_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#include "utils.h"
#include <cuda_runtime.h>
#include <algorithm>

#define ELEMS_PER_THREAD 32
#define NVSHM_TEST_NUM_TPB 32
#undef MAX_ELEMS
#define MAX_ELEMS (ELEMS_PER_THREAD * NVSHM_TEST_NUM_TPB)
#define MAX_NPES 128
#define MAX_ITER 32
#define LARGEST_DT uint64_t

#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            status = -1;                                                          \
            goto out;                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#endif /*COLL_TEST_H*/
