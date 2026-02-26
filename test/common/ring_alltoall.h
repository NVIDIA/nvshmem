/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _RING_ALL_TO_ALL_H_
#define _RING_ALL_TO_ALL_H_

#include <cuda_runtime.h>

#define MAX_MSG_SIZE 65536
#define ITER 100

typedef void (*launch_alltoall_ptr_t)(void *, void *, size_t, int, int, cudaStream_t);
typedef void (*launch_ring_ptr_t)(void *, void *, size_t, int, int, cudaStream_t);

int setup(bool is_scalar, int disp, size_t max_size = MAX_MSG_SIZE, uint64_t max_iter = ITER,
          bool local_dest = false, int *argc = NULL, char ***argv = NULL);
void cleanup();
template <typename T>
int test(launch_alltoall_ptr_t, launch_ring_ptr_t);

#endif
