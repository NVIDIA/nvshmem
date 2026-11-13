/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _RING_ALL_TO_ALL_H_
#define _RING_ALL_TO_ALL_H_

#include <cuda_runtime.h>

#define MAX_MSG_SIZE 65536
#define ITER 100

typedef void (*launch_alltoall_ptr_t)(void *, void *, size_t, int, int, cudaStream_t);
typedef void (*launch_ring_ptr_t)(void *, void *, size_t, int, int, cudaStream_t);

template <typename T>
static unsigned int thread_scope_num_threads(size_t len, size_t min_size) {
    /* These generic tests also exercise pointer/TMA fallback, whose CTA-wide
     * staging buffer and barrier require serialized thread-scope callers.
     * Concurrent CFT callers are covered by the CFT completion validation. */
    (void)len;
    (void)min_size;
    return 1;
}

int setup(bool is_scalar, int disp, size_t max_size = MAX_MSG_SIZE, uint64_t max_iter = ITER,
          bool local_dest = false, int *argc = NULL, char ***argv = NULL);
void cleanup();
template <typename T>
int test(launch_alltoall_ptr_t, launch_ring_ptr_t);

#endif
