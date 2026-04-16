/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__global__ void test_kernel() { nvshmem_global_exit(0); }

int main(int c, char *v[]) {
    int status = 0;

    init_wrapper(&c, &v);

#ifdef _NVSHMEM_DEBUG
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
#endif

    DEBUG_PRINT("[%d of %d] hello world! \n", mype, npes);

    nvshmem_barrier_all();

    if (mype == 0) {
        test_kernel<<<1, 1, 0>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        /* Note, this should be unreachable. return a unique error code if we reach here. */
        status = 2;
    } else {
        sleep(60); /* This is added to allow the PE0's global_exit to abort the program before PE1+
                      finalize themselves */
        fprintf(stderr, "Was able to get to the end of the test.\n");
        finalize_wrapper();
        return 1;
    }

    return status;
}
