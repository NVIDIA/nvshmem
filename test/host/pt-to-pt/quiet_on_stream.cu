/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include "nvshmem.h"
#include "nvshmemx.h"

#include "utils.h"

#define NUM_ITERS 100

int main(int argc, char **argv) {
    int status = 0;
    int num_iters = NUM_ITERS;
    cudaStream_t cstrm;

    init_wrapper(&argc, &argv);

    CUDA_CHECK(cudaStreamCreate(&cstrm));

    for (int i = 0; i < num_iters; i++) {
        nvshmemx_quiet_on_stream(cstrm);
        CUDA_CHECK(cudaStreamSynchronize(cstrm));
    }

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    nvshmem_barrier_all();
    finalize_wrapper();

    return status;
}
