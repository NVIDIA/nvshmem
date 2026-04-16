/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define N 100

int test_wait_until_all_on_stream(uint64_t *flags, int *status, int mype, int npes,
                                  cudaStream_t stream) {
    for (int i = 0; i < npes; i++)
        nvshmemx_signal_op_on_stream(flags + mype, 1, NVSHMEM_SIGNAL_SET, i, stream);
    nvshmemx_quiet_on_stream(stream);
    nvshmemx_uint64_wait_until_all_on_stream(flags, npes, status, NVSHMEM_CMP_EQ, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t *flags_h = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    CUDA_CHECK(cudaMemcpy(flags_h, flags, npes * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    /* Check the flags array */
    for (int i = 0; i < npes; i++) {
        if (flags_h[i] != 1) {
            printf("Incorrect flag value = %lu, expected = %lu\n", flags_h[i], (uint64_t)1);
            free(flags_h);
            return 1;
        }
    }

    free(flags_h);
    return 0;
}

int main(int argc, char **argv) {
    int ret_val = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint64_t *flags = NULL;
    if (use_mmap) {
        flags =
            (uint64_t *)allocate_mmap_buffer(npes * sizeof(uint64_t), _mem_handle_type, use_egm);
    } else {
        flags = (uint64_t *)nvshmem_malloc(npes * sizeof(uint64_t));
    }
    int *status = NULL;
    CUDA_CHECK(cudaMalloc((void **)&status, npes * sizeof(int)));
    CUDA_CHECK(cudaMemset(status, 0, npes * sizeof(int)));
    nvshmem_barrier_all();

    ret_val = test_wait_until_all_on_stream(flags, status, mype, npes, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (use_mmap) {
        free_mmap_buffer(flags);
    } else {
        nvshmem_free(flags);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    finalize_wrapper();

    return ret_val;
}
