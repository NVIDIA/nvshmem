/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>
#include "utils.h"

int test_nvshmem_signal_set_on_stream(uint64_t *remote, cudaStream_t stream) {
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    nvshmemx_signal_op_on_stream(remote, (uint64_t)mype, NVSHMEM_SIGNAL_SET, (mype + 1) % npes,
                                 stream);
    cudaStreamSynchronize(stream);
    nvshmem_barrier_all();

    uint64_t remote_val = nvshmem_signal_fetch(remote);
    if (remote_val != (uint64_t)((mype + npes - 1) % npes)) {
        printf("PE %i received incorrect value\n", mype);
        return 1;
    }
    return 0;
}

int test_nvshmem_signal_add_on_stream(uint64_t *remote, cudaStream_t stream) {
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    for (int i = 0; i < npes; i++)
        nvshmemx_signal_op_on_stream(remote, (uint64_t)(mype + 1), NVSHMEM_SIGNAL_ADD, i, stream);
    cudaStreamSynchronize(stream);
    nvshmem_barrier_all();

    uint64_t remote_val = nvshmem_signal_fetch(remote);
    if (remote_val != (uint64_t)(npes * (npes + 1) / 2)) {
        printf("PE %i observed error\n", mype);
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int ret_val = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint64_t *remote = NULL;
    if (use_mmap) {
        remote = (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t), _mem_handle_type, use_egm);
    } else {
        remote = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    }
    nvshmem_barrier_all();

    ret_val = test_nvshmem_signal_set_on_stream(remote, stream);
    if (ret_val) goto out;

    if (use_egm) {
        memset((void *)remote, 0, sizeof(uint64_t));
    } else {
        CUDA_CHECK(cudaMemset((void *)remote, 0, sizeof(uint64_t)));
    }
    nvshmem_barrier_all();
    ret_val = test_nvshmem_signal_add_on_stream(remote, stream);
out:
    if (use_mmap) {
        free_mmap_buffer(remote);
    } else {
        nvshmem_free(remote);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    finalize_wrapper();
    return ret_val;
}
