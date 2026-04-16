/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

int test_nvshmemx_uint64_wait_until_all_vector_on_stream(uint64_t *ivars, int *status,
                                                         uint64_t *cmp_values,
                                                         cudaStream_t stream) {
    const int my_pe = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();

    int i = 0;
    int expected_sum = 0;
    int total_sum = 0;

    for (i = 0; i < npes; i++) {
        uint64_t value = i;
        CUDA_CHECK(cudaMemcpy((void *)(cmp_values + i), (void *)&value, sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    }

    for (i = 0; i < npes; i++) {
        nvshmemx_signal_op_on_stream(ivars + my_pe, (uint64_t)my_pe, NVSHMEM_SIGNAL_SET, i, stream);
    }

    expected_sum = (npes - 1) * npes / 2;
    nvshmemx_uint64_wait_until_all_vector_on_stream(ivars, npes, status, NVSHMEM_CMP_EQ, cmp_values,
                                                    stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t *ivars_h = (uint64_t *)malloc(npes * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(ivars_h, ivars, npes * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (i = 0; i < npes; i++) {
        total_sum += ivars_h[i];
    }

    if (expected_sum != total_sum) {
        printf("Incorrect total_sum = %d, expected sum = %d\n", total_sum, expected_sum);
        return 1;
    }
    free(ivars_h);
    return 0;
}

int main(int argc, char **argv) {
    int ret_val = 0;
    cudaStream_t stream;

    init_wrapper(&argc, &argv);
    const int npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint64_t *ivars = (uint64_t *)nvshmem_calloc(npes, sizeof(uint64_t));
    int *status;
    uint64_t *cmp_values;
    CUDA_CHECK(cudaMalloc((void **)&status, npes * sizeof(int)));
    CUDA_CHECK(cudaMemset(status, 0, npes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((uint64_t **)&cmp_values, npes * sizeof(uint64_t)));
    nvshmem_barrier_all();
    ret_val =
        test_nvshmemx_uint64_wait_until_all_vector_on_stream(ivars, status, cmp_values, stream);
    if (ret_val) goto out;

out:
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(cmp_values));
    finalize_wrapper();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ret_val;
}
