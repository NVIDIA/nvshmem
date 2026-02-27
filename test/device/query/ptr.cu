/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__global__ void check_ptr(int *v_h, int *v_d, int32_t *errors) {
    int me = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    for (int i = 0; i < npes; i++) {
        int *ptr = (int *)nvshmem_ptr(v_d, i);

        if (i == me && ptr == NULL) {
            printf("[%d] Device expected non-NULL for %p\n", me, ptr);
            ++(*errors);
        }

        if (ptr != NULL) atomicAdd_system(ptr, 1);
    }

    int *ptr = (int *)nvshmem_ptr(v_h, me);

    if (ptr != NULL) {
        printf("[%d] Device expected NULL for %p\n", me, ptr);
        ++(*errors);
    }
}

int main(int argc, char **argv) {
    int32_t errors_h, errors = 0;
    int32_t *errors_d;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    int me = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int npeers = 0;
    int *v_d;
    if (use_mmap) {
        v_d = (int *)allocate_mmap_buffer(sizeof(int), _mem_handle_type, use_egm);
    } else {
        v_d = (int *)nvshmem_malloc(sizeof(int));
    }
    if (v_d == NULL) {
        fprintf(stderr, "[%d] nvshmem allocation failed\n", me);
        return EXIT_FAILURE;
    }
    int *v_h = (int *)malloc(sizeof(int));
    if (v_h == NULL) {
        fprintf(stderr, "[%d] host allocation failed\n", me);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemset(v_d, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&errors_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(errors_d, 0, sizeof(int32_t)));

    for (int i = 0; i < npes; i++) {
        int *ptr = (int *)nvshmem_ptr(v_d, i);

        if (i == me && ptr == NULL) {
            printf("[%d] Host expected non-NULL for %p\n", me, ptr);
            ++errors;
        }

        if (ptr != NULL) npeers++;
    }

    int *ptr = (int *)nvshmem_ptr(v_h, me);

    if (ptr != NULL) {
        printf("[%d] Host expected NULL for %p\n", me, ptr);
        ++errors;
    }

    check_ptr<<<1, 1>>>(v_h, v_d, errors_d);

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemcpy(&errors_h, errors_d, sizeof(int32_t), cudaMemcpyDeviceToHost));
    errors += errors_h;

    CUDA_CHECK(cudaMemcpy(v_h, v_d, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (*v_h != npeers) {
        printf("[%d] v_d (%d) != npeers (%d)\n", me, *v_h, npeers);
        ++errors;
    } else {
        printf("[%d] v_d (%d) == npeers (%d)\n", me, *v_h, npeers);
    }

    CUDA_CHECK(cudaFree(errors_d));
    finalize_wrapper();
    return errors != 0;
}
