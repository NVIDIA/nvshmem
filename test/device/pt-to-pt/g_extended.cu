/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "g_extended.cubin"
#include <cstdio>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <iostream>

#include "utils.h"

#define N_RUNS 30

__device__ int d_error;

struct neighbors {
    struct {
        int minus;
        int plus;
    } x;
};

#define TEST_NVSHMEM_LAP_CUBIN()                                                         \
    void *args_lap[] = {                                                                 \
        (void *)&f_in_a, (void *)&stride_y, (void *)&stride_z, (void *)&nx,              \
        (void *)&ny,     (void *)&n,        (void *)&alloc_sz, (void *)&expected_value}; \
    CUfunction test_lap_cubin;                                                           \
    init_test_case_kernel(&test_lap_cubin, NVSHMEMI_TEST_STRINGIFY(lap));                \
    CU_CHECK(cuLaunchKernel(test_lap_cubin, nx / block_size.x, ny / block_size.y,        \
                            nz / block_size.z, 32, 8, 1, 0, stream, args_lap, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void lap(float const *f_in, int stride_y, int stride_z, int nx, int ny, neighbors n,
                    int sz, float expected_value) {
    int const idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int const idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int const idx_z = threadIdx.z + blockIdx.z * blockDim.z;

    int const pos = idx_x + idx_y * stride_y + idx_z * stride_z;
    auto at_in = [=](int idx) -> float const & {
        if (idx < 0 || idx >= sz) printf("OOB Error %d\n", idx);
        return f_in[idx];
    };
    nvshmem_fence();

    float x_minus_v = idx_x != 0 ? 355.0f : nvshmem_float_g(&at_in(pos + (nx - 1)), n.x.minus);
    float x_plus_v = idx_x != nx - 1 ? 355.0f : nvshmem_float_g(&at_in(pos - (nx - 1)), n.x.plus);
    if (idx_x == 0) {
        if (x_minus_v != expected_value) {
            atomicAdd(&d_error, 1);
            float x_minus_v2 = nvshmem_float_g(&at_in(pos + (nx - 1)), n.x.minus);
            printf(
                "Error 1 at (%i %i %i): Old Value: %.2f. New value: %.2f. Expected Value: %.2f\n",
                idx_x, idx_y, idx_z, x_minus_v, x_minus_v2, expected_value);
        }
    }
    if (idx_x == nx - 1) {
        if (x_plus_v != expected_value) {
            atomicAdd(&d_error, 1);
            float x_plus_v2 = nvshmem_float_g(&at_in(pos - (nx - 1)), n.x.plus);
            printf(
                "Error 2 at (%i %i %i): Old Value: %.2f. New value: %.2f. Expected Value: %.2f\n",
                idx_x, idx_y, idx_z, x_plus_v, x_plus_v2, expected_value);
        }
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    neighbors n;
    n.x.plus = (mype + 1) % npes;
    n.x.minus = (mype - 1 + npes) % npes;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int nx = 1024;
    constexpr int ny = 1024;
    constexpr int nz = 64;

    constexpr int nx_tot = nx;
    constexpr int ny_tot = ny;
    constexpr int nz_tot = nz;

    constexpr int stride_x = 1;
    constexpr int stride_y = stride_x * nx_tot;
    constexpr int stride_z = stride_y * ny_tot;
    constexpr int alloc_sz = stride_z * nz_tot;

    float expected_value;
    float *f_in_h = (float *)malloc(alloc_sz * sizeof(float));
    float *f_in_a;
    if (use_mmap) {
        f_in_a = (float *)allocate_mmap_buffer(alloc_sz * sizeof(float), _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        f_in_a = (float *)nvshmem_malloc(alloc_sz * sizeof(float));
    }
    int h_error = 0;

    dim3 block_size{32, 8, 1};
    dim3 grid_size{nx / block_size.x, ny / block_size.y, nz / block_size.z};

    CUDA_CHECK(cudaMemcpyToSymbol(d_error, &h_error, sizeof(int), 0));
    for (int i = 0; i < N_RUNS; i++) {
        expected_value = (float)i;
        nvshmemx_barrier_all_on_stream(stream); /* Make sure that the last loop has finished. */
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int pos = x * stride_x + y * stride_y + z * stride_z;
                    f_in_h[pos] = expected_value;
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(f_in_a, f_in_h, alloc_sz * sizeof(float), cudaMemcpyHostToDevice,
                                   stream));
        nvshmemx_barrier_all_on_stream(stream); /* Make sure that all of the PEs have finished
                                                   copying before starting the lap. */
        if (use_cubin) {
            TEST_NVSHMEM_LAP_CUBIN();
        } else {
            lap<<<grid_size, block_size, 0, stream>>>(f_in_a, stride_y, stride_z, nx, ny, n,
                                                      alloc_sz, expected_value);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_error, d_error, sizeof(int), 0));
        if (h_error != 0) {
            break;
        }
    }

    if (use_mmap) {
        free_mmap_buffer(f_in_a);
    } else {
        nvshmem_free(f_in_a);
    }

    free(f_in_h);
    finalize_wrapper();
    return h_error;
}
