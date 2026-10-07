/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

constexpr int NUM_ELEMS = 1024;

__device__ int flat_thread_id() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ void delay_nonleader_warps() {
    if (flat_thread_id() >= warpSize) {
        unsigned long long start = clock64();
        while (clock64() - start < 10000) {
        }
    }
}

__global__ void test_quiet_block(int *source, int *received, int *errors) {
    int tid = flat_thread_id();
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int peer = (nvshmem_my_pe() + 1) % nvshmem_n_pes();

    // The entry barrier must include operations issued by later-arriving warps.
    delay_nonleader_warps();
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        nvshmem_int_get_nbi(received + i, source + i, 1, peer);
    }
    nvshmemx_quiet_block();

    // Each thread consumes another thread's result immediately after return.
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        int j = (i + 33) % NUM_ELEMS;
        if (received[j] != peer * NUM_ELEMS + j + 1) {
            atomicAdd(errors, 1);
        }
    }
}

__global__ void test_fence_block(int *received, int *ready, int *errors) {
    int tid = flat_thread_id();
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    int sender = (mype + npes - 1) % npes;

    delay_nonleader_warps();
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        nvshmem_int_p(received + i, mype * NUM_ELEMS + i + 1, peer);
    }
    nvshmemx_fence_block();
    if (tid == nthreads - 1) {
        nvshmem_int_p(ready, 1, peer);
    }

    // Check data after the notification, before quiet can mask missing ordering.
    nvshmem_int_wait_until(ready, NVSHMEM_CMP_EQ, 1);
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        if (received[i] != sender * NUM_ELEMS + i + 1) {
            atomicAdd(errors, 1);
        }
    }
    nvshmemx_quiet_block();
}

__global__ void test_tma_block(int *received, int *ready, int *errors, int smem_size,
                               bool use_fence) {
    extern __shared__ char nvshmem_smem[];
    __shared__ __align__(16) int source[NUM_ELEMS];
    int tid = flat_thread_id();
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int mype = nvshmem_my_pe();

    nvshmemx_give_smem(nvshmem_smem, smem_size);
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        source[i] = mype * NUM_ELEMS + i + 1;
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    // Self puts always have a P2P path; multiple warps own pending bulk groups.
    for (int i = tid * 4; i < NUM_ELEMS; i += nthreads * 4) {
        nvshmem_putmem_nbi(received + i, source + i, 4 * sizeof(int), mype);
    }
    if (use_fence) {
        nvshmemx_fence_block();
        if (tid == nthreads - 1) {
            nvshmem_int_p(ready, 1, mype);
        }
        nvshmem_int_wait_until(ready, NVSHMEM_CMP_EQ, 1);
    } else {
        nvshmemx_quiet_block();
    }
    // Validate before release_smem can complete any remaining transfers.
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        int j = (i + 33) % NUM_ELEMS;
        if (received[j] != mype * NUM_ELEMS + j + 1) {
            atomicAdd(errors, 1);
        }
    }
    __syncthreads();
    nvshmemx_release_smem();
}

int main(int argc, char **argv) {
    setenv("NVSHMEM_TMA_POLICY", "ENABLE", 1);
    init_wrapper(&argc, &argv);
    int mype = nvshmem_my_pe();
    int *source = static_cast<int *>(nvshmem_malloc(NUM_ELEMS * sizeof(int)));
    int *received = static_cast<int *>(nvshmem_malloc(NUM_ELEMS * sizeof(int)));
    int *ready = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    int *errors = nullptr;
    if (!source || !received || !ready) {
        fprintf(stderr, "[PE %d] FAIL: symmetric allocation failed\n", mype);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMalloc(&errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(int)));

    std::vector<int> values(NUM_ELEMS);
    for (int i = 0; i < NUM_ELEMS; i++) {
        values[i] = mype * NUM_ELEMS + i + 1;
    }
    CUDA_CHECK(cudaMemcpy(source, values.data(), NUM_ELEMS * sizeof(int), cudaMemcpyHostToDevice));

    // Include partial warps and multidimensional blocks to check leader selection.
    const dim3 blocks[] = {dim3(1), dim3(17), dim3(32), dim3(128), dim3(8, 4, 4)};
    for (dim3 block : blocks) {
        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        nvshmem_barrier_all();
        test_quiet_block<<<1, block>>>(source, received, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        CUDA_CHECK(cudaMemset(ready, 0, sizeof(int)));
        nvshmem_barrier_all();
        void *args[] = {&received, &ready, &errors};
        int status = nvshmemx_collective_launch(reinterpret_cast<const void *>(test_fence_block),
                                                dim3(1), block, args, 0, nullptr);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "[PE %d] FAIL: collective launch returned %d\n", mype, status);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
    }

    int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    for (bool use_fence : {false, true}) {
        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        CUDA_CHECK(cudaMemset(ready, 0, sizeof(int)));
        test_tma_block<<<1, dim3(8, 4, 4), smem_size>>>(received, ready, errors, smem_size,
                                                        use_fence);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int host_errors = 0;
    CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[PE %d] %s: block fence/quiet (%d errors)\n", mype, host_errors ? "FAIL" : "PASS",
           host_errors);
    CUDA_CHECK(cudaFree(errors));
    nvshmem_free(ready);
    nvshmem_free(received);
    nvshmem_free(source);
    finalize_wrapper();
    return host_errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
