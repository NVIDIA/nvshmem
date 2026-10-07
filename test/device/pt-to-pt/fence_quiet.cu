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
constexpr int MAX_WARPS = 4;

__device__ int flat_thread_id() {
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

template <bool WARP>
__device__ void quiet_scope() {
    if constexpr (WARP) {
        nvshmemx_quiet_warp();
    } else {
        nvshmemx_quiet_block();
    }
}

template <bool WARP>
__device__ void fence_scope() {
    if constexpr (WARP) {
        nvshmemx_fence_warp();
    } else {
        nvshmemx_fence_block();
    }
}

template <bool WARP>
__device__ void group_layout(int &lane, int &size, int &group, int &nelems) {
    int tid = flat_thread_id();
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    lane = WARP ? tid % warpSize : tid;
    group = WARP ? tid / warpSize : 0;
    size = WARP ? min(warpSize, nthreads - group * warpSize) : nthreads;
    int ngroups = WARP ? (nthreads + warpSize - 1) / warpSize : 1;
    nelems = NUM_ELEMS / ngroups;
}

__device__ void delay_nonleader_warps() {
    if (flat_thread_id() >= warpSize) {
        unsigned long long start = clock64();
        while (clock64() - start < 10000) {
        }
    }
}

template <bool WARP>
__global__ void test_quiet_scope(int *source, int *received, int *errors) {
    int lane, size, group, nelems;
    group_layout<WARP>(lane, size, group, nelems);
    int base = group * nelems;
    int peer = (nvshmem_my_pe() + 1) % nvshmem_n_pes();

    // The entry barrier must include operations issued by later-arriving warps.
    delay_nonleader_warps();
    for (int i = lane; i < nelems; i += size) {
        nvshmem_int_get_nbi(received + base + i, source + base + i, 1, peer);
    }
    quiet_scope<WARP>();

    // Consume another thread's result within this group immediately after return.
    for (int i = lane; i < nelems; i += size) {
        int j = base + (i + 33) % nelems;
        if (received[j] != peer * NUM_ELEMS + j + 1) {
            atomicAdd(errors, 1);
        }
    }
}

template <bool WARP>
__global__ void test_fence_scope(int *received, int *ready, int *errors) {
    int lane, size, group, nelems;
    group_layout<WARP>(lane, size, group, nelems);
    int base = group * nelems;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    int sender = (mype + npes - 1) % npes;

    delay_nonleader_warps();
    for (int i = lane; i < nelems; i += size) {
        nvshmem_int_p(received + base + i, mype * NUM_ELEMS + base + i + 1, peer);
    }
    fence_scope<WARP>();
    if (lane == size - 1) {
        nvshmem_int_p(ready + group, 1, peer);
    }

    // Check data after the notification, before quiet can mask missing ordering.
    nvshmem_int_wait_until(ready + group, NVSHMEM_CMP_EQ, 1);
    for (int i = lane; i < nelems; i += size) {
        if (received[base + i] != sender * NUM_ELEMS + base + i + 1) {
            atomicAdd(errors, 1);
        }
    }
    quiet_scope<WARP>();
}

template <bool WARP>
__global__ void test_tma_scope(int *received, int *ready, int *errors, int smem_size,
                               bool use_fence) {
    extern __shared__ char nvshmem_smem[];
    __shared__ __align__(16) int source[NUM_ELEMS];
    int tid = flat_thread_id();
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int mype = nvshmem_my_pe();
    int lane, size, group, nelems;
    group_layout<WARP>(lane, size, group, nelems);
    int base = group * nelems;

    nvshmemx_give_smem(nvshmem_smem, smem_size);
    for (int i = tid; i < NUM_ELEMS; i += nthreads) {
        source[i] = mype * NUM_ELEMS + i + 1;
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    // Self puts always have a P2P path; multiple warps own pending bulk groups.
    for (int i = lane * 4; i < nelems; i += size * 4) {
        nvshmem_putmem_nbi(received + base + i, source + base + i, 4 * sizeof(int), mype);
    }
    if (use_fence) {
        fence_scope<WARP>();
        if (lane == size - 1) {
            nvshmem_int_p(ready + group, 1, mype);
        }
        nvshmem_int_wait_until(ready + group, NVSHMEM_CMP_EQ, 1);
    } else {
        quiet_scope<WARP>();
    }
    // Validate before release_smem can complete any remaining transfers.
    for (int i = lane; i < nelems; i += size) {
        int j = base + (i + 33) % nelems;
        if (received[j] != mype * NUM_ELEMS + j + 1) {
            atomicAdd(errors, 1);
        }
    }
    __syncthreads();
    nvshmemx_release_smem();
}

template <bool WARP>
void run_scope_tests(int *source, int *received, int *ready, int *errors) {
    // Include partial warps and multidimensional blocks to check leader selection.
    const dim3 blocks[] = {dim3(1), dim3(17), dim3(32), dim3(48), dim3(128), dim3(8, 4, 4)};
    for (dim3 block : blocks) {
        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        nvshmem_barrier_all();
        test_quiet_scope<WARP><<<1, block>>>(source, received, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        CUDA_CHECK(cudaMemset(ready, 0, MAX_WARPS * sizeof(int)));
        nvshmem_barrier_all();
        void *args[] = {&received, &ready, &errors};
        int status =
            nvshmemx_collective_launch(reinterpret_cast<const void *>(test_fence_scope<WARP>),
                                       dim3(1), block, args, 0, nullptr);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "[PE %d] FAIL: collective launch returned %d\n", nvshmem_my_pe(),
                    status);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
    }

    int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    for (bool use_fence : {false, true}) {
        CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
        CUDA_CHECK(cudaMemset(ready, 0, MAX_WARPS * sizeof(int)));
        test_tma_scope<WARP>
            <<<1, dim3(8, 4, 4), smem_size>>>(received, ready, errors, smem_size, use_fence);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvshmem_barrier_all();
}

__global__ void test_warp_isolation(int *source, int *received, int *errors) {
    __shared__ int done;
    int tid = flat_thread_id();
    int peer = (nvshmem_my_pe() + 1) % nvshmem_n_pes();
    if (tid == 0) {
        done = 0;
    }
    __syncthreads();
    if (tid / warpSize == 1) {
        int lane = tid % warpSize;
        nvshmem_int_get_nbi(received + lane, source + lane, 1, peer);
        nvshmemx_quiet_warp();
        int next = (lane + 1) % warpSize;
        if (received[next] != peer * NUM_ELEMS + next + 1) {
            atomicAdd(errors, 1);
        }
        nvshmemx_fence_warp();
        if (lane == 0) {
            atomicExch(&done, 1);
        }
    } else {
        // A block barrier inside either warp API would deadlock with these warps.
        while (atomicAdd(&done, 0) == 0) {
            __nanosleep(64);
        }
    }
}

int main(int argc, char **argv) {
    setenv("NVSHMEM_TMA_POLICY", "ENABLE", 1);
    init_wrapper(&argc, &argv);
    int mype = nvshmem_my_pe();
    int *source = static_cast<int *>(nvshmem_malloc(NUM_ELEMS * sizeof(int)));
    int *received = static_cast<int *>(nvshmem_malloc(NUM_ELEMS * sizeof(int)));
    int *ready = static_cast<int *>(nvshmem_malloc(MAX_WARPS * sizeof(int)));
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

    run_scope_tests<false>(source, received, ready, errors);
    run_scope_tests<true>(source, received, ready, errors);
    CUDA_CHECK(cudaMemset(received, 0, NUM_ELEMS * sizeof(int)));
    test_warp_isolation<<<1, dim3(8, 4, 4)>>>(source, received, errors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int host_errors = 0;
    CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[PE %d] %s: warp/block fence/quiet (%d errors)\n", mype, host_errors ? "FAIL" : "PASS",
           host_errors);
    CUDA_CHECK(cudaFree(errors));
    nvshmem_free(ready);
    nvshmem_free(received);
    nvshmem_free(source);
    finalize_wrapper();
    return host_errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
