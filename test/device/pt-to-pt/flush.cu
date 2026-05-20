/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Test: nvshmemx_flush
 *
 * Validates that nvshmemx_flush and its scoped variants make source buffers
 * reusable after a non-blocking put.  The shared-memory source test always calls
 * nvshmemx_give_smem(); on non-TMA-capable devices that registration is a
 * no-op and the put falls back to the regular path.  The global-memory source
 * test also runs on every device.  Both kernels overwrite the source buffer
 * immediately after flush; the peer must still receive the original pattern.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

constexpr int NUM_BLOCKS = 4;
constexpr int THREADS_PER_BLOCK = 64;
constexpr int PATTERN_SCALE = 1000;
constexpr int POISON_VALUE = -42;

enum flush_test_scope_t {
    FLUSH_TEST_SCOPE_THREAD,
    FLUSH_TEST_SCOPE_WARP,
    FLUSH_TEST_SCOPE_BLOCK,
};

__device__ int flat_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ int flat_block_size() { return blockDim.x * blockDim.y * blockDim.z; }

template <flush_test_scope_t SCOPE>
__device__ void putmem_nbi_scope(void *dest, const void *source, size_t bytes, int pe);

template <>
__device__ void putmem_nbi_scope<FLUSH_TEST_SCOPE_THREAD>(void *dest, const void *source,
                                                          size_t bytes, int pe) {
    if (flat_thread_id() == 0) nvshmem_putmem_nbi(dest, source, bytes, pe);
}

template <>
__device__ void putmem_nbi_scope<FLUSH_TEST_SCOPE_WARP>(void *dest, const void *source,
                                                        size_t bytes, int pe) {
    if (flat_thread_id() < warpSize) nvshmemx_putmem_nbi_warp(dest, source, bytes, pe);
}

template <>
__device__ void putmem_nbi_scope<FLUSH_TEST_SCOPE_BLOCK>(void *dest, const void *source,
                                                         size_t bytes, int pe) {
    nvshmemx_putmem_nbi_block(dest, source, bytes, pe);
}

template <flush_test_scope_t SCOPE>
__device__ void flush_scope();

template <>
__device__ void flush_scope<FLUSH_TEST_SCOPE_THREAD>() {
    if (flat_thread_id() == 0) nvshmemx_flush();
}

template <>
__device__ void flush_scope<FLUSH_TEST_SCOPE_WARP>() {
    if (flat_thread_id() < warpSize) nvshmemx_flush_warp();
}

template <>
__device__ void flush_scope<FLUSH_TEST_SCOPE_BLOCK>() {
    nvshmemx_flush_block();
}

template <flush_test_scope_t SCOPE>
__device__ void poison_source(int *source, int nelems);

template <>
__device__ void poison_source<FLUSH_TEST_SCOPE_THREAD>(int *source, int nelems) {
    if (flat_thread_id() == 0) {
        for (int i = 0; i < nelems; i++) source[i] = POISON_VALUE;
    }
}

template <>
__device__ void poison_source<FLUSH_TEST_SCOPE_WARP>(int *source, int nelems) {
    int tid = flat_thread_id();
    if (tid < warpSize) {
        for (int i = tid; i < nelems; i += warpSize) source[i] = POISON_VALUE;
    }
}

template <>
__device__ void poison_source<FLUSH_TEST_SCOPE_BLOCK>(int *source, int nelems) {
    int tid = flat_thread_id();
    for (int i = tid; i < nelems; i += flat_block_size()) source[i] = POISON_VALUE;
}

template <flush_test_scope_t SCOPE>
__global__ void test_flush_reuses_smem_source(int *recv_data, int elems_per_block, int mype,
                                              int npes) {
    extern __shared__ char nvshmem_smem[];
    int smem_offset = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    int *payload = (int *)(nvshmem_smem + smem_offset);
    int tid = flat_thread_id();
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    nvshmemx_give_smem(nvshmem_smem, smem_offset);
    __syncthreads();

    for (int i = tid; i < elems_per_block; i += flat_block_size())
        payload[i] = mype * PATTERN_SCALE + offset + i;
    __syncthreads();

#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif

    putmem_nbi_scope<SCOPE>(recv_data + offset, payload, (size_t)elems_per_block * sizeof(int),
                            peer);
    flush_scope<SCOPE>();

    poison_source<SCOPE>(payload, elems_per_block);
    __syncthreads();

    nvshmem_quiet();
    nvshmemx_release_smem();
}

template <flush_test_scope_t SCOPE>
__global__ void test_flush_reuses_gmem_source_without_tma(int *source_data, int *recv_data,
                                                          int elems_per_block, int mype, int npes) {
    int tid = flat_thread_id();
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    for (int i = tid; i < elems_per_block; i += flat_block_size())
        source_data[offset + i] = mype * PATTERN_SCALE + offset + i;
    __syncthreads();

    putmem_nbi_scope<SCOPE>(recv_data + offset, source_data + offset,
                            (size_t)elems_per_block * sizeof(int), peer);
    flush_scope<SCOPE>();

    poison_source<SCOPE>(source_data + offset, elems_per_block);
    __syncthreads();

    nvshmem_quiet();
}

__global__ void test_flush_reuses_gmem_source_block_put_thread_flush(int *source_data,
                                                                     int *recv_data,
                                                                     int elems_per_block, int mype,
                                                                     int npes) {
    int tid = flat_thread_id();
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    for (int i = tid; i < elems_per_block; i += flat_block_size())
        source_data[offset + i] = mype * PATTERN_SCALE + offset + i;
    __syncthreads();

    nvshmemx_putmem_nbi_block(recv_data + offset, source_data + offset,
                              (size_t)elems_per_block * sizeof(int), peer);
    nvshmemx_flush();
    __syncthreads();

    for (int i = tid; i < elems_per_block; i += flat_block_size())
        source_data[offset + i] = POISON_VALUE;
    __syncthreads();

    nvshmem_quiet();
}

static int verify_recv_data(const int *host, int nelems, int prev_pe) {
    for (int i = 0; i < nelems; i++) {
        int expected = prev_pe * PATTERN_SCALE + i;
        if (host[i] != expected) {
            printf("[PE %d] FAIL: recv[%d] = %d, expected %d\n", nvshmem_my_pe(), i, host[i],
                   expected);
            return 1;
        }
    }
    return 0;
}

static const char *scope_name(flush_test_scope_t scope) {
    switch (scope) {
        case FLUSH_TEST_SCOPE_THREAD:
            return "thread";
        case FLUSH_TEST_SCOPE_WARP:
            return "warp";
        case FLUSH_TEST_SCOPE_BLOCK:
            return "block";
    }
    return "unknown";
}

template <flush_test_scope_t SCOPE>
static int run_smem_source_scope(int *recv_data, std::vector<int> &host, int total_elems,
                                 int elems_per_block, int smem_size, int mype, int npes,
                                 int prev_pe, bool use_tma) {
    CUDA_CHECK(cudaFuncSetAttribute(test_flush_reuses_smem_source<SCOPE>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
    nvshmem_barrier_all();
    test_flush_reuses_smem_source<SCOPE><<<NUM_BLOCKS, THREADS_PER_BLOCK, smem_size>>>(
        recv_data, elems_per_block, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems,
                          cudaMemcpyDeviceToHost));
    if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
        printf("[PE %d] PASS: nvshmemx_flush %s-scope shared-memory source reuse (%s path)\n",
               mype, scope_name(SCOPE), use_tma ? "TMA" : "non-TMA");
        return 0;
    }
    return 1;
}

template <flush_test_scope_t SCOPE>
static int run_gmem_source_scope(int *source_data, int *recv_data, std::vector<int> &host,
                                 int total_elems, int elems_per_block, int mype, int npes,
                                 int prev_pe) {
    CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
    nvshmem_barrier_all();
    test_flush_reuses_gmem_source_without_tma<SCOPE><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        source_data, recv_data, elems_per_block, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems,
                          cudaMemcpyDeviceToHost));
    if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
        printf("[PE %d] PASS: nvshmemx_flush %s-scope global-memory source reuse\n", mype,
               scope_name(SCOPE));
        return 0;
    }
    return 1;
}

static int run_gmem_source_block_put_thread_flush(int *source_data, int *recv_data,
                                                  std::vector<int> &host, int total_elems,
                                                  int elems_per_block, int mype, int npes,
                                                  int prev_pe) {
    CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
    nvshmem_barrier_all();
    test_flush_reuses_gmem_source_block_put_thread_flush<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        source_data, recv_data, elems_per_block, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems,
                          cudaMemcpyDeviceToHost));
    if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
        printf("[PE %d] PASS: nvshmemx_flush thread-scope global-memory source reuse after "
               "block-scope put\n",
               mype);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int status = 0;
    int dev_id = 0;
    cudaDeviceProp prop;

    setenv("NVSHMEM_TMA_POLICY", "ENABLE", 1);
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int prev_pe = (mype - 1 + npes) % npes;
    bool use_tma = false;

    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    use_tma = (prop.major >= 9);

    if (npes < 2) {
        printf("[PE %d] SKIP: nvshmemx_flush test requires at least 2 PEs\n", mype);
        finalize_wrapper();
        return EXIT_SUCCESS;
    }

    if (!use_tma) {
        printf(
            "[PE %d] INFO: nvshmemx_flush TMA source-reuse path requires sm_90+, current "
            "device is sm_%d%d; give_smem is expected to be a no-op\n",
            mype, prop.major, prop.minor);
    }

    /* Check functionality of nvshmemx_flush on host */
    nvshmemx_flush();

    {
        const int total_elems = NUM_BLOCKS * THREADS_PER_BLOCK;
        int *source_data = NULL;
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        if (!recv_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            finalize_wrapper();
            return EXIT_FAILURE;
        }
        source_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        if (!source_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            nvshmem_free(recv_data);
            finalize_wrapper();
            return EXIT_FAILURE;
        }

        std::vector<int> host(total_elems);
        int peer = (mype + 1) % npes;
        bool peer_p2p_reachable = (nvshmem_ptr(recv_data, peer) != NULL);
        int smem_offset = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
        int smem_size = smem_offset + THREADS_PER_BLOCK * (int)sizeof(int);

        if (peer_p2p_reachable) {
            status |= run_smem_source_scope<FLUSH_TEST_SCOPE_THREAD>(
                recv_data, host, total_elems, THREADS_PER_BLOCK, smem_size, mype, npes, prev_pe,
                use_tma);
            status |= run_smem_source_scope<FLUSH_TEST_SCOPE_WARP>(
                recv_data, host, total_elems, THREADS_PER_BLOCK, smem_size, mype, npes, prev_pe,
                use_tma);
            status |= run_smem_source_scope<FLUSH_TEST_SCOPE_BLOCK>(
                recv_data, host, total_elems, THREADS_PER_BLOCK, smem_size, mype, npes, prev_pe,
                use_tma);
        }

        status |= run_gmem_source_scope<FLUSH_TEST_SCOPE_THREAD>(
            source_data, recv_data, host, total_elems, THREADS_PER_BLOCK, mype, npes, prev_pe);
        status |= run_gmem_source_block_put_thread_flush(source_data, recv_data, host, total_elems,
                                                         THREADS_PER_BLOCK, mype, npes, prev_pe);
        status |= run_gmem_source_scope<FLUSH_TEST_SCOPE_WARP>(
            source_data, recv_data, host, total_elems, THREADS_PER_BLOCK, mype, npes, prev_pe);
        status |= run_gmem_source_scope<FLUSH_TEST_SCOPE_BLOCK>(
            source_data, recv_data, host, total_elems, THREADS_PER_BLOCK, mype, npes, prev_pe);

        nvshmem_free(source_data);
        nvshmem_free(recv_data);
    }

    finalize_wrapper();
    return (status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
