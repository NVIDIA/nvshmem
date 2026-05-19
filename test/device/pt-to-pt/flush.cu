/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Test: nvshmemx_flush
 *
 * Validates that nvshmemx_flush() makes source buffers reusable after a
 * non-blocking put.  The shared-memory source test always calls
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

__global__ void test_flush_reuses_smem_source(int *recv_data, int elems_per_block, int mype,
                                              int npes) {
    extern __shared__ char nvshmem_smem[];
    int smem_offset = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    int *payload = (int *)(nvshmem_smem + smem_offset);
    int tid = threadIdx.x;
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    nvshmemx_give_smem(nvshmem_smem, smem_offset);
    __syncthreads();

    if (tid < elems_per_block) {
        payload[tid] = mype * PATTERN_SCALE + offset + tid;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif

    nvshmemx_putmem_nbi_block(recv_data + offset, payload, (size_t)elems_per_block * sizeof(int),
                              peer);
    nvshmemx_flush();

    /* nvshmemx_flush() is thread-scoped.  For block puts, synchronize after
     * every thread has called it so non-leader threads do not overwrite the
     * source before the elected TMA issuer has completed its flush. */
    __syncthreads();

    if (tid < elems_per_block) {
        payload[tid] = POISON_VALUE;
    }
    __syncthreads();

    nvshmem_quiet();
    nvshmemx_release_smem();
}

__global__ void test_flush_reuses_gmem_source_without_tma(int *source_data, int *recv_data,
                                                          int elems_per_block, int mype, int npes) {
    int tid = threadIdx.x;
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    if (tid < elems_per_block) {
        source_data[offset + tid] = mype * PATTERN_SCALE + offset + tid;
    }
    __syncthreads();

    nvshmemx_putmem_nbi_block(recv_data + offset, source_data + offset,
                              (size_t)elems_per_block * sizeof(int), peer);
    nvshmemx_flush();
    __syncthreads();

    if (tid < elems_per_block) {
        source_data[offset + tid] = POISON_VALUE;
    }
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

        CUDA_CHECK(cudaFuncSetAttribute(test_flush_reuses_smem_source,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

        if (peer_p2p_reachable) {
            CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
            nvshmem_barrier_all();
            test_flush_reuses_smem_source<<<NUM_BLOCKS, THREADS_PER_BLOCK, smem_size>>>(
                recv_data, THREADS_PER_BLOCK, mype, npes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems,
                                  cudaMemcpyDeviceToHost));
            if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
                printf("[PE %d] PASS: nvshmemx_flush shared-memory source reuse (%s path)\n", mype,
                       use_tma ? "TMA" : "non-TMA");
            } else {
                status = 1;
            }
        }

        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
        nvshmem_barrier_all();
        test_flush_reuses_gmem_source_without_tma<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            source_data, recv_data, THREADS_PER_BLOCK, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(
            cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems, cudaMemcpyDeviceToHost));
        if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
            printf("[PE %d] PASS: nvshmemx_flush global-memory source reuse\n", mype);
        } else {
            status = 1;
        }

        nvshmem_free(source_data);
        nvshmem_free(recv_data);
    }

    finalize_wrapper();
    return (status == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
