/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

/*
 * Test: TMA Shared Memory Management APIs
 *
 * Validates that nvshmemx_ask_smem returns consistent values and that shared
 * memory written by the CTA can be delegated to NVSHMEM and used as the source
 * buffer for TMA-backed put operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define NUM_ELEMS 256
#define PATTERN_SCALE 1000

__global__ void test_ask_smem(int *results) {
    if (threadIdx.x == 0) {
        results[0] = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        results[1] = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
        results[2] = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    }
}

__global__ void test_put_from_smem(int *recv_data, int elems_per_block, int mype, int npes) {
    extern __shared__ char nvshmem_smem[];
    int *payload = (int *)nvshmem_smem;
    int tid = threadIdx.x;
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
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
    nvshmem_quiet();
    nvshmemx_release_smem();
}

__global__ void init_source_data(int *source_data, int nelems, int mype) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelems) source_data[idx] = mype * PATTERN_SCALE + idx;
}

__global__ void test_get_to_gmem(int *source_data, int *recv_data, int nelems, int mype,
                                 int npes) {
    extern __shared__ char nvshmem_smem[];
    int peer = (mype + 1) % npes;

    nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    __syncthreads();

    nvshmemx_getmem_block(recv_data, source_data, (size_t)nelems * sizeof(int), peer);

    __syncthreads();
    nvshmemx_release_smem();
}

__global__ void test_get_to_smem(int *source_data, int *recv_data, int elems_per_block, int mype,
                                 int npes) {
    extern __shared__ char nvshmem_smem[];
    int smem_offset = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    int *payload = (int *)(nvshmem_smem + smem_offset);
    int tid = threadIdx.x;
    int offset = blockIdx.x * elems_per_block;
    int peer = (mype + 1) % npes;

    nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    __syncthreads();

    nvshmemx_getmem_block(payload, source_data + offset, (size_t)elems_per_block * sizeof(int),
                          peer);
    __syncthreads();

    if (tid < elems_per_block) {
        recv_data[offset + tid] = payload[tid];
    }
    __syncthreads();
    nvshmemx_release_smem();
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

    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

    if (npes < 2) {
        printf("[PE %d] SKIP: tma_smem requires at least 2 PEs\n", mype);
        goto out;
    }

    if (prop.major < 9) {
        printf("[PE %d] SKIP: TMA shared memory requires sm_90+, current device is sm_%d%d\n",
               mype, prop.major, prop.minor);
        goto out;
    }

    {
        int host_recommended = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        int host_minimum = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
        int host_barriers = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);

        if (host_recommended <= 0 || host_minimum <= 0 || host_barriers <= 0) {
            printf("[PE %d] FAIL: ask_smem returned non-positive value on host\n", mype);
            status = 1;
            goto out;
        }
        if (host_recommended < host_minimum || host_minimum < host_barriers) {
            printf("[PE %d] FAIL: ask_smem size ordering violated: recommended=%d minimum=%d "
                   "barriers=%d\n",
                   mype, host_recommended, host_minimum, host_barriers);
            status = 1;
            goto out;
        }

        int *d_results = NULL;
        CUDA_CHECK(cudaMalloc(&d_results, 3 * sizeof(int)));
        test_ask_smem<<<1, 1>>>(d_results);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_results[3];
        CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_results));

        if (h_results[0] != host_recommended || h_results[1] != host_minimum ||
            h_results[2] != host_barriers) {
            printf("[PE %d] FAIL: ask_smem host/device mismatch: host=(%d,%d,%d) "
                   "device=(%d,%d,%d)\n",
                   mype, host_recommended, host_minimum, host_barriers, h_results[0], h_results[1],
                   h_results[2]);
            status = 1;
            goto out;
        }

        printf("[PE %d] PASS: ask_smem (recommended=%d, minimum=%d, barriers=%d)\n", mype,
               host_recommended, host_minimum, host_barriers);
    }

    {
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        if (!recv_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            status = 1;
            goto out;
        }
        std::vector<int> host(NUM_ELEMS);
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);

        CUDA_CHECK(cudaFuncSetAttribute(test_put_from_smem,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * NUM_ELEMS));

        nvshmem_barrier_all();
        test_put_from_smem<<<1, NUM_ELEMS, smem_size>>>(recv_data, NUM_ELEMS, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * NUM_ELEMS, cudaMemcpyDeviceToHost));
        if (verify_recv_data(host.data(), NUM_ELEMS, prev_pe) == 0) {
            printf("[PE %d] PASS: single-block shared-memory put\n", mype);
        } else {
            status = 1;
        }

        nvshmem_free(recv_data);
    }

    {
        const int num_blocks = 4;
        const int threads_per_block = 64;
        const int total_elems = num_blocks * threads_per_block;
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        if (!recv_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            status = 1;
            goto out;
        }
        std::vector<int> host(total_elems);
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);

        CUDA_CHECK(cudaFuncSetAttribute(test_put_from_smem,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));

        nvshmem_barrier_all();
        test_put_from_smem<<<num_blocks, threads_per_block, smem_size>>>(
            recv_data, threads_per_block, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems, cudaMemcpyDeviceToHost));
        if (verify_recv_data(host.data(), total_elems, prev_pe) == 0) {
            printf("[PE %d] PASS: multi-block shared-memory put\n", mype);
        } else {
            status = 1;
        }

        nvshmem_free(recv_data);
    }

    {
        const int threads_per_block = 64;
        const int total_elems = NUM_ELEMS;
        int *source_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        if (!source_data || !recv_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            status = 1;
            goto out;
        }
        std::vector<int> host(total_elems);
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        int peer = (mype + 1) % npes;

        CUDA_CHECK(cudaFuncSetAttribute(test_get_to_gmem,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
        init_source_data<<<(total_elems + threads_per_block - 1) / threads_per_block,
                           threads_per_block>>>(source_data, total_elems, mype);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        nvshmem_barrier_all();
        test_get_to_gmem<<<1, threads_per_block, smem_size>>>(source_data, recv_data, total_elems,
                                                              mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(
            cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems, cudaMemcpyDeviceToHost));
        if (verify_recv_data(host.data(), total_elems, peer) == 0) {
            printf("[PE %d] PASS: TMA get to global memory\n", mype);
        } else {
            status = 1;
        }

        nvshmem_free(recv_data);
        nvshmem_free(source_data);
    }

    {
        const int num_blocks = 4;
        const int threads_per_block = 64;
        const int total_elems = num_blocks * threads_per_block;
        int *source_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        if (!source_data || !recv_data) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            status = 1;
            goto out;
        }
        std::vector<int> host(total_elems);
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        int peer = (mype + 1) % npes;

        CUDA_CHECK(cudaFuncSetAttribute(test_get_to_smem,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * total_elems));
        init_source_data<<<num_blocks, threads_per_block>>>(source_data, total_elems, mype);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        nvshmem_barrier_all();
        test_get_to_smem<<<num_blocks, threads_per_block, smem_size>>>(
            source_data, recv_data, threads_per_block, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        CUDA_CHECK(
            cudaMemcpy(host.data(), recv_data, sizeof(int) * total_elems, cudaMemcpyDeviceToHost));
        if (verify_recv_data(host.data(), total_elems, peer) == 0) {
            printf("[PE %d] PASS: TMA get to shared memory\n", mype);
        } else {
            status = 1;
        }

        nvshmem_free(recv_data);
        nvshmem_free(source_data);
    }

out:
    finalize_wrapper();
    return status;
}
