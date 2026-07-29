/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Test: Concurrent TMA Shared Memory Registration
 *
 * Launches two grids with identical CTA indices on independent streams. Both
 * grids register their dynamic shared memory before either starts its PUT.
 * This catches registration tables indexed only by blockIdx, where one grid
 * can use the other grid's shared-memory staging buffer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda.h>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

namespace {

constexpr int kFlows = 2;
constexpr int kBlocks = 1;
constexpr int kThreads = 64;
constexpr int kRounds = 8;
constexpr size_t kBytesPerFlow = 4 * 1024 * 1024;
constexpr int kOverlapMilliseconds = 100;

struct LaunchGate {
    int active;
    int max_active;
};

__global__ void concurrent_tma_put(const unsigned char *source, unsigned char *destination,
                                   size_t bytes, int peer, int smem_size,
                                   unsigned long long hold_cycles, LaunchGate *gate) {
    extern __shared__ char nvshmem_smem[];

    size_t block_offset = blockIdx.x * (bytes / gridDim.x);
    size_t block_bytes = bytes / gridDim.x;

    nvshmemx_give_smem(nvshmem_smem, smem_size);
    __syncthreads();

    if (threadIdx.x == 0) {
        int active = atomicAdd(&gate->active, 1) + 1;
        atomicMax(&gate->max_active, active);
        unsigned long long begin = clock64();
        while (clock64() - begin < hold_cycles) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();

    nvshmemx_putmem_nbi_block(destination + block_offset, source + block_offset, block_bytes, peer);
    nvshmemx_qp_quiet_block(NVSHMEMX_PE_ALL, nullptr, NVSHMEMX_QP_ALL);
    __syncthreads();

    nvshmemx_release_smem();
    __syncthreads();
    if (threadIdx.x == 0) atomicSub(&gate->active, 1);
}

unsigned char pattern_for(int pe, int flow) {
    return static_cast<unsigned char>(1 + pe * 31 + flow * 97);
}

int allreduce_status(int status) {
    int global_status = 0;
    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    if (scratch == nullptr) {
        fprintf(stderr, "[PE %d] FAIL: nvshmem_malloc failed for status reduction\n",
                nvshmem_my_pe());
        return 1;
    }

    CUDA_CHECK(cudaMemcpy(scratch, &status, sizeof(status), cudaMemcpyHostToDevice));
    nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, scratch + 1, scratch, 1);
    CUDA_CHECK(
        cudaMemcpy(&global_status, scratch + 1, sizeof(global_status), cudaMemcpyDeviceToHost));
    nvshmem_free(scratch);
    return global_status;
}

}  // namespace

int main(int argc, char **argv) {
    int status = 0;
    bool skipped = false;
    int device = 0;
    int clock_rate = 0;
    cudaDeviceProp properties;
    unsigned char *source = nullptr;
    unsigned char *destination = nullptr;
    LaunchGate *gate = nullptr;
    cudaStream_t streams[kFlows] = {};

    setenv("NVSHMEM_TMA_POLICY", "ENABLE", 1);
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    int previous_pe = (mype - 1 + npes) % npes;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device));

    if (npes < 2) {
        printf("[PE %d] SKIP: concurrent TMA registration requires at least 2 PEs\n", mype);
        goto out;
    }
    if (properties.major < 9) {
        printf(
            "[PE %d] SKIP: concurrent TMA registration requires sm_90+, current device is "
            "sm_%d%d\n",
            mype, properties.major, properties.minor);
        goto out;
    }
    if (!properties.concurrentKernels) {
        printf("[PE %d] SKIP: device does not support concurrent kernels\n", mype);
        goto out;
    }

    {
        constexpr size_t total_bytes = kFlows * kBytesPerFlow;
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        unsigned long long hold_cycles =
            static_cast<unsigned long long>(clock_rate) * kOverlapMilliseconds;
        std::vector<unsigned char> host_destination(total_bytes);

        source = static_cast<unsigned char *>(nvshmem_malloc(total_bytes));
        destination = static_cast<unsigned char *>(nvshmem_malloc(total_bytes));
        if (source == nullptr || destination == nullptr) {
            printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
            status = 1;
            goto cleanup;
        }

        {
            int peer_unreachable = nvshmem_ptr(destination, peer) == nullptr;
            if (allreduce_status(peer_unreachable) != 0) {
                if (peer_unreachable) {
                    printf("[PE %d] SKIP: peer PE %d is not directly P2P reachable\n", mype, peer);
                }
                if (mype == 0) {
                    printf(
                        "SKIP: concurrent TMA registration requires direct P2P reachability "
                        "between every participating PE\n");
                }
                skipped = true;
                goto cleanup;
            }
        }

        CUDA_CHECK(cudaMalloc(&gate, sizeof(*gate)));
        for (int flow = 0; flow < kFlows; flow++) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[flow], cudaStreamNonBlocking));
            CUDA_CHECK(
                cudaMemset(source + flow * kBytesPerFlow, pattern_for(mype, flow), kBytesPerFlow));
        }

        CUDA_CHECK(cudaFuncSetAttribute(concurrent_tma_put,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

        for (int round = 0; round < kRounds; round++) {
            CUDA_CHECK(cudaMemset(gate, 0, sizeof(*gate)));
            CUDA_CHECK(cudaMemset(destination, 0, total_bytes));

            nvshmem_barrier_all();
            for (int flow = 0; flow < kFlows; flow++) {
                concurrent_tma_put<<<kBlocks, kThreads, smem_size, streams[flow]>>>(
                    source + flow * kBytesPerFlow, destination + flow * kBytesPerFlow,
                    kBytesPerFlow, peer, smem_size, hold_cycles, gate);
                CUDA_CHECK(cudaGetLastError());
            }

            for (int flow = 0; flow < kFlows; flow++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[flow]));
            }

            LaunchGate host_gate = {};
            CUDA_CHECK(cudaMemcpy(&host_gate, gate, sizeof(host_gate), cudaMemcpyDeviceToHost));
            if (host_gate.max_active != kFlows || host_gate.active != 0) {
                printf(
                    "[PE %d] FAIL: round %d observed max_active=%d and active=%d, expected "
                    "max_active=%d and active=0\n",
                    mype, round, host_gate.max_active, host_gate.active, kFlows);
                status = 1;
            }

            nvshmem_barrier_all();
            CUDA_CHECK(cudaMemcpy(host_destination.data(), destination, total_bytes,
                                  cudaMemcpyDeviceToHost));

            size_t errors = 0;
            for (int flow = 0; flow < kFlows; flow++) {
                unsigned char expected = pattern_for(previous_pe, flow);
                size_t flow_offset = flow * kBytesPerFlow;
                for (size_t index = 0; index < kBytesPerFlow; index++) {
                    unsigned char found = host_destination[flow_offset + index];
                    if (found != expected) {
                        if (errors < 8) {
                            printf(
                                "[PE %d] FAIL: round %d flow %d byte %zu = 0x%02x, expected "
                                "0x%02x\n",
                                mype, round, flow, index, found, expected);
                        }
                        errors++;
                    }
                }
            }
            if (errors != 0) {
                printf("[PE %d] FAIL: round %d found %zu corrupted bytes\n", mype, round, errors);
                status = 1;
            }
        }

    cleanup:
        for (int flow = 0; flow < kFlows; flow++) {
            if (streams[flow] != nullptr) CUDA_CHECK(cudaStreamDestroy(streams[flow]));
        }
        if (gate != nullptr) CUDA_CHECK(cudaFree(gate));
        if (destination != nullptr) nvshmem_free(destination);
        if (source != nullptr) nvshmem_free(source);
    }

    if (skipped) goto out;

    status = allreduce_status(status);
    if (mype == 0 && status == 0) {
        printf(
            "PASS: concurrent TMA shared-memory registrations remained isolated across %d "
            "rounds\n",
            kRounds);
    }

out:
    finalize_wrapper();
    return status;
}
