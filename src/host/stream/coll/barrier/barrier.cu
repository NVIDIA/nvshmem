/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "internal/host/util.h"
#include "internal/host/debug.h"
#include "non_abi/device/coll/barrier.cuh"
#include "device/nvshmemx_defines.h"

template <threadgroup_t SCOPE>
__global__ void barrier_on_stream_kernel_threadgroup(nvshmem_team_t team, int in_cuda_graph) {
#ifdef __CUDA_ARCH__
    extern __shared__ char smem[];
    nvshmemx_give_smem(smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    int myidx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (nvshmemi_device_state_d.job_connectivity >= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        nvshmemi_transfer_quiet<SCOPE>(false, NVSHMEMX_PE_ANY, NULL, NVSHMEMX_QP_ALL);
    }
    if (in_cuda_graph) {
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myidx) {
            __threadfence_system();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    nvshmemi_sync_algo_threadgroup<SCOPE>(team);

    if (!myidx) {
        if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY) {
            nvshmemi_transfer_enforce_consistency_at_target(false);
        }
    }
    nvshmemx_release_smem();
#endif
}

template <threadgroup_t SCOPE>
__global__ void sync_on_stream_kernel_threadgroup(nvshmem_team_t team, int in_cuda_graph) {
#ifdef __CUDA_ARCH__
    extern __shared__ char smem[];
    nvshmemx_give_smem(smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    nvshmemi_sync_algo_threadgroup<SCOPE>(team);
    nvshmemx_release_smem();
#endif
}

int nvshmemi_call_barrier_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    int in_cuda_graph = 0;

    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
        in_cuda_graph = 1;
    }

    size_t smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
    if (num_threads_per_block <= 32) {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, smem_size, stream>>>(team, in_cuda_graph);
    } else {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, smem_size, stream>>>(team, in_cuda_graph);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}

int nvshmemi_call_sync_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    int in_cuda_graph = 0;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
        in_cuda_graph = 1;
    }

    size_t smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
    if (num_threads_per_block <= 32) {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, smem_size, stream>>>(team, in_cuda_graph);
    } else {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, smem_size, stream>>>(team, in_cuda_graph);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}
