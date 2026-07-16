/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "coll_test.h"

int main(int argc, char *argv[]) {
    int status = 0;
    int mype;
    size_t size = 1;

    read_args(argc, argv);
    float ms;
    double latency_value;
    perf_stats_t latency_stats = {};
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    int npes = nvshmem_n_pes();
#endif
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    for (size_t iter = 0; iter < warmup_iters; iter++) {
        nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        for (size_t iter = 0; iter < iters; iter++) {
            nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        latency_value = (ms / iters) * 1000;
        perf_stats_add(latency_stats, latency_value);
    }

    if (!mype) {
        print_basic_table("barrier_on_stream", "None", "latency", "us", '-', &size, &latency_value,
                          1, &latency_stats);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    finalize_wrapper();

    return status;
}
