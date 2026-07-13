/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "utils.h"

namespace {

constexpr int kOperations = 64;

__host__ __device__ constexpr unsigned long long operation_value(int source_pe, int operation) {
    return static_cast<unsigned long long>(source_pe + 1) * 1000 + operation + 1;
}

// Issue enough operations to cycle every default QP, then verify RMA and AMO routing.
__global__ void proxy_multinic_kernel(unsigned long long *target, uint64_t *counters, int mype,
                                      int npes) {
    int next_pe = (mype + 1) % npes;

    for (int operation = 0; operation < kOperations; ++operation) {
        size_t index = static_cast<size_t>(mype) * kOperations + operation;
        nvshmem_ulonglong_p(&target[index], operation_value(mype, operation), next_pe);
    }
    nvshmem_quiet();

    for (int operation = 0; operation < kOperations; ++operation) {
        size_t index = static_cast<size_t>(mype) * kOperations + operation;
        nvshmemx_signal_op(&counters[index], 1, NVSHMEM_SIGNAL_ADD, next_pe);
    }
    nvshmem_quiet();
}

}  // namespace

int main(int argc, char **argv) {
    int errors = 0;

    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    size_t data_count = static_cast<size_t>(npes) * kOperations;
    size_t target_count = data_count + 1;
    auto *target =
        static_cast<unsigned long long *>(nvshmem_calloc(target_count, sizeof(unsigned long long)));
    auto *counters = static_cast<uint64_t *>(nvshmem_calloc(data_count, sizeof(uint64_t)));
    if (!target || !counters) {
        std::fprintf(stderr, "PE %d failed to allocate symmetric test buffers\n", mype);
        nvshmem_global_exit(1);
    }

    nvshmem_barrier_all();
    proxy_multinic_kernel<<<1, 1>>>(target, counters, mype, npes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int next_pe = (mype + 1) % npes;
    nvshmem_ulonglong_p(&target[data_count], static_cast<unsigned long long>(mype + 1), next_pe);
    nvshmem_quiet();
    nvshmem_barrier_all();

    std::vector<unsigned long long> observed(target_count);
    CUDA_CHECK(cudaMemcpy(observed.data(), target, target_count * sizeof(observed[0]),
                          cudaMemcpyDeviceToHost));
    std::vector<uint64_t> observed_counters(data_count);
    CUDA_CHECK(cudaMemcpy(observed_counters.data(), counters,
                          data_count * sizeof(observed_counters[0]), cudaMemcpyDeviceToHost));

    int previous_pe = (mype + npes - 1) % npes;
    for (int source_pe = 0; source_pe < npes; ++source_pe) {
        for (int operation = 0; operation < kOperations; ++operation) {
            size_t index = static_cast<size_t>(source_pe) * kOperations + operation;
            unsigned long long expected =
                source_pe == previous_pe ? operation_value(source_pe, operation) : 0;
            if (observed[index] != expected) {
                std::fprintf(stderr,
                             "PE %d RMA mismatch at index %zu: expected %llu, observed %llu\n",
                             mype, index, expected, observed[index]);
                ++errors;
            }
        }
    }

    unsigned long long expected_marker = static_cast<unsigned long long>(previous_pe + 1);
    if (observed[data_count] != expected_marker) {
        std::fprintf(stderr, "PE %d host endpoint mismatch: expected %llu, observed %llu\n", mype,
                     expected_marker, observed[data_count]);
        ++errors;
    }

    for (int source_pe = 0; source_pe < npes; ++source_pe) {
        for (int operation = 0; operation < kOperations; ++operation) {
            size_t index = static_cast<size_t>(source_pe) * kOperations + operation;
            uint64_t expected = source_pe == previous_pe ? 1 : 0;
            if (observed_counters[index] != expected) {
                std::fprintf(stderr,
                             "PE %d AMO mismatch at index %zu: expected %llu, observed %llu\n",
                             mype, index, static_cast<unsigned long long>(expected),
                             static_cast<unsigned long long>(observed_counters[index]));
                ++errors;
            }
        }
    }

    nvshmem_barrier_all();
    nvshmem_free(counters);
    nvshmem_free(target);
    finalize_wrapper();
    return errors == 0 ? 0 : 1;
}
