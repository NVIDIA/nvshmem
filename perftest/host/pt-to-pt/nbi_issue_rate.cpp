/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "utils.h"

namespace {

struct rate_results {
    std::vector<uint64_t> sizes;
    std::vector<double> usec_per_op;
    std::vector<double> mops;
    std::vector<double> bandwidth;
    std::vector<perf_stats_t> usec_stats;
    std::vector<perf_stats_t> mops_stats;
    std::vector<perf_stats_t> bandwidth_stats;
};

size_t size_count() {
    size_t count = 0;
    for (size_t size = min_size; size <= max_size;) {
        ++count;
        if (size > max_size / step_factor) {
            break;
        }
        size *= step_factor;
    }
    return count;
}

template <bool IS_WRITE>
void issue_nbi(char *destination, const char *source, size_t bytes, int peer, size_t operations) {
    for (size_t operation = 0; operation < operations; ++operation) {
        if constexpr (IS_WRITE) {
            nvshmem_putmem_nbi(destination, source, bytes, peer);
        } else {
            nvshmem_getmem_nbi(destination, source, bytes, peer);
        }
    }
}

template <bool IS_WRITE>
double measure(char *destination, const char *source, size_t bytes, int peer, size_t operations) {
    const auto start = std::chrono::steady_clock::now();
    issue_nbi<IS_WRITE>(destination, source, bytes, peer, operations);
    const auto stop = std::chrono::steady_clock::now();
    nvshmem_quiet();
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

template <bool IS_WRITE>
bool verify(char *destination, size_t bytes, int mype) {
    const bool verify_here = IS_WRITE ? mype == 1 : mype == 0;
    if (!verify_here) {
        return true;
    }

    const unsigned char expected = IS_WRITE ? 1 : 2;
    std::vector<unsigned char> host_data(bytes);
    CUDA_CHECK(cudaMemcpy(host_data.data(), destination, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; ++i) {
        if (host_data[i] != expected) {
            std::fprintf(stderr, "PE %d: verification failed at byte %zu: got %u, expected %u\n",
                         mype, i, host_data[i], expected);
            return false;
        }
    }
    return true;
}

template <bool IS_WRITE>
void print_results(rate_results &results) {
    const char *operation = IS_WRITE ? "put" : "get";
    print_basic_table("nbi_issue_rate", operation, "usec_per_op", "us", '-', results.sizes.data(),
                      results.usec_per_op.data(), results.sizes.size(), results.usec_stats.data());
    print_basic_table("nbi_issue_rate", operation, "logical_op_rate", "MMOPS", '+',
                      results.sizes.data(), results.mops.data(), results.sizes.size(),
                      results.mops_stats.data());
    print_basic_table("nbi_issue_rate", operation, "payload_bandwidth", "GB/s", '+',
                      results.sizes.data(), results.bandwidth.data(), results.sizes.size(),
                      results.bandwidth_stats.data());
}

template <bool IS_WRITE>
bool run_benchmark(char *destination, const char *source, int mype, rate_results &results) {
    bool valid = true;
    size_t entry = 0;
    for (size_t bytes = min_size; bytes <= max_size;) {
        results.sizes[entry] = bytes;

        nvshmem_barrier_all();
        if (mype == 0) {
            issue_nbi<IS_WRITE>(destination, source, bytes, 1, warmup_iters);
            nvshmem_quiet();
        }
        nvshmem_barrier_all();
        CUDA_CHECK(cudaMemset(destination, 0, max_size));
        nvshmem_barrier_all();

        for (size_t repetition = 0; repetition < repetitions; ++repetition) {
            nvshmem_barrier_all();
            if (mype == 0) {
                const double elapsed_us = measure<IS_WRITE>(destination, source, bytes, 1, iters);
                const double usec_per_op = elapsed_us / static_cast<double>(iters);
                const double rate = 1.0 / usec_per_op;
                const double bw = static_cast<double>(bytes) * rate / 1000.0;
                results.usec_per_op[entry] = usec_per_op;
                results.mops[entry] = rate;
                results.bandwidth[entry] = bw;
                perf_stats_add(results.usec_stats[entry], usec_per_op);
                perf_stats_add(results.mops_stats[entry], rate);
                perf_stats_add(results.bandwidth_stats[entry], bw);
            }
            nvshmem_barrier_all();
        }

        valid &= verify<IS_WRITE>(destination, bytes, mype);
        ++entry;
        if (bytes > max_size / step_factor) {
            break;
        }
        bytes *= step_factor;
    }

    if (mype == 0) {
        print_results<IS_WRITE>(results);
    }
    return valid;
}

}  // namespace

int main(int argc, char **argv) {
    read_args(argc, argv);
    if (min_size == 0 || step_factor < 2 || iters == 0) {
        std::fprintf(stderr, "min_size and iters must be positive; step must be at least two\n");
        return EXIT_FAILURE;
    }

    init_wrapper(&argc, &argv);
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes != 2) {
        std::fprintf(stderr, "This benchmark requires exactly two processes\n");
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    char *source = static_cast<char *>(nvshmem_malloc(max_size));
    char *destination = static_cast<char *>(nvshmem_malloc(max_size));
    if (source == nullptr || destination == nullptr) {
        std::fprintf(stderr, "PE %d: symmetric allocation failed\n", mype);
        if (source != nullptr) {
            nvshmem_free(source);
        }
        if (destination != nullptr) {
            nvshmem_free(destination);
        }
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemset(source, mype + 1, max_size));
    CUDA_CHECK(cudaMemset(destination, 0, max_size));

    const size_t entries = size_count();
    rate_results results{std::vector<uint64_t>(entries),     std::vector<double>(entries),
                         std::vector<double>(entries),       std::vector<double>(entries),
                         std::vector<perf_stats_t>(entries), std::vector<perf_stats_t>(entries),
                         std::vector<perf_stats_t>(entries)};

    bool valid = dir.type == WRITE ? run_benchmark<true>(destination, source, mype, results)
                                   : run_benchmark<false>(destination, source, mype, results);

    nvshmem_free(destination);
    nvshmem_free(source);
    finalize_wrapper();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
