/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"

namespace {

enum class region_mode { no_region, no_hint_region, batch_rma };

struct rate_results {
    std::vector<uint64_t> sizes;
    std::vector<double> usec_per_op;
    std::vector<double> mops;
    std::vector<double> bandwidth;
    std::vector<perf_stats_t> usec_stats;
    std::vector<perf_stats_t> mops_stats;
    std::vector<perf_stats_t> bandwidth_stats;
};

const char *mode_name(region_mode mode) {
    switch (mode) {
        case region_mode::no_region:
            return "no_region";
        case region_mode::no_hint_region:
            return "no_hint_region";
        case region_mode::batch_rma:
            return "batch_rma";
    }
    return "unknown";
}

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

template <region_mode MODE, bool IS_WRITE>
int issue_regions(char *destination, const char *source, size_t bytes, int peer, size_t regions) {
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints =
        MODE == region_mode::batch_rma ? NVSHMEMX_REGION_HINT_BATCH_RMA : NVSHMEMX_REGION_HINT_NONE;

    for (size_t region_index = 0; region_index < regions; ++region_index) {
        nvshmemx_region_handle_t handle = 0;
        if constexpr (MODE != region_mode::no_region) {
            const int status = nvshmemx_region_start(&handle, &attrs);
            if (status != NVSHMEMX_SUCCESS) {
                return status;
            }
        }

        for (size_t operation = 0; operation < region_ops; ++operation) {
            const size_t offset = operation * bytes;
            if constexpr (IS_WRITE) {
                nvshmem_putmem_nbi(destination + offset, source + offset, bytes, peer);
            } else {
                nvshmem_getmem_nbi(destination + offset, source + offset, bytes, peer);
            }
        }

        if constexpr (MODE != region_mode::no_region) {
            const int status = nvshmemx_region_stop(handle);
            if (status != NVSHMEMX_SUCCESS) {
                return status;
            }
        }

        nvshmem_quiet();
    }
    return NVSHMEMX_SUCCESS;
}

template <region_mode MODE, bool IS_WRITE>
int measure(char *destination, const char *source, size_t bytes, int peer, size_t regions,
            double *elapsed_us) {
    const auto start = std::chrono::steady_clock::now();
    const int status = issue_regions<MODE, IS_WRITE>(destination, source, bytes, peer, regions);
    const auto stop = std::chrono::steady_clock::now();
    *elapsed_us = std::chrono::duration<double, std::micro>(stop - start).count();
    return status;
}

template <bool IS_WRITE>
bool verify(char *destination, size_t bytes, int mype) {
    const bool verify_here = IS_WRITE ? mype == 1 : mype == 0;
    if (!verify_here) {
        return true;
    }

    const size_t extent = bytes * region_ops;
    const unsigned char expected = IS_WRITE ? 1 : 2;
    std::vector<unsigned char> host_data(extent);
    CUDA_CHECK(cudaMemcpy(host_data.data(), destination, extent, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < extent; ++i) {
        if (host_data[i] != expected) {
            std::fprintf(stderr, "PE %d: verification failed at byte %zu: got %u, expected %u\n",
                         mype, i, host_data[i], expected);
            return false;
        }
    }
    return true;
}

template <region_mode MODE, bool IS_WRITE>
void print_results(rate_results &results) {
    const char *operation = IS_WRITE ? "put" : "get";
    const std::string subjob =
        std::string(operation) + '_' + mode_name(MODE) + '_' + std::to_string(region_ops) + "ops";
    print_basic_table("region_nbi_rate", subjob.c_str(), "usec_per_op", "us", '-',
                      results.sizes.data(), results.usec_per_op.data(), results.sizes.size(),
                      results.usec_stats.data());
    print_basic_table("region_nbi_rate", subjob.c_str(), "logical_op_rate", "MMOPS", '+',
                      results.sizes.data(), results.mops.data(), results.sizes.size(),
                      results.mops_stats.data());
    print_basic_table("region_nbi_rate", subjob.c_str(), "payload_bandwidth", "GB/s", '+',
                      results.sizes.data(), results.bandwidth.data(), results.sizes.size(),
                      results.bandwidth_stats.data());
}

template <region_mode MODE, bool IS_WRITE>
void run_mode(char *destination, const char *source, size_t allocation_size, int mype,
              size_t entries, bool &valid) {
    rate_results results{std::vector<uint64_t>(entries),     std::vector<double>(entries),
                         std::vector<double>(entries),       std::vector<double>(entries),
                         std::vector<perf_stats_t>(entries), std::vector<perf_stats_t>(entries),
                         std::vector<perf_stats_t>(entries)};

    size_t entry = 0;
    for (size_t bytes = min_size; bytes <= max_size;) {
        results.sizes[entry] = bytes;

        nvshmem_barrier_all();
        if (mype == 0) {
            const int status =
                issue_regions<MODE, IS_WRITE>(destination, source, bytes, 1, warmup_iters);
            if (status != NVSHMEMX_SUCCESS) {
                std::fprintf(stderr, "region warmup failed with status %d\n", status);
                valid = false;
            }
        }
        nvshmem_barrier_all();
        CUDA_CHECK(cudaMemset(destination, 0, allocation_size));
        nvshmem_barrier_all();

        for (size_t repetition = 0; repetition < repetitions; ++repetition) {
            nvshmem_barrier_all();
            if (mype == 0 && valid) {
                double elapsed_us = 0.0;
                const int status =
                    measure<MODE, IS_WRITE>(destination, source, bytes, 1, iters, &elapsed_us);
                if (status != NVSHMEMX_SUCCESS) {
                    std::fprintf(stderr, "region measurement failed with status %d\n", status);
                    valid = false;
                } else {
                    const double logical_ops =
                        static_cast<double>(iters) * static_cast<double>(region_ops);
                    const double usec_per_op = elapsed_us / logical_ops;
                    const double rate = 1.0 / usec_per_op;
                    const double bw = static_cast<double>(bytes) * rate / 1000.0;
                    results.usec_per_op[entry] = usec_per_op;
                    results.mops[entry] = rate;
                    results.bandwidth[entry] = bw;
                    perf_stats_add(results.usec_stats[entry], usec_per_op);
                    perf_stats_add(results.mops_stats[entry], rate);
                    perf_stats_add(results.bandwidth_stats[entry], bw);
                }
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

    if (mype == 0 && valid) {
        print_results<MODE, IS_WRITE>(results);
    }
}

template <bool IS_WRITE>
bool run_benchmark(char *destination, const char *source, size_t allocation_size, int mype) {
    bool valid = true;
    const size_t entries = size_count();
    run_mode<region_mode::no_region, IS_WRITE>(destination, source, allocation_size, mype, entries,
                                               valid);
    run_mode<region_mode::no_hint_region, IS_WRITE>(destination, source, allocation_size, mype,
                                                    entries, valid);
    run_mode<region_mode::batch_rma, IS_WRITE>(destination, source, allocation_size, mype, entries,
                                               valid);
    return valid;
}

}  // namespace

int main(int argc, char **argv) {
    read_args(argc, argv);
    if (min_size == 0 || step_factor < 2 || iters == 0 ||
        region_ops > std::numeric_limits<size_t>::max() / max_size) {
        std::fprintf(stderr,
                     "sizes, iterations, and region operations must describe a nonempty buffer; "
                     "step must be at least two\n");
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

    const size_t allocation_size = max_size * region_ops;
    char *source = static_cast<char *>(nvshmem_malloc(allocation_size));
    char *destination = static_cast<char *>(nvshmem_malloc(allocation_size));
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

    CUDA_CHECK(cudaMemset(source, mype + 1, allocation_size));
    CUDA_CHECK(cudaMemset(destination, 0, allocation_size));

    bool valid = dir.type == WRITE
                     ? run_benchmark<true>(destination, source, allocation_size, mype)
                     : run_benchmark<false>(destination, source, allocation_size, mype);

    nvshmem_free(destination);
    nvshmem_free(source);
    finalize_wrapper();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
