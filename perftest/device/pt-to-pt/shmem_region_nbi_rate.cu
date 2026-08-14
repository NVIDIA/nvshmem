/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"

namespace {

enum class region_mode : int { no_region, no_hint_region, batch_rma };
enum class operation_scope : int { thread, block };

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

const char *scope_name(operation_scope scope) {
    return scope == operation_scope::thread ? "thread" : "block";
}

template <operation_scope SCOPE, bool IS_WRITE>
__device__ void issue_operation_group(char *destination, const char *source, size_t bytes, int peer,
                                      size_t operations_per_region) {
    if constexpr (SCOPE == operation_scope::thread) {
        if (threadIdx.x != 0) {
            return;
        }
    }

    const size_t block_offset = blockIdx.x * operations_per_region * bytes;
    for (size_t operation = 0; operation < operations_per_region; ++operation) {
        const size_t offset = block_offset + operation * bytes;
        if constexpr (SCOPE == operation_scope::thread) {
            if constexpr (IS_WRITE) {
                nvshmem_putmem_nbi(destination + offset, source + offset, bytes, peer);
            } else {
                nvshmem_getmem_nbi(destination + offset, source + offset, bytes, peer);
            }
        } else {
            if constexpr (IS_WRITE) {
                nvshmemx_putmem_nbi_block(destination + offset, source + offset, bytes, peer);
            } else {
                nvshmemx_getmem_nbi_block(destination + offset, source + offset, bytes, peer);
            }
        }
    }
}

template <operation_scope SCOPE, region_mode MODE, bool IS_WRITE>
__global__ void issue_regions(char *destination, const char *source, size_t bytes, int peer,
                              size_t regions, size_t operations_per_region, int *status) {
    if constexpr (MODE == region_mode::no_region && SCOPE == operation_scope::thread) {
        if (threadIdx.x == 0) {
            for (size_t region_index = 0; region_index < regions; ++region_index) {
                issue_operation_group<SCOPE, IS_WRITE>(destination, source, bytes, peer,
                                                       operations_per_region);
                nvshmem_quiet();
            }
        }
    } else {
        for (size_t region_index = 0; region_index < regions; ++region_index) {
            nvshmemx_region_handle_t handle = 0;
            if constexpr (MODE != region_mode::no_region) {
                nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
                attrs.hints = MODE == region_mode::batch_rma ? NVSHMEMX_REGION_HINT_BATCH_RMA
                                                             : NVSHMEMX_REGION_HINT_NONE;
                if (nvshmemx_region_start_block(&handle, &attrs) != NVSHMEMX_SUCCESS) {
                    if (threadIdx.x == 0) {
                        atomicCAS(status, 0, 1);
                    }
                    return;
                }
            }

            issue_operation_group<SCOPE, IS_WRITE>(destination, source, bytes, peer,
                                                   operations_per_region);
            __syncthreads();

            if constexpr (MODE != region_mode::no_region) {
                if (nvshmemx_region_stop_block(handle) != NVSHMEMX_SUCCESS) {
                    if (threadIdx.x == 0) {
                        atomicCAS(status, 0, 2);
                    }
                    return;
                }
            }

            if (threadIdx.x == 0) {
                nvshmem_quiet();
            }
            __syncthreads();
        }
    }
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

bool verify(char *destination, size_t bytes, int mype, int peer) {
    const size_t extent = bytes * region_ops * num_blocks;
    const unsigned char expected = static_cast<unsigned char>(peer + 1);
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

void print_results(operation_scope scope, region_mode mode, rate_results &results) {
    const char *operation = dir.type == WRITE ? "put" : "get";
    const std::string subjob = std::string(operation) + '_' + scope_name(scope) + '_' +
                               mode_name(mode) + '_' + std::to_string(region_ops) + "ops_" +
                               std::to_string(num_blocks) + "ctas";
    print_basic_table("shmem_region_nbi_rate", subjob.c_str(), "usec_per_op", "us", '-',
                      results.sizes.data(), results.usec_per_op.data(), results.sizes.size(),
                      results.usec_stats.data());
    print_basic_table("shmem_region_nbi_rate", subjob.c_str(), "logical_op_rate", "MMOPS", '+',
                      results.sizes.data(), results.mops.data(), results.sizes.size(),
                      results.mops_stats.data());
    print_basic_table("shmem_region_nbi_rate", subjob.c_str(), "payload_bandwidth", "GB/s", '+',
                      results.sizes.data(), results.bandwidth.data(), results.sizes.size(),
                      results.bandwidth_stats.data());
}

template <operation_scope SCOPE, region_mode MODE, bool IS_WRITE>
void run_mode(char *destination, const char *source, size_t allocation_size, int mype, int peer,
              int *device_status, cudaEvent_t start, cudaEvent_t stop, size_t entries,
              bool &valid) {
    rate_results results{std::vector<uint64_t>(entries),     std::vector<double>(entries),
                         std::vector<double>(entries),       std::vector<double>(entries),
                         std::vector<perf_stats_t>(entries), std::vector<perf_stats_t>(entries),
                         std::vector<perf_stats_t>(entries)};

    size_t entry = 0;
    for (size_t bytes = min_size; bytes <= max_size;) {
        results.sizes[entry] = bytes;

        nvshmem_barrier_all();
        CUDA_CHECK(cudaMemset(device_status, 0, sizeof(*device_status)));
        issue_regions<SCOPE, MODE, IS_WRITE><<<static_cast<unsigned int>(num_blocks),
                                               static_cast<unsigned int>(threads_per_block)>>>(
            destination, source, bytes, peer, warmup_iters, region_ops, device_status);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int host_status = 0;
        CUDA_CHECK(
            cudaMemcpy(&host_status, device_status, sizeof(host_status), cudaMemcpyDeviceToHost));
        valid &= host_status == 0;
        nvshmem_barrier_all();
        CUDA_CHECK(cudaMemset(destination, 0, allocation_size));
        nvshmem_barrier_all();

        for (size_t repetition = 0; repetition < repetitions; ++repetition) {
            nvshmem_barrier_all();
            CUDA_CHECK(cudaMemset(device_status, 0, sizeof(*device_status)));
            CUDA_CHECK(cudaEventRecord(start));
            issue_regions<SCOPE, MODE, IS_WRITE><<<static_cast<unsigned int>(num_blocks),
                                                   static_cast<unsigned int>(threads_per_block)>>>(
                destination, source, bytes, peer, iters, region_ops, device_status);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaMemcpy(&host_status, device_status, sizeof(host_status),
                                  cudaMemcpyDeviceToHost));
            valid &= host_status == 0;

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            const double logical_ops = static_cast<double>(iters) *
                                       static_cast<double>(region_ops) *
                                       static_cast<double>(num_blocks);
            const double usec_per_op = elapsed_ms * 1000.0 / logical_ops;
            const double rate = 1.0 / usec_per_op;
            const double bw = static_cast<double>(bytes) * rate / 1000.0;
            results.usec_per_op[entry] = usec_per_op;
            results.mops[entry] = rate;
            results.bandwidth[entry] = bw;
            perf_stats_add(results.usec_stats[entry], usec_per_op);
            perf_stats_add(results.mops_stats[entry], rate);
            perf_stats_add(results.bandwidth_stats[entry], bw);
            nvshmem_barrier_all();
        }

        valid &= verify(destination, bytes, mype, peer);
        ++entry;
        if (bytes > max_size / step_factor) {
            break;
        }
        bytes *= step_factor;
    }

    if (mype == 0 && valid) {
        print_results(SCOPE, MODE, results);
    }
}

template <operation_scope SCOPE, bool IS_WRITE>
bool run_benchmark(char *destination, const char *source, size_t allocation_size, int mype,
                   int peer, int *device_status, cudaEvent_t start, cudaEvent_t stop) {
    bool valid = true;
    const size_t entries = size_count();
    run_mode<SCOPE, region_mode::no_region, IS_WRITE>(destination, source, allocation_size, mype,
                                                      peer, device_status, start, stop, entries,
                                                      valid);
    run_mode<SCOPE, region_mode::no_hint_region, IS_WRITE>(destination, source, allocation_size,
                                                           mype, peer, device_status, start, stop,
                                                           entries, valid);
    run_mode<SCOPE, region_mode::batch_rma, IS_WRITE>(destination, source, allocation_size, mype,
                                                      peer, device_status, start, stop, entries,
                                                      valid);
    return valid;
}

template <bool IS_WRITE>
bool run_selected_scopes(char *destination, const char *source, size_t allocation_size, int mype,
                         int peer, int *device_status, cudaEvent_t start, cudaEvent_t stop) {
    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            return run_benchmark<operation_scope::thread, IS_WRITE>(
                destination, source, allocation_size, mype, peer, device_status, start, stop);
        case NVSHMEM_BLOCK:
            return run_benchmark<operation_scope::block, IS_WRITE>(
                destination, source, allocation_size, mype, peer, device_status, start, stop);
        case NVSHMEM_ALL_SCOPES: {
            bool valid = run_benchmark<operation_scope::thread, IS_WRITE>(
                destination, source, allocation_size, mype, peer, device_status, start, stop);
            valid &= run_benchmark<operation_scope::block, IS_WRITE>(
                destination, source, allocation_size, mype, peer, device_status, start, stop);
            return valid;
        }
        default:
            std::fprintf(stderr, "This benchmark supports thread and block scope\n");
            return false;
    }
}

}  // namespace

int main(int argc, char **argv) {
    read_args(argc, argv);
    const bool invalid_extent =
        min_size == 0 || max_size == 0 ||
        region_ops > std::numeric_limits<size_t>::max() / max_size ||
        num_blocks > std::numeric_limits<size_t>::max() / (max_size * region_ops);
    if (invalid_extent || step_factor < 2 || iters == 0 || num_blocks == 0 ||
        threads_per_block == 0 || num_blocks > std::numeric_limits<unsigned int>::max() ||
        threads_per_block > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr,
                     "sizes, iterations, region operations, CTAs, and threads must be positive and "
                     "in range\n");
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
    const int peer = mype ^ 1;
    print_device_uuid_and_peer(mype, peer);

    const size_t allocation_size = max_size * region_ops * num_blocks;
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

    int *device_status = nullptr;
    CUDA_CHECK(cudaMalloc(&device_status, sizeof(*device_status)));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    bool valid = dir.type == WRITE
                     ? run_selected_scopes<true>(destination, source, allocation_size, mype, peer,
                                                 device_status, start, stop)
                     : run_selected_scopes<false>(destination, source, allocation_size, mype, peer,
                                                  device_status, start, stop);

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(device_status));
    nvshmem_free(destination);
    nvshmem_free(source);
    finalize_wrapper();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
