/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"

namespace {

// Models a multi-field halo exchange without requiring an application-side packing buffer.
enum class exchange_mode { scalar, batch_rma };

struct exchange_results {
    std::vector<uint64_t> sizes;
    std::vector<double> usec_per_step;
    std::vector<double> message_rate;
    std::vector<double> bandwidth;
    std::vector<perf_stats_t> usec_stats;
    std::vector<perf_stats_t> message_rate_stats;
    std::vector<perf_stats_t> bandwidth_stats;
};

const char *mode_name(exchange_mode mode) {
    return mode == exchange_mode::scalar ? "scalar" : "batch_rma";
}

int neighbor_for(int pe, int npes, int direction) {
    if (npes == 2) {
        return pe ^ 1;
    }
    return direction == 0 ? (pe + npes - 1) % npes : (pe + 1) % npes;
}

size_t field_slot(int direction, size_t field_index, size_t fields_per_neighbor) {
    return static_cast<size_t>(direction) * fields_per_neighbor + field_index;
}

size_t inbox_slot(int sender, int direction, size_t field_index, int neighbor_count,
                  size_t fields_per_neighbor) {
    return (static_cast<size_t>(sender) * neighbor_count + direction) * fields_per_neighbor +
           field_index;
}

unsigned char field_value(int sender, int direction, size_t field_index) {
    constexpr size_t nonzero_byte_values = std::numeric_limits<unsigned char>::max();
    return static_cast<unsigned char>(
        (static_cast<size_t>(sender) + direction + field_index) % nonzero_byte_values + 1);
}

bool checked_multiply(size_t lhs, size_t rhs, size_t *product) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *product = lhs * rhs;
    return true;
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

template <exchange_mode MODE>
int exchange_halos(char *inbox, const char *fields, size_t field_stride, size_t bytes, int mype,
                   int npes, int neighbor_count, size_t steps) {
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;

    for (size_t step = 0; step < steps; ++step) {
        nvshmemx_region_handle_t handle = 0;
        if constexpr (MODE == exchange_mode::batch_rma) {
            const int status = nvshmemx_region_start(&handle, &attrs);
            if (status != NVSHMEMX_SUCCESS) {
                return status;
            }
        }

        for (int direction = 0; direction < neighbor_count; ++direction) {
            const int peer = neighbor_for(mype, npes, direction);
            for (size_t field_index = 0; field_index < region_ops; ++field_index) {
                const size_t local_slot = field_slot(direction, field_index, region_ops);
                const size_t remote_slot =
                    inbox_slot(mype, direction, field_index, neighbor_count, region_ops);
                nvshmem_putmem_nbi(inbox + remote_slot * bytes, fields + local_slot * field_stride,
                                   bytes, peer);
            }
        }

        if constexpr (MODE == exchange_mode::batch_rma) {
            const int status = nvshmemx_region_stop(handle);
            if (status != NVSHMEMX_SUCCESS) {
                return status;
            }
        }
        nvshmem_quiet();
    }
    return NVSHMEMX_SUCCESS;
}

bool verify(const char *inbox, size_t allocation_size, size_t bytes, int mype, int npes,
            int neighbor_count) {
    std::vector<unsigned char> actual(allocation_size);
    std::vector<unsigned char> expected(allocation_size);
    CUDA_CHECK(cudaMemcpy(actual.data(), inbox, allocation_size, cudaMemcpyDeviceToHost));

    for (int sender = 0; sender < npes; ++sender) {
        for (int direction = 0; direction < neighbor_count; ++direction) {
            if (neighbor_for(sender, npes, direction) != mype) {
                continue;
            }
            for (size_t field_index = 0; field_index < region_ops; ++field_index) {
                const size_t slot =
                    inbox_slot(sender, direction, field_index, neighbor_count, region_ops);
                const size_t first_byte = slot * bytes;
                std::fill(expected.begin() + first_byte, expected.begin() + first_byte + bytes,
                          field_value(sender, direction, field_index));
            }
        }
    }

    for (size_t byte = 0; byte < allocation_size; ++byte) {
        if (actual[byte] != expected[byte]) {
            std::fprintf(stderr, "PE %d: verification failed at byte %zu: got %u, expected %u\n",
                         mype, byte, actual[byte], expected[byte]);
            return false;
        }
    }
    return true;
}

template <exchange_mode MODE>
void print_results(exchange_results &results, int neighbor_count) {
    const std::string subjob = std::string(mode_name(MODE)) + '_' + std::to_string(region_ops) +
                               "fields_" + std::to_string(neighbor_count) + "neighbors";
    print_basic_table("halo_exchange_rate", subjob.c_str(), "usec_per_step", "us", '-',
                      results.sizes.data(), results.usec_per_step.data(), results.sizes.size(),
                      results.usec_stats.data());
    print_basic_table("halo_exchange_rate", subjob.c_str(), "per_pe_message_rate", "MMOPS", '+',
                      results.sizes.data(), results.message_rate.data(), results.sizes.size(),
                      results.message_rate_stats.data());
    print_basic_table("halo_exchange_rate", subjob.c_str(), "per_pe_payload_bandwidth", "GB/s", '+',
                      results.sizes.data(), results.bandwidth.data(), results.sizes.size(),
                      results.bandwidth_stats.data());
}

template <exchange_mode MODE>
bool run_mode(char *inbox, const char *fields, size_t field_stride, size_t allocation_size,
              int mype, int npes, int neighbor_count) {
    const size_t entries = size_count();
    exchange_results results{std::vector<uint64_t>(entries),     std::vector<double>(entries),
                             std::vector<double>(entries),       std::vector<double>(entries),
                             std::vector<perf_stats_t>(entries), std::vector<perf_stats_t>(entries),
                             std::vector<perf_stats_t>(entries)};

    size_t entry = 0;
    for (size_t bytes = min_size; bytes <= max_size;) {
        results.sizes[entry] = bytes;
        nvshmem_barrier_all();
        const int warmup_status = exchange_halos<MODE>(inbox, fields, field_stride, bytes, mype,
                                                       npes, neighbor_count, warmup_iters);
        if (warmup_status != NVSHMEMX_SUCCESS) {
            std::fprintf(stderr, "PE %d: halo warmup failed with status %d\n", mype, warmup_status);
            return false;
        }
        nvshmem_barrier_all();
        CUDA_CHECK(cudaMemset(inbox, 0, allocation_size));
        nvshmem_barrier_all();

        for (size_t repetition = 0; repetition < repetitions; ++repetition) {
            nvshmem_barrier_all();
            const auto start = std::chrono::steady_clock::now();
            const int status = exchange_halos<MODE>(inbox, fields, field_stride, bytes, mype, npes,
                                                    neighbor_count, iters);
            const auto stop = std::chrono::steady_clock::now();
            if (status != NVSHMEMX_SUCCESS) {
                std::fprintf(stderr, "PE %d: halo measurement failed with status %d\n", mype,
                             status);
                return false;
            }
            nvshmem_barrier_all();

            if (mype == 0) {
                const double elapsed_us =
                    std::chrono::duration<double, std::micro>(stop - start).count();
                const double usec_per_step = elapsed_us / static_cast<double>(iters);
                const double messages = static_cast<double>(iters) * neighbor_count * region_ops;
                const double message_rate = messages / elapsed_us;
                const double bandwidth = static_cast<double>(bytes) * message_rate / 1000.0;
                results.usec_per_step[entry] = usec_per_step;
                results.message_rate[entry] = message_rate;
                results.bandwidth[entry] = bandwidth;
                perf_stats_add(results.usec_stats[entry], usec_per_step);
                perf_stats_add(results.message_rate_stats[entry], message_rate);
                perf_stats_add(results.bandwidth_stats[entry], bandwidth);
            }
        }

        if (!verify(inbox, allocation_size, bytes, mype, npes, neighbor_count)) {
            return false;
        }
        ++entry;
        if (bytes > max_size / step_factor) {
            break;
        }
        bytes *= step_factor;
    }

    if (mype == 0) {
        print_results<MODE>(results, neighbor_count);
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    read_args(argc, argv);
    if (min_size == 0 || max_size < min_size || step_factor < 2 || iters == 0 || region_ops == 0) {
        std::fprintf(stderr, "sizes, iterations, and fields must be positive and in range\n");
        return EXIT_FAILURE;
    }

    init_wrapper(&argc, &argv);
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        std::fprintf(stderr, "This benchmark requires at least two processes\n");
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    const int neighbor_count = npes == 2 ? 1 : 2;
    print_device_uuid_and_peer(mype, neighbor_for(mype, npes, 0));

    size_t slots = 0;
    size_t allocation_size = 0;
    const bool extent_valid =
        checked_multiply(static_cast<size_t>(npes), static_cast<size_t>(neighbor_count), &slots) &&
        checked_multiply(slots, region_ops, &slots) &&
        checked_multiply(slots, max_size, &allocation_size);
    if (!extent_valid) {
        std::fprintf(stderr, "The requested inbox extent exceeds the supported size\n");
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    size_t fields_per_pe = 0;
    size_t fields_allocation_size = 0;
    const bool fields_extent_valid =
        checked_multiply(static_cast<size_t>(neighbor_count), region_ops, &fields_per_pe) &&
        checked_multiply(fields_per_pe, max_size, &fields_allocation_size);
    if (!fields_extent_valid) {
        std::fprintf(stderr, "The requested field extent exceeds the supported size\n");
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    char *fields = static_cast<char *>(nvshmem_malloc(fields_allocation_size));
    char *inbox = static_cast<char *>(nvshmem_malloc(allocation_size));
    if (fields == nullptr || inbox == nullptr) {
        std::fprintf(stderr, "PE %d: symmetric allocation failed\n", mype);
        if (inbox != nullptr) {
            nvshmem_free(inbox);
        }
        if (fields != nullptr) {
            nvshmem_free(fields);
        }
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    std::vector<unsigned char> host_fields(fields_allocation_size);
    for (int direction = 0; direction < neighbor_count; ++direction) {
        for (size_t field_index = 0; field_index < region_ops; ++field_index) {
            const size_t slot = field_slot(direction, field_index, region_ops);
            std::fill(host_fields.begin() + slot * max_size,
                      host_fields.begin() + (slot + 1) * max_size,
                      field_value(mype, direction, field_index));
        }
    }
    CUDA_CHECK(
        cudaMemcpy(fields, host_fields.data(), fields_allocation_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(inbox, 0, allocation_size));

    bool valid = run_mode<exchange_mode::scalar>(inbox, fields, max_size, allocation_size, mype,
                                                 npes, neighbor_count);
    valid &= run_mode<exchange_mode::batch_rma>(inbox, fields, max_size, allocation_size, mype,
                                                npes, neighbor_count);

    nvshmem_free(inbox);
    nvshmem_free(fields);
    finalize_wrapper();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
