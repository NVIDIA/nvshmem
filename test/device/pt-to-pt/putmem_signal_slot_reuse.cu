/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Check directed peer payloads across signal generations and acknowledged slot reuse. */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "utils.h"

namespace {
constexpr size_t kGuard = 16;
constexpr size_t kMaxBytes = 1048577;

struct Options {
    size_t bytes = 257;
    int slots = 4;
    uint64_t epochs = 32;
    size_t source_offset = 0;
    size_t target_offset = 0;
    uint64_t delay_cycles = 0;
    bool scan_offsets = false;
};

struct FirstFailure {
    unsigned int claimed;
    int source;
    int destination;
    int slot;
    uint64_t epoch;
    size_t offset;
    unsigned int expected;
    unsigned int observed;
};

bool unsigned_argument(const char *text, uint64_t *value) {
    if (!text[0]) return false;
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "64-bit generations required");
    if (errno == ERANGE || *end) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_options(int argc, char **argv, Options *options) {
    if (argc > 1 && std::strcmp(argv[argc - 1], "--all-offsets") == 0) {
        options->scan_offsets = true;
        --argc;
    }
    if (argc == 1) return true;
    if (argc < 4 || argc > 7) return false;
    uint64_t values[] = {0, 0, 0, 0, 0, 0};
    for (int i = 1; i < argc; ++i) {
        if (!unsigned_argument(argv[i], &values[i - 1])) return false;
    }
    if (!values[0] || values[0] > kMaxBytes || values[1] < 2 || values[1] > 16 ||
        values[2] <= 3 * values[1] || values[2] > 1048576 || values[3] > 15 ||
        values[4] > 15 || values[5] > 1000000 ||
        (options->scan_offsets && (values[3] || values[4]))) {
        return false;
    }
    options->bytes = static_cast<size_t>(values[0]);
    options->slots = static_cast<int>(values[1]);
    options->epochs = values[2];
    options->source_offset = static_cast<size_t>(values[3]);
    options->target_offset = static_cast<size_t>(values[4]);
    options->delay_cycles = values[5];
    return true;
}

__device__ unsigned char pattern(int source, int destination, uint64_t epoch, int slot,
                                 size_t offset) {
    uint64_t x = offset ^ (epoch * 0x9e3779b97f4a7c15ULL) ^
                 (uint64_t(source + 1) << 40) ^ (uint64_t(destination + 1) << 24) ^
                 (uint64_t(slot) << 52);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return static_cast<unsigned char>(x ^ (x >> 31));
}

__device__ void check_byte(unsigned char observed, unsigned char expected,
                           unsigned long long *wrong, FirstFailure *failure, int source,
                           int destination, uint64_t epoch, int slot, size_t offset) {
    if (observed == expected) return;
    atomicAdd(wrong, 1ULL);
    if (atomicCAS(&failure->claimed, 0U, 1U) == 0U) {
        failure->source = source;
        failure->destination = destination;
        failure->epoch = epoch;
        failure->slot = slot;
        failure->offset = offset;
        failure->expected = expected;
        failure->observed = observed;
    }
}

__device__ void delay(uint64_t cycles) {
    if (threadIdx.x == 0) {
        uint64_t started = clock64();
        while (clock64() - started < cycles) {}
    }
    __syncthreads();
}

__global__ void slot_reuse(unsigned char *target, unsigned char *source, uint64_t *signals,
                           uint64_t *acks, unsigned long long *wrong, FirstFailure *failure,
                           Options options, size_t stride, int pe, int peers) {
    const size_t source_begin = kGuard + options.source_offset;
    const size_t target_begin = kGuard + options.target_offset;
    for (uint64_t epoch = 1; epoch <= options.epochs; ++epoch) {
        int slot = static_cast<int>((epoch - 1) % options.slots);
        for (int distance = 1; distance < peers; ++distance) {
            int destination = (pe + distance) % peers;
            int sender = (pe - distance + peers) % peers;
            size_t outgoing = static_cast<size_t>(pe) * options.slots + slot;
            size_t incoming = static_cast<size_t>(sender) * options.slots + slot;
            if (threadIdx.x == 0 && epoch > static_cast<uint64_t>(options.slots)) {
                nvshmem_signal_wait_until(acks + static_cast<size_t>(destination) * options.slots + slot,
                                           NVSHMEM_CMP_GE, epoch - options.slots);
            }
            __syncthreads();
            for (size_t i = threadIdx.x; i < options.bytes; i += blockDim.x) {
                source[source_begin + i] = pattern(pe, destination, epoch, slot, i);
            }
            __syncthreads();
            delay(((epoch + pe) & 1) ? options.delay_cycles : 0);
            nvshmemx_putmem_signal_block(target + outgoing * stride + target_begin,
                                         source + source_begin, options.bytes, signals + outgoing,
                                         epoch, NVSHMEM_SIGNAL_SET, destination);
            // Complete every initiating thread's source accesses before the next fill.
            nvshmem_quiet();
            __syncthreads();
            if (threadIdx.x == 0) {
                nvshmem_signal_wait_until(signals + incoming, NVSHMEM_CMP_GE, epoch);
            }
            // The waiter and all payload readers belong to this single CTA.
            __syncthreads();
            delay(((epoch + sender) & 1) ? options.delay_cycles : 0);
            for (size_t i = threadIdx.x; i < stride; i += blockDim.x) {
                unsigned char expected = (i >= target_begin && i < target_begin + options.bytes)
                                             ? pattern(sender, pe, epoch, slot, i - target_begin)
                                             : 0xa5;
                check_byte(target[incoming * stride + i], expected, wrong, failure, sender, pe,
                            epoch, slot, i);
                if (i < source_begin || i >= source_begin + options.bytes) {
                    check_byte(source[i], 0xa5, wrong, failure, pe, destination, epoch, slot, i);
                }
            }
            __syncthreads();
            // An ACK returns the destination slot only after every read and guard check.
            if (threadIdx.x == 0) {
                nvshmemx_signal_op(acks + static_cast<size_t>(pe) * options.slots + slot,
                                   epoch, NVSHMEM_SIGNAL_SET, sender);
                nvshmem_quiet();
            }
            __syncthreads();
        }
    }
}
}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::fprintf(stderr,
                     "Usage: %s [bytes slots epochs [source_offset [target_offset [delay_cycles]]]] "
                     "[--all-offsets]\n"
                     "bytes=1..1048577, slots=2..16, epochs>3*slots and <=1048576, "
                     "offsets=0..15, delay_cycles=0..1000000\n", argv[0]);
        return 2;
    }
    init_wrapper(&argc, &argv);
    int pe = nvshmem_my_pe(), peers = nvshmem_n_pes();
    if (peers < 2 || use_cubin) {
        std::fprintf(stderr, "Slot reuse requires at least two PEs and a linked device library.\n");
        finalize_wrapper();
        return 2;
    }
    // Each slot starts on a 16-byte boundary, so offsets select the exact residue.
    size_t stride = (options.bytes + 2 * kGuard + 15 + 15) & ~size_t(15);
    size_t entries = static_cast<size_t>(peers);
    if (entries > std::numeric_limits<size_t>::max() / options.slots) {
        nvshmem_global_exit(3);
    }
    entries *= options.slots;
    if (entries > std::numeric_limits<size_t>::max() / stride) nvshmem_global_exit(3);
    auto *target = static_cast<unsigned char *>(nvshmem_malloc(entries * stride));
    auto *source = static_cast<unsigned char *>(nvshmem_malloc(stride));
    auto *signals = static_cast<uint64_t *>(nvshmem_malloc(entries * sizeof(uint64_t)));
    auto *acks = static_cast<uint64_t *>(nvshmem_malloc(entries * sizeof(uint64_t)));
    auto *wrong = static_cast<unsigned long long *>(nvshmem_malloc(sizeof(unsigned long long)));
    auto *total = static_cast<unsigned long long *>(nvshmem_malloc(sizeof(unsigned long long)));
    if (!target || !source || !signals || !acks || !wrong || !total) nvshmem_global_exit(4);
    FirstFailure *failure = nullptr, first = {};
    CUDA_CHECK(cudaMalloc(&failure, sizeof(*failure)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int failed = 0;
    int residues = options.scan_offsets ? 16 : 1;
    for (int source_residue = 0; source_residue < residues; ++source_residue) {
        for (int target_residue = 0; target_residue < residues; ++target_residue) {
            Options current = options;
            if (options.scan_offsets) {
                current.source_offset = static_cast<size_t>(source_residue);
                current.target_offset = static_cast<size_t>(target_residue);
            }
            CUDA_CHECK(cudaMemset(failure, 0, sizeof(*failure)));
            CUDA_CHECK(cudaMemset(target, 0xa5, entries * stride));
            CUDA_CHECK(cudaMemset(source, 0xa5, stride));
            CUDA_CHECK(cudaMemset(signals, 0, entries * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(acks, 0, entries * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(wrong, 0, sizeof(*wrong)));
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            void *args[] = {&target, &source, &signals, &acks, &wrong, &failure, &current, &stride, &pe, &peers};
            if (nvshmemx_collective_launch(reinterpret_cast<const void *>(slot_reuse), dim3(1), dim3(256),
                                          args, 0, stream)) nvshmem_global_exit(5);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (nvshmem_ulonglong_sum_reduce(NVSHMEM_TEAM_WORLD, total, wrong, 1)) nvshmem_global_exit(6);
            unsigned long long observed = 0, global = 0;
            CUDA_CHECK(cudaMemcpy(&observed, wrong, sizeof(observed), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&global, total, sizeof(global), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&first, failure, sizeof(first), cudaMemcpyDeviceToHost));
            if (first.claimed) {
                std::fprintf(stderr, "PE %d: first mismatch src=%d dst=%d epoch=%llu slot=%d offset=%zu "
                                     "expected=%u observed=%u\n", pe, first.source, first.destination,
                             static_cast<unsigned long long>(first.epoch), first.slot, first.offset,
                             first.expected, first.observed);
            }
            std::printf("{\"rank\":%d,\"npes\":%d,\"bytes\":%zu,\"slots\":%d,\"epochs\":%llu,"
                        "\"source_offset\":%zu,\"target_offset\":%zu,\"delay_cycles\":%llu,"
                        "\"directed_messages\":%llu,\"wrong_payload_or_guard\":%llu,\"status\":\"%s\"}\n",
                        pe, peers, current.bytes, current.slots,
                        static_cast<unsigned long long>(current.epochs), current.source_offset,
                        current.target_offset, static_cast<unsigned long long>(current.delay_cycles),
                        static_cast<unsigned long long>(peers - 1) * current.epochs, observed,
                        global ? "FAIL" : "PASS");
            failed |= global ? 1 : 0;
            // All device-side quiets finished before the collective reduction. This
            // boundary permits fresh generations and a new address configuration.
            nvshmem_barrier_all();
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(failure));
    nvshmem_free(total);
    nvshmem_free(wrong);
    nvshmem_free(acks);
    nvshmem_free(signals);
    nvshmem_free(source);
    nvshmem_free(target);
    finalize_wrapper();
    return failed;
}
