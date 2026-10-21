/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Shared host/device helpers for the transport-specific tests. Each
 * test defines NVSHMEMX_ENABLE_TRANSPORT_POLICY before including this
 * header. */

#ifndef TRANSPORT_SPECIFIC_COMMON_CUH
#define TRANSPORT_SPECIFIC_COMMON_CUH

#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

namespace ts_test {

/* Tests name the local operand by its put-source spelling; gets use
 * local_dest() for the same memory space. The remote operand is global. */
using LocalSpace = nvshmemx::memory_space::source;
inline constexpr LocalSpace kShared = LocalSpace::shared;
inline constexpr LocalSpace kGlobal = LocalSpace::global;
inline constexpr LocalSpace kGeneric = LocalSpace::generic;
inline constexpr auto kRemoteDest = nvshmemx::memory_space::dest::global;
inline constexpr auto kRemoteSource = nvshmemx::memory_space::source::global;

__host__ __device__ constexpr nvshmemx::memory_space::dest local_dest(LocalSpace local) {
    return local == LocalSpace::shared   ? nvshmemx::memory_space::dest::shared
           : local == LocalSpace::global ? nvshmemx::memory_space::dest::global
                                         : nvshmemx::memory_space::dest::generic;
}
inline constexpr auto kBlock = nvshmemx::threadgroup::block;

inline constexpr int kThreads = 128;
inline constexpr unsigned char kSentinelByte = 0xee;
inline constexpr unsigned char kPoisonByte = 0xa5;
inline constexpr uint64_t kSentinelSignal = 0x5e5e5e5e5e5e5e5eull;
inline constexpr uint64_t kSignalBase = 0x1000;

__host__ __device__ constexpr unsigned char payload_value(int pe, size_t offset) {
    return static_cast<unsigned char>(1 + (pe * 37 + static_cast<int>(offset % 4096) * 13) % 251);
}

__device__ inline void fill_payload_block(unsigned char *buffer, size_t bytes, int pe) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        buffer[i] = payload_value(pe, i);
    }
}

__device__ inline void fill_block(unsigned char *buffer, size_t bytes, unsigned char value) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        buffer[i] = value;
    }
}

__device__ inline void copy_block(unsigned char *dest, const unsigned char *source, size_t bytes) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dest[i] = source[i];
    }
}

__global__ void fill_payload(unsigned char *buffer, size_t bytes, int pe) {
    fill_payload_block(buffer, bytes, pe);
}

/* Kernel status words start as kStatusUnset (run_case fills them with 0xff).
 * Data checks made inside a kernel report kDataMismatch, and a receiver whose
 * signal never arrived reports kReceiveTimeout. */
inline constexpr int kStatusUnset = -1;
inline constexpr int kDataMismatch = -1001;
inline constexpr int kReceiveTimeout = -1002;
inline constexpr uint64_t kReceiveTimeoutNs = 5ull * 1000 * 1000 * 1000;

__device__ inline uint64_t global_time_ns() {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

/* A sender's failure always wins: it is why a concurrent receiver timed out.
 * A sender's success never replaces a receiver's failure. */
__device__ inline void report_sender(int *status, int result) {
    if (threadIdx.x == 0) {
        if (result != NVSHMEMX_SUCCESS) {
            atomicExch(status, result);
        } else {
            atomicCAS(status, kStatusUnset, result);
        }
    }
}

/* A receiver's failure is recorded only while no sender failure is. */
__device__ inline void report_receiver(int *status, int failure) {
    int observed = *reinterpret_cast<volatile int *>(status);
    while (observed == kStatusUnset || observed == NVSHMEMX_SUCCESS) {
        const int previous = atomicCAS(status, observed, failure);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

/* Whether this thread's share of `bytes` holds `pe`'s payload. Each thread
 * checks offsets lane, lane + width, ... so a group covers its whole slice. */
__device__ inline bool payload_matches(const unsigned char *buffer, size_t bytes, int pe,
                                       size_t base, size_t lane, size_t width) {
    const volatile unsigned char *data = buffer;
    bool matches = true;
    for (size_t i = lane; i < bytes; i += width) {
        matches &= data[i] == payload_value(pe, base + i);
    }
    return matches;
}

/*
 * The receiving half of a signalled case, run by its own CTA while the peer
 * sends: wait for `expected` and check the payload at once, before any kernel
 * boundary, barrier, or completion call could hide a signal that overtook its
 * payload. A receiver that never sees the signal reports kReceiveTimeout,
 * which run_case accepts only when the case is skipped.
 */
__device__ inline void receive_and_check(uint64_t *signal, uint64_t expected,
                                         const unsigned char *payload, size_t bytes, int source_pe,
                                         int *status) {
    __shared__ int observed;
    if (threadIdx.x == 0) {
        const uint64_t start = global_time_ns();
        int seen = 0;
        while (!seen && global_time_ns() - start < kReceiveTimeoutNs) {
            seen = nvshmem_signal_fetch(signal) == expected;
        }
        __threadfence();
        observed = seen;
    }
    __syncthreads();
    const bool matches =
        !observed || payload_matches(payload, bytes, source_pe, 0, threadIdx.x, blockDim.x);
    const int mismatch = __syncthreads_or(!matches);
    if (threadIdx.x == 0 && (!observed || mismatch)) {
        report_receiver(status, observed ? kDataMismatch : kReceiveTimeout);
    }
}

__global__ void signal_wait(uint64_t *signal, uint64_t expected, int *status) {
    if (threadIdx.x == 0) {
        const uint64_t observed = nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, expected);
        *status = observed == expected ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
    }
}

inline int read_status(const int *status) {
    int host_status = NVSHMEMX_ERROR_INTERNAL;
    CUDA_CHECK(cudaMemcpy(&host_status, status, sizeof(host_status), cudaMemcpyDeviceToHost));
    return host_status;
}

/* WORLD max-reduction through a symmetric int[2]. A failed reduction reports 1
 * so callers treat it as a failure rather than a pass. */
inline int world_max(int *scratch, int local_value) {
    CUDA_CHECK(cudaMemcpy(scratch, &local_value, sizeof(local_value), cudaMemcpyHostToDevice));
    if (nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, scratch + 1, scratch, 1) != NVSHMEMX_SUCCESS) {
        return 1;
    }
    int global_value = 1;
    CUDA_CHECK(
        cudaMemcpy(&global_value, scratch + 1, sizeof(global_value), cudaMemcpyDeviceToHost));
    return global_value;
}

inline int check_payload(const char *what, const unsigned char *device, size_t bytes, int pe) {
    std::vector<unsigned char> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; i++) {
        if (host[i] != payload_value(pe, i)) {
            fprintf(stderr, "PE %d: %s: payload[%zu] = %u, expected %u\n", nvshmem_my_pe(), what, i,
                    static_cast<unsigned int>(host[i]),
                    static_cast<unsigned int>(payload_value(pe, i)));
            return 1;
        }
    }
    return 0;
}

inline int check_fill(const char *what, const unsigned char *device, size_t bytes,
                      unsigned char value) {
    std::vector<unsigned char> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; i++) {
        if (host[i] != value) {
            fprintf(stderr, "PE %d: %s: byte[%zu] = %u, expected unchanged %u\n", nvshmem_my_pe(),
                    what, i, static_cast<unsigned int>(host[i]), static_cast<unsigned int>(value));
            return 1;
        }
    }
    return 0;
}

inline void set_signal(uint64_t *device, uint64_t value) {
    CUDA_CHECK(cudaMemcpy(device, &value, sizeof(value), cudaMemcpyHostToDevice));
}

inline int check_signal(const char *what, const uint64_t *device, uint64_t expected) {
    uint64_t observed = 0;
    CUDA_CHECK(cudaMemcpy(&observed, device, sizeof(observed), cudaMemcpyDeviceToHost));
    if (observed != expected) {
        fprintf(stderr, "PE %d: %s: signal = 0x%llx, expected 0x%llx\n", nvshmem_my_pe(), what,
                static_cast<unsigned long long>(observed),
                static_cast<unsigned long long>(expected));
        return 1;
    }
    return 0;
}

/* Checks the value directly, then confirms the stable wait API observes it. */
inline int check_signal_and_wait(const char *what, uint64_t *signal, uint64_t expected,
                                 int *status) {
    if (check_signal(what, signal, expected)) {
        return 1;
    }
    CUDA_CHECK(cudaMemset(status, 0xff, sizeof(*status)));
    signal_wait<<<1, 1>>>(signal, expected, status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (read_status(status) != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "PE %d: %s: nvshmem_signal_wait_until failed\n", nvshmem_my_pe(), what);
        return 1;
    }
    return 0;
}

struct world {
    int mype;
    int npes;
    int *scratch; /* symmetric int[2] */
    int *status;  /* device int */
};

/*
 * One route case: every PE resets its buffers, active PEs launch the sender,
 * and all PEs agree on the sender statuses before any receiver inspects data,
 * so a failed sender never leaves a peer verifying (or waiting on) data that
 * was not sent. Inactive PEs only join the barriers.
 *
 * A case that expects success but receives NVSHMEMX_ERROR_NOT_SUPPORTED on any
 * PE is a capability skip on every PE: it is reported and not verified. Any
 * other unexpected status, NVSHMEMX_ERROR_INTERNAL included, is a failure.
 */
inline const char *status_name(int status) {
    return status == kDataMismatch     ? " (data mismatch observed in the kernel)"
           : status == kReceiveTimeout ? " (receiver never observed the signal)"
                                       : "";
}

template <typename Prepare, typename Launch, typename Verify>
int run_case(const world &w, bool active, const char *what, int expected_status, Prepare prepare,
             Launch launch, Verify verify) {
    prepare();
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int status = expected_status;
    if (active) {
        CUDA_CHECK(cudaMemset(w.status, 0xff, sizeof(*w.status)));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        status = read_status(w.status);
    }
    nvshmem_barrier_all();

    const bool local_skipped =
        expected_status == NVSHMEMX_SUCCESS && status == NVSHMEMX_ERROR_NOT_SUPPORTED;
    /* A receiver waits in vain when its sender skipped. */
    const bool any_skipped = world_max(w.scratch, local_skipped) != 0;
    const bool local_failed =
        status != expected_status && !local_skipped && !(any_skipped && status == kReceiveTimeout);
    if (local_failed) {
        fprintf(stderr, "PE %d: %s returned %d%s, expected %d\n", w.mype, what, status,
                status_name(status), expected_status);
    }
    if (world_max(w.scratch, local_failed) != 0) {
        nvshmem_barrier_all();
        return 1;
    }
    if (any_skipped) {
        if (w.mype == 0) {
            printf("%s: SKIP (route capability not supported)\n", what);
        }
        nvshmem_barrier_all();
        return 0;
    }
    const int failures = active ? verify() : 0;
    nvshmem_barrier_all();
    return failures;
}

/*
 * Scope-parametrized data and ordering coverage. Each of the first `issuers`
 * threadgroups of Scope in the CTA moves its own contiguous slice of the
 * buffers, so thread and warp scopes run with several concurrent issuers.
 */
using threadgroup = nvshmemx::threadgroup;

enum class scoped_op {
    put_get,       /* put + quiet, then get_nbi + quiet */
    put_nbi_flush, /* put_nbi + flush, poison the source slice, quiet */
    put_nbi_fence, /* put_nbi + fence + signal_op ADD 1 + quiet */
};

template <threadgroup Scope>
__device__ inline int scope_group_index() {
    if constexpr (Scope == threadgroup::thread) {
        return threadIdx.x;
    } else if constexpr (Scope == threadgroup::warp) {
        return threadIdx.x / warpSize;
    } else {
        return 0;
    }
}

template <threadgroup Scope>
__device__ inline void fill_group(unsigned char *buffer, size_t bytes, unsigned char value) {
    const size_t lane = Scope == threadgroup::thread ? 0
                        : Scope == threadgroup::warp ? threadIdx.x % warpSize
                                                     : threadIdx.x;
    const size_t width = Scope == threadgroup::thread ? 1
                         : Scope == threadgroup::warp ? warpSize
                                                      : blockDim.x;
    for (size_t i = lane; i < bytes; i += width) {
        buffer[i] = value;
    }
    if constexpr (Scope == threadgroup::warp) {
        __syncwarp();
    } else if constexpr (Scope == threadgroup::block) {
        __syncthreads();
    }
}

template <typename A, typename P, threadgroup Scope, LocalSpace Local>
__device__ int scoped_rma(scoped_op which, unsigned char *remote, unsigned char *source,
                          unsigned char *out, uint64_t *signal, size_t bytes, int peer) {
    constexpr auto kLocalDest = local_dest(Local);
    int result =
        which == scoped_op::put_get
            ? nvshmemx_put<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, source, bytes, peer)
            : nvshmemx_put_nbi<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, source, bytes,
                                                                         peer);
    switch (which) {
        case scoped_op::put_get:
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_quiet<A, P, Scope>();
            }
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_get_nbi<uint8_t, A, P, Scope, kRemoteSource, kLocalDest>(
                    out, source, bytes, peer);
            }
            break;
        case scoped_op::put_nbi_flush:
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_flush<A, P, Scope>();
            }
            if (result == NVSHMEMX_SUCCESS) {
                /* flush makes the source reusable; the peer must still see the original. */
                fill_group<Scope>(source, bytes, kPoisonByte);
            }
            break;
        case scoped_op::put_nbi_fence:
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_fence<A, P, Scope>();
            }
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_signal_op<A, P, Scope>(signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            }
            break;
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, Scope>();
    }
    return result;
}

/*
 * Reports only failures; the caller zeroes *status. Block 0 issues; for
 * put_nbi_fence, block 1 receives the previous PE's payload and signal. Each
 * group checks its get result right after its route quiet.
 */
template <typename A, typename P, threadgroup Scope, LocalSpace Local>
__global__ void scoped_kernel(scoped_op which, unsigned char *remote, unsigned char *source,
                              unsigned char *out, uint64_t *signal, size_t bytes, int issuers,
                              int peer, int previous, int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, kSignalBase + static_cast<uint64_t>(issuers), remote, bytes,
                          previous, status);
        return;
    }
    const int group = scope_group_index<Scope>();
    if (group < issuers) {
        const size_t slice = bytes / issuers;
        const size_t offset = static_cast<size_t>(group) * slice;
        int result = scoped_rma<A, P, Scope, Local>(which, remote + offset, source + offset,
                                                    out + offset, signal, slice, peer);
        if (result == NVSHMEMX_SUCCESS && which == scoped_op::put_get) {
            const size_t lane = Scope == threadgroup::warp ? threadIdx.x % warpSize : 0;
            const size_t width = Scope == threadgroup::warp ? warpSize : 1;
            if (!payload_matches(out + offset, slice, peer, offset, lane, width)) {
                result = kDataMismatch;
            }
        }
        if (result != NVSHMEMX_SUCCESS) {
            atomicExch(status, result);
        }
    }
}

struct scoped_target {
    world w;
    bool active;
    int peer;
    int previous;
    unsigned char *remote; /* symmetric */
    unsigned char *source; /* symmetric */
    unsigned char *out;    /* symmetric */
    uint64_t *signal;      /* symmetric */
    size_t bytes;
};

template <typename A, typename P, threadgroup Scope, LocalSpace Local>
int scoped_case(const scoped_target &t, const char *what, scoped_op which, int issuers) {
    return run_case(
        t.w, t.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(t.source, t.bytes, t.w.mype);
            CUDA_CHECK(cudaMemset(t.remote, 0, t.bytes));
            CUDA_CHECK(cudaMemset(t.out, 0, t.bytes));
            set_signal(t.signal, kSignalBase);
        },
        [&] {
            CUDA_CHECK(cudaMemset(t.w.status, 0, sizeof(*t.w.status)));
            const int blocks = which == scoped_op::put_nbi_fence ? 2 : 1;
            scoped_kernel<A, P, Scope, Local>
                <<<blocks, kThreads>>>(which, t.remote, t.source, t.out, t.signal, t.bytes, issuers,
                                       t.peer, t.previous, t.w.status);
        },
        [&] {
            int failures = check_payload(what, t.remote, t.bytes, t.previous);
            if (which == scoped_op::put_get) {
                failures += check_payload(what, t.out, t.bytes, t.peer);
            } else if (which == scoped_op::put_nbi_fence) {
                failures += check_signal_and_wait(
                    what, t.signal, kSignalBase + static_cast<uint64_t>(issuers), t.w.status);
            }
            return failures;
        });
}

/* put/get, put_nbi + flush, and put_nbi + fence + signal at thread and warp
 * scope, with every thread or warp of the CTA issuing. */
template <typename A, typename P, LocalSpace Local>
int thread_warp_cases(const scoped_target &t, const char *route) {
    const int thread_issuers = kThreads;
    const int warp_issuers = kThreads / 32;
    const scoped_op ops[] = {scoped_op::put_get, scoped_op::put_nbi_flush,
                             scoped_op::put_nbi_fence};
    const char *names[] = {"put+quiet/get_nbi+quiet", "put_nbi+flush reuse+quiet",
                           "put_nbi+fence+signal_op+quiet"};
    int failures = 0;
    for (int i = 0; i < 3; i++) {
        char what[160];
        snprintf(what, sizeof(what), "%s thread-scope %s", route, names[i]);
        failures += scoped_case<A, P, threadgroup::thread, Local>(t, what, ops[i], thread_issuers);
        snprintf(what, sizeof(what), "%s warp-scope %s", route, names[i]);
        failures += scoped_case<A, P, threadgroup::warp, Local>(t, what, ops[i], warp_issuers);
    }
    return failures;
}

/*
 * Auto / Auto against the existing portable API at the matching scope: the
 * same transfer is issued once through each into separate buffers, and both
 * must produce identical data and signal values. The stable APIs return no
 * status, so the route must report success.
 */
enum class compare_op { put, put_nbi, get, get_nbi, put_signal, put_signal_nbi };

template <threadgroup Scope>
__device__ inline void stable_rma(compare_op which, unsigned char *remote, unsigned char *source,
                                  unsigned char *out, uint64_t *signal, size_t bytes, int peer) {
    switch (which) {
        case compare_op::put:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_putmem(remote, source, bytes, peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_putmem_warp(remote, source, bytes, peer);
            } else {
                nvshmemx_putmem_block(remote, source, bytes, peer);
            }
            break;
        case compare_op::put_nbi:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_putmem_nbi(remote, source, bytes, peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_putmem_nbi_warp(remote, source, bytes, peer);
            } else {
                nvshmemx_putmem_nbi_block(remote, source, bytes, peer);
            }
            break;
        case compare_op::get:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_getmem(out, source, bytes, peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_getmem_warp(out, source, bytes, peer);
            } else {
                nvshmemx_getmem_block(out, source, bytes, peer);
            }
            break;
        case compare_op::get_nbi:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_getmem_nbi(out, source, bytes, peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_getmem_nbi_warp(out, source, bytes, peer);
            } else {
                nvshmemx_getmem_nbi_block(out, source, bytes, peer);
            }
            break;
        case compare_op::put_signal:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_putmem_signal(remote, source, bytes, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_putmem_signal_warp(remote, source, bytes, signal, 1, NVSHMEM_SIGNAL_ADD,
                                            peer);
            } else {
                nvshmemx_putmem_signal_block(remote, source, bytes, signal, 1, NVSHMEM_SIGNAL_ADD,
                                             peer);
            }
            break;
        case compare_op::put_signal_nbi:
            if constexpr (Scope == threadgroup::thread) {
                nvshmem_putmem_signal_nbi(remote, source, bytes, signal, 1, NVSHMEM_SIGNAL_ADD,
                                          peer);
            } else if constexpr (Scope == threadgroup::warp) {
                nvshmemx_putmem_signal_nbi_warp(remote, source, bytes, signal, 1,
                                                NVSHMEM_SIGNAL_ADD, peer);
            } else {
                nvshmemx_putmem_signal_nbi_block(remote, source, bytes, signal, 1,
                                                 NVSHMEM_SIGNAL_ADD, peer);
            }
            break;
    }
}

template <threadgroup Scope>
__device__ inline int auto_rma(compare_op which, unsigned char *remote, unsigned char *source,
                               unsigned char *out, uint64_t *signal, size_t bytes, int peer) {
    using A = nvshmemx::addressing::Auto;
    using P = nvshmemx::transport_policy::Auto;
    switch (which) {
        case compare_op::put:
            return nvshmemx_put<uint8_t, A, P, Scope>(remote, source, bytes, peer);
        case compare_op::put_nbi:
            return nvshmemx_put_nbi<uint8_t, A, P, Scope>(remote, source, bytes, peer);
        case compare_op::get:
            return nvshmemx_get<uint8_t, A, P, Scope>(out, source, bytes, peer);
        case compare_op::get_nbi:
            return nvshmemx_get_nbi<uint8_t, A, P, Scope>(out, source, bytes, peer);
        case compare_op::put_signal:
            return nvshmemx_put_signal<uint8_t, A, P, Scope>(remote, source, bytes, signal, 1,
                                                             NVSHMEM_SIGNAL_ADD, peer);
        default:
            return nvshmemx_put_signal_nbi<uint8_t, A, P, Scope>(remote, source, bytes, signal, 1,
                                                                 NVSHMEM_SIGNAL_ADD, peer);
    }
}

/* ref_* receive the stable API's transfer; the others the route's. */
struct compare_buffers {
    unsigned char *remote;
    unsigned char *out;
    uint64_t *signal;
    unsigned char *ref_remote;
    unsigned char *ref_out;
    uint64_t *ref_signal;
};

/* Reports only failures; the caller zeroes *status. */
template <threadgroup Scope>
__global__ void auto_vs_stable_kernel(compare_op which, compare_buffers b, unsigned char *source,
                                      size_t bytes, int issuers, int peer, int *status) {
    using A = nvshmemx::addressing::Auto;
    using P = nvshmemx::transport_policy::Auto;
    const int group = scope_group_index<Scope>();
    if (group < issuers) {
        const size_t slice = bytes / issuers;
        const size_t offset = static_cast<size_t>(group) * slice;
        stable_rma<Scope>(which, b.ref_remote + offset, source + offset, b.ref_out + offset,
                          b.ref_signal, slice, peer);
        int result = auto_rma<Scope>(which, b.remote + offset, source + offset, b.out + offset,
                                     b.signal, slice, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_quiet<A, P, Scope>();
        }
        if (result != NVSHMEMX_SUCCESS) {
            atomicExch(status, result);
        }
    }
}

inline int check_same(const char *what, const unsigned char *device, const unsigned char *ref,
                      size_t bytes) {
    std::vector<unsigned char> host(bytes);
    std::vector<unsigned char> host_ref(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_ref.data(), ref, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; i++) {
        if (host[i] != host_ref[i]) {
            fprintf(stderr, "PE %d: %s: byte[%zu] = %u, portable API wrote %u\n", nvshmem_my_pe(),
                    what, i, static_cast<unsigned int>(host[i]),
                    static_cast<unsigned int>(host_ref[i]));
            return 1;
        }
    }
    return 0;
}

template <threadgroup Scope>
int auto_vs_stable_case(const scoped_target &t, const compare_buffers &b, const char *what,
                        compare_op which, int issuers) {
    const bool is_get = which == compare_op::get || which == compare_op::get_nbi;
    const bool is_signal = which == compare_op::put_signal || which == compare_op::put_signal_nbi;
    return run_case(
        t.w, t.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(t.source, t.bytes, t.w.mype);
            CUDA_CHECK(cudaMemset(b.remote, 0, t.bytes));
            CUDA_CHECK(cudaMemset(b.out, 0, t.bytes));
            CUDA_CHECK(cudaMemset(b.ref_remote, 0, t.bytes));
            CUDA_CHECK(cudaMemset(b.ref_out, 0, t.bytes));
            set_signal(b.signal, kSignalBase);
            set_signal(b.ref_signal, kSignalBase);
        },
        [&] {
            CUDA_CHECK(cudaMemset(t.w.status, 0, sizeof(*t.w.status)));
            auto_vs_stable_kernel<Scope>
                <<<1, kThreads>>>(which, b, t.source, t.bytes, issuers, t.peer, t.w.status);
        },
        [&] {
            if (is_get) {
                return check_same(what, b.out, b.ref_out, t.bytes) +
                       check_payload(what, b.out, t.bytes, t.peer);
            }
            int failures = check_same(what, b.remote, b.ref_remote, t.bytes) +
                           check_payload(what, b.remote, t.bytes, t.previous);
            if (is_signal) {
                uint64_t ref = 0;
                CUDA_CHECK(cudaMemcpy(&ref, b.ref_signal, sizeof(ref), cudaMemcpyDeviceToHost));
                failures +=
                    check_signal(what, b.signal, ref) +
                    check_signal(what, b.ref_signal, kSignalBase + static_cast<uint64_t>(issuers));
            }
            return failures;
        });
}

/* Every RMA family at thread, warp, and block scope. */
inline int auto_vs_stable_cases(const scoped_target &t, const compare_buffers &b,
                                const char *route) {
    const compare_op ops[] = {compare_op::put,        compare_op::put_nbi,
                              compare_op::get,        compare_op::get_nbi,
                              compare_op::put_signal, compare_op::put_signal_nbi};
    const char *names[] = {"put", "put_nbi", "get", "get_nbi", "put_signal", "put_signal_nbi"};
    int failures = 0;
    for (int i = 0; i < 6; i++) {
        char what[160];
        snprintf(what, sizeof(what), "%s thread-scope %s vs portable API", route, names[i]);
        failures += auto_vs_stable_case<threadgroup::thread>(t, b, what, ops[i], kThreads);
        snprintf(what, sizeof(what), "%s warp-scope %s vs portable API", route, names[i]);
        failures += auto_vs_stable_case<threadgroup::warp>(t, b, what, ops[i], kThreads / 32);
        snprintf(what, sizeof(what), "%s block-scope %s vs portable API", route, names[i]);
        failures += auto_vs_stable_case<threadgroup::block>(t, b, what, ops[i], 1);
    }
    return failures;
}

inline void report(int mype, const char *test, const char *route, int failures) {
    if (mype != 0) {
        return;
    }
    if (failures) {
        printf("%s: FAIL %s\n", test, route);
    } else {
        printf("%s: PASS %s\n", test, route);
    }
}

}  // namespace ts_test

#endif  // TRANSPORT_SPECIFIC_COMMON_CUH
