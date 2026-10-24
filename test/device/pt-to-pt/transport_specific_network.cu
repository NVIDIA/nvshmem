/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Functional coverage for the strict Auto / Network route, plus an Auto / Auto
 * control. The route never falls back to a P2P pointer, so once the job uses a
 * device network transport every positive case must succeed, including those
 * that target a P2P-mapped peer. In a P2P-only build, or a job NVSHMEM runs
 * over LD/ST alone, the route reports NOT_SUPPORTED and its data cases are a
 * clean skip. Shared-memory local operands and argument errors are rejected
 * with INVALID_VALUE in every build and job, before any work reaches the NIC;
 * only where a sibling preflight (the signal half of put_signal, or another
 * lane of the group) also finds the route unavailable does the more severe
 * NOT_SUPPORTED win, as the group-uniform agreement defines.
 * The dedicated launcher runs one PE per node with P2P disabled.
 *
 * Signalled cases launch a second CTA that receives the previous PE's payload
 * and signal concurrently and checks the payload as soon as the signal lands,
 * before any barrier or kernel boundary. Gets check their result in the
 * kernel right after the call that completes them. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_common.cuh"

using namespace ts_test;

namespace {

constexpr size_t kBytes = 512;
using AutoAddressing = nvshmemx::addressing::Auto;
using Network = nvshmemx::transport_policy::Network;
using AutoPolicy = nvshmemx::transport_policy::Auto;

enum class op {
    put,                 /* put */
    put_nbi_flush,       /* put_nbi + flush, poison the source slice, quiet */
    get,                 /* get */
    get_nbi_quiet,       /* get_nbi + quiet */
    put_signal,          /* put_signal */
    put_signal_nbi,      /* put_signal_nbi + quiet */
    signal_op,           /* signal_op + quiet */
    put_nbi_fence,       /* put_nbi + fence + signal_op ADD 1 + quiet */
    quiet,               /* quiet alone */
    flush,               /* flush alone */
    fence,               /* fence alone */
    typed_count_overflow /* uint64 put whose element count overflows the byte count */
};

__host__ __device__ constexpr bool is_get(op which) {
    return which == op::get || which == op::get_nbi_quiet;
}

__host__ __device__ constexpr bool sends_signal(op which) {
    return which == op::put_signal || which == op::put_signal_nbi || which == op::signal_op ||
           which == op::put_nbi_fence;
}

/* A SET put_signal from several issuers gives each issuer its own signal word:
 * a receiver of one shared word could not tell which slice had landed. */
constexpr int kMaxIssuers = kThreads;

template <typename P, threadgroup Scope>
__device__ int order(op which) {
    using A = AutoAddressing;
    return which == op::flush   ? nvshmemx_flush<A, P, Scope>()
           : which == op::fence ? nvshmemx_fence<A, P, Scope>()
                                : nvshmemx_quiet<A, P, Scope>();
}

/* One threadgroup's operation on its slice. `local` is the put source or the
 * get destination; get sources are the peer's `source` slice. */
template <typename P, threadgroup Scope, LocalSpace Local>
__device__ int issue(op which, unsigned char *remote, unsigned char *local,
                     const unsigned char *remote_source, uint64_t *signal, uint64_t value,
                     int sig_op, size_t bytes, int peer) {
    using A = AutoAddressing;
    constexpr auto kLocalDest = local_dest(Local);
    int result = NVSHMEMX_SUCCESS;
    switch (which) {
        case op::put:
            return nvshmemx_put<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, local, bytes,
                                                                          peer);
        case op::put_nbi_flush:
            result = nvshmemx_put_nbi<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, local,
                                                                                bytes, peer);
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_flush<A, P, Scope>();
            }
            if (result == NVSHMEMX_SUCCESS) {
                /* flush makes the source reusable; the peer must still see the original. */
                fill_group<Scope>(local, bytes, kPoisonByte);
                result = nvshmemx_quiet<A, P, Scope>();
            }
            return result;
        case op::get:
            return nvshmemx_get<uint8_t, A, P, Scope, kRemoteSource, kLocalDest>(
                local, remote_source, bytes, peer);
        case op::get_nbi_quiet:
            result = nvshmemx_get_nbi<uint8_t, A, P, Scope, kRemoteSource, kLocalDest>(
                local, remote_source, bytes, peer);
            break;
        case op::put_signal:
            return nvshmemx_put_signal<uint8_t, A, P, Scope, Local, kRemoteDest>(
                remote, local, bytes, signal, value, sig_op, peer);
        case op::put_signal_nbi:
            result = nvshmemx_put_signal_nbi<uint8_t, A, P, Scope, Local, kRemoteDest>(
                remote, local, bytes, signal, value, sig_op, peer);
            break;
        case op::signal_op:
            result = nvshmemx_signal_op<A, P, Scope>(signal, value, sig_op, peer);
            break;
        case op::put_nbi_fence:
            result = nvshmemx_put_nbi<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, local,
                                                                                bytes, peer);
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_fence<A, P, Scope>();
            }
            if (result == NVSHMEMX_SUCCESS) {
                result = nvshmemx_signal_op<A, P, Scope>(signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            }
            break;
        case op::quiet:
        case op::flush:
        case op::fence:
            return order<P, Scope>(which);
        case op::typed_count_overflow:
            return nvshmemx_put<uint64_t, A, P, Scope, Local, kRemoteDest>(
                reinterpret_cast<uint64_t *>(remote), reinterpret_cast<const uint64_t *>(local),
                static_cast<size_t>(-1) / 4, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, Scope>();
    }
    return result;
}

/* The receiving half of a signalled case with one signal word per issuer:
 * each of the first `issuers` threads waits for its issuer's word and checks
 * that issuer's slice at once. */
__device__ void receive_slices_and_check(uint64_t *signal, uint64_t expected,
                                         const unsigned char *payload, size_t bytes, int issuers,
                                         int source_pe, int *status) {
    const int group = threadIdx.x;
    if (group >= issuers) {
        return;
    }
    const size_t slice = bytes / issuers;
    const size_t offset = static_cast<size_t>(group) * slice;
    const uint64_t start = global_time_ns();
    bool seen = false;
    while (!seen && global_time_ns() - start < kReceiveTimeoutNs) {
        seen = nvshmem_signal_fetch(signal + group) == expected;
    }
    __threadfence();
    if (!seen) {
        report_receiver(status, kReceiveTimeout);
    } else if (!payload_matches(payload + offset, slice, source_pe, offset, 0, 1)) {
        report_receiver(status, kDataMismatch);
    }
}

/* This thread's position within its Scope group, for in-kernel checks. */
template <threadgroup Scope>
__device__ size_t group_lane() {
    return Scope == threadgroup::thread ? 0
           : Scope == threadgroup::warp ? threadIdx.x % warpSize
                                        : threadIdx.x;
}

template <threadgroup Scope>
__device__ size_t group_width() {
    return Scope == threadgroup::thread ? 1 : Scope == threadgroup::warp ? warpSize : blockDim.x;
}

/* Arguments of the concurrent receiver CTA (block 1, when launched). */
struct receiver {
    uint64_t expected; /* the signal value that completes the case */
    size_t bytes;      /* payload bytes ordered before the signal */
    int previous;      /* the PE sending to this one */
    int signal_stride; /* 1: one signal word per issuer; 0: one shared word */
};

/* Each of the first `issuers` threadgroups of Scope moves its own contiguous
 * slice, signalling signal[group * signal_stride]. With InShared the local
 * operand is a dynamic shared-memory copy of `local`. Block 0 issues; block 1,
 * when launched, receives the previous PE's payload and signal concurrently.
 * Gets are checked right after they complete. Reports only failures; the
 * caller zeroes *status. */
template <typename P, threadgroup Scope, LocalSpace Local, bool InShared>
__global__ void op_kernel(op which, unsigned char *remote, unsigned char *local,
                          const unsigned char *remote_source, uint64_t *signal, uint64_t value,
                          int sig_op, size_t bytes, int issuers, int peer, int *status,
                          receiver recv) {
    if (blockIdx.x == 1) {
        if (recv.signal_stride != 0) {
            receive_slices_and_check(signal, recv.expected, remote, recv.bytes, issuers,
                                     recv.previous, status);
        } else {
            receive_and_check(signal, recv.expected, remote, recv.bytes, recv.previous, status);
        }
        return;
    }
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *operand = local;
    if constexpr (InShared) {
        copy_block(smem, local, bytes);
        __syncthreads();
        operand = smem;
    }
    const int group = scope_group_index<Scope>();
    if (group < issuers) {
        const size_t slice = bytes / issuers;
        const size_t offset = static_cast<size_t>(group) * slice;
        unsigned char *slice_operand = operand == nullptr ? nullptr : operand + offset;
        int result =
            issue<P, Scope, Local>(which, remote + offset, slice_operand, remote_source + offset,
                                   signal + group * recv.signal_stride, value, sig_op, slice, peer);
        if (result == NVSHMEMX_SUCCESS && is_get(which) &&
            !payload_matches(slice_operand, slice, peer, offset, group_lane<Scope>(),
                             group_width<Scope>())) {
            result = kDataMismatch;
        }
        if (result != NVSHMEMX_SUCCESS) {
            atomicExch(status, result);
        }
    }
}

/* Group-uniform preflight at warp scope: the last lane of each warp names an
 * out-of-range PE, so every lane must return the group's status (INVALID_VALUE
 * while the route is available) and no warp may write its slice. */
__global__ void divergent_warp_kernel(unsigned char *remote, const unsigned char *source,
                                      size_t bytes, int peer, int npes, int expected, int *status) {
    const int group = scope_group_index<threadgroup::warp>();
    const size_t slice = bytes / (blockDim.x / warpSize);
    const bool last = threadIdx.x % warpSize == warpSize - 1;
    const int result =
        nvshmemx_put<uint8_t, AutoAddressing, Network, threadgroup::warp, kGlobal, kRemoteDest>(
            remote + group * slice, source + group * slice, slice, last ? npes : peer);
    if (result != expected) {
        atomicExch(status, result);
    }
}

/* Zero-length typed transfers move nothing and need no local operand; the
 * probe at the start of main uses the same call. */
template <typename T>
__global__ void typed_kernel(unsigned char *remote, unsigned char *source, unsigned char *out,
                             size_t bytes, int peer, int *status) {
    using A = AutoAddressing;
    T *remote_typed = reinterpret_cast<T *>(remote);
    T *source_typed = reinterpret_cast<T *>(source);
    T *out_typed = reinterpret_cast<T *>(out);
    const size_t nelems = bytes / sizeof(T);
    int result = nvshmemx_put<T, A, Network, kBlock, kGlobal, kRemoteDest>(
        remote_typed, source_typed, nelems, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<T, A, Network, kBlock, kRemoteSource, local_dest(kGlobal)>(
            out_typed, source_typed, nelems, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, Network, kBlock>();
    }
    if (result == NVSHMEMX_SUCCESS &&
        __syncthreads_or(!payload_matches(out, bytes, peer, 0, threadIdx.x, blockDim.x))) {
        result = kDataMismatch;
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* A zero-length put_signal moves no payload and still delivers its signal:
 * SET into signal[0] and, through the NBI spelling, ADD 1 into signal[1]. */
__global__ void zero_length_kernel(unsigned char *remote, const unsigned char *source,
                                   uint64_t *signal, uint64_t set_value, int peer, int *status) {
    using A = AutoAddressing;
    int result =
        nvshmemx_put<uint8_t, A, Network, kBlock, kGeneric, kRemoteDest>(remote, nullptr, 0, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get<uint8_t, A, Network, kBlock, kRemoteSource, local_dest(kGlobal)>(
            nullptr, source, 0, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_put_signal<uint8_t, A, Network, kBlock, kGeneric, kRemoteDest>(
            remote, nullptr, 0, signal, set_value, NVSHMEM_SIGNAL_SET, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_put_signal_nbi<uint8_t, A, Network, kBlock, kGeneric, kRemoteDest>(
            remote, nullptr, 0, signal + 1, 1, NVSHMEM_SIGNAL_ADD, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, Network, kBlock>();
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* ---- Host-side cases ------------------------------------------------------ */

struct route_context {
    world w;
    bool active;
    int peer;
    int previous;
    unsigned char *remote; /* symmetric put destination */
    unsigned char *source; /* symmetric put source and peer get source */
    unsigned char *out;    /* symmetric get destination */
    uint64_t *signal;      /* symmetric */
    unsigned char *private_buffer;
    uint64_t *private_signal;

    scoped_target scoped() const {
        return scoped_target{w, active, peer, previous, remote, source, out, signal, kBytes};
    }
};

/* Sets `count` consecutive signal words to `value`. */
void set_signals(uint64_t *device, int count, uint64_t value) {
    std::vector<uint64_t> host(count, value);
    CUDA_CHECK(cudaMemcpy(device, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

template <threadgroup Scope>
constexpr int issuers_for() {
    return Scope == threadgroup::thread ? kThreads : Scope == threadgroup::warp ? kThreads / 32 : 1;
}

template <threadgroup Scope>
const char *scope_name() {
    return Scope == threadgroup::thread ? "thread" : Scope == threadgroup::warp ? "warp" : "block";
}

/* A single issuing CTA with no receiver. */
template <typename P, threadgroup Scope, LocalSpace Local, bool InShared = false, typename... Args>
void launch_op(Args... args) {
    op_kernel<P, Scope, Local, InShared>
        <<<1, kThreads, InShared ? kBytes : 0>>>(args..., receiver{});
}

/* Data, signal, and ordering success with every threadgroup of Scope issuing.
 * A SET put_signal from several issuers sets one signal word per issuer to
 * the sender's value; a SET signal_op sets the shared word to it from every
 * issuer; an ADD adds the sender's value once per issuer to the shared word.
 * Signalled cases run a concurrent receiver CTA. */
template <typename P, threadgroup Scope, LocalSpace Local>
int data_case(const route_context &c, const char *route, const char *local, const char *name,
              op which, int sig_op = NVSHMEM_SIGNAL_SET) {
    constexpr int kIssuers = issuers_for<Scope>();
    const bool add = sig_op == NVSHMEM_SIGNAL_ADD;
    const bool signals = sends_signal(which);
    const bool moves_payload = which != op::signal_op;
    const bool per_issuer_signal =
        !add && kIssuers > 1 && (which == op::put_signal || which == op::put_signal_nbi);
    const int signal_words = per_issuer_signal ? kIssuers : 1;
    const uint64_t value = add ? static_cast<uint64_t>(c.w.mype) + 1 : kSignalBase + c.w.mype;
    const uint64_t expected = which == op::put_nbi_fence ? kSignalBase + kIssuers
                              : add
                                  ? kSignalBase + static_cast<uint64_t>(kIssuers) * (c.previous + 1)
                                  : kSignalBase + static_cast<uint64_t>(c.previous);
    const uint64_t initial = add || which == op::put_nbi_fence ? kSignalBase : 0;
    const receiver recv{expected, moves_payload ? kBytes : 0, c.previous,
                        per_issuer_signal ? 1 : 0};
    char what[192];
    snprintf(what, sizeof(what), "%s %s-scope %s %s", route, scope_name<Scope>(), local, name);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            CUDA_CHECK(cudaMemset(c.out, 0, kBytes));
            set_signals(c.signal, kMaxIssuers, initial);
        },
        [&] {
            CUDA_CHECK(cudaMemset(c.w.status, 0, sizeof(*c.w.status)));
            unsigned char *local_operand = is_get(which) ? c.out : c.source;
            op_kernel<P, Scope, Local, false><<<signals ? 2 : 1, kThreads>>>(
                which, c.remote, local_operand, static_cast<const unsigned char *>(c.source),
                c.signal, value, sig_op, kBytes, kIssuers, c.peer, c.w.status, recv);
        },
        [&] {
            if (is_get(which)) {
                return check_payload(what, c.out, kBytes, c.peer);
            }
            int failures = 0;
            if (signals) {
                for (int i = 1; i < signal_words; i++) {
                    failures += check_signal(what, c.signal + i, expected);
                }
                failures += check_signal_and_wait(what, c.signal, expected, c.w.status);
            }
            if (moves_payload) {
                failures += check_payload(what, c.remote, kBytes, c.previous);
            }
            return failures;
        });
}

template <typename P, threadgroup Scope, LocalSpace Local>
int scope_cases(const route_context &c, const char *route, const char *local) {
    constexpr int kSet = NVSHMEM_SIGNAL_SET;
    constexpr int kAdd = NVSHMEM_SIGNAL_ADD;
    int failures = 0;
    failures += data_case<P, Scope, Local>(c, route, local, "put", op::put);
    failures +=
        data_case<P, Scope, Local>(c, route, local, "put_nbi+flush reuse+quiet", op::put_nbi_flush);
    failures += data_case<P, Scope, Local>(c, route, local, "get", op::get);
    failures += data_case<P, Scope, Local>(c, route, local, "get_nbi+quiet", op::get_nbi_quiet);
    failures += data_case<P, Scope, Local>(c, route, local, "put_signal SET", op::put_signal, kSet);
    failures += data_case<P, Scope, Local>(c, route, local, "put_signal ADD", op::put_signal, kAdd);
    failures += data_case<P, Scope, Local>(c, route, local, "put_signal_nbi SET+quiet",
                                           op::put_signal_nbi, kSet);
    failures += data_case<P, Scope, Local>(c, route, local, "put_signal_nbi ADD+quiet",
                                           op::put_signal_nbi, kAdd);
    failures +=
        data_case<P, Scope, Local>(c, route, local, "signal_op SET+quiet", op::signal_op, kSet);
    failures +=
        data_case<P, Scope, Local>(c, route, local, "signal_op ADD+quiet", op::signal_op, kAdd);
    failures += data_case<P, Scope, Local>(c, route, local, "put_nbi+fence+signal_op ADD+quiet",
                                           op::put_nbi_fence);
    return failures;
}

template <typename P, LocalSpace Local>
int all_scope_cases(const route_context &c, const char *route, const char *local) {
    return scope_cases<P, threadgroup::thread, Local>(c, route, local) +
           scope_cases<P, threadgroup::warp, Local>(c, route, local) +
           scope_cases<P, threadgroup::block, Local>(c, route, local);
}

template <typename T>
int typed_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            CUDA_CHECK(cudaMemset(c.out, 0, kBytes));
        },
        [&] {
            typed_kernel<T><<<1, kThreads>>>(c.remote, c.source, c.out, kBytes, c.peer, c.w.status);
        },
        [&] {
            return check_payload(what, c.remote, kBytes, c.previous) +
                   check_payload(what, c.out, kBytes, c.peer);
        });
}

int zero_length_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.source, kSentinelByte, kBytes));
            set_signals(c.signal, 2, kSignalBase);
        },
        [&] {
            zero_length_kernel<<<1, kThreads>>>(c.remote, c.source, c.signal,
                                                kSignalBase + 1 + c.w.mype, c.peer, c.w.status);
        },
        [&] {
            return check_fill(what, c.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.source, kBytes, kSentinelByte) +
                   check_signal_and_wait(what, c.signal,
                                         kSignalBase + 1 + static_cast<uint64_t>(c.previous),
                                         c.w.status) +
                   check_signal_and_wait(what, c.signal + 1, kSignalBase + 1, c.w.status);
        });
}

/* Every Auto / Network data case for one context, with global and generic
 * local operands in global memory. */
int network_cases(const route_context &c, const char *route) {
    int failures = 0;
    failures += all_scope_cases<Network, kGlobal>(c, route, "global");
    failures += all_scope_cases<Network, kGeneric>(c, route, "generic(global)");
    /* The common scoped coverage: several concurrent issuers per CTA. */
    char name[96];
    snprintf(name, sizeof(name), "%s global", route);
    failures += thread_warp_cases<AutoAddressing, Network, kGlobal>(c.scoped(), name);
    snprintf(name, sizeof(name), "%s generic(global)", route);
    failures += thread_warp_cases<AutoAddressing, Network, kGeneric>(c.scoped(), name);
    return failures;
}

/* ---- Rejected calls ------------------------------------------------------- */

/* A rejected call must not issue work: every PE's destination, get buffer, and
 * signal keep their sentinels. */
template <typename Launch>
int rejected_case(const route_context &c, const char *what, int expected, Launch launch) {
    return run_case(
        c.w, true, what, expected,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.out, kSentinelByte, kBytes));
            set_signal(c.signal, kSentinelSignal);
        },
        [&] {
            CUDA_CHECK(cudaMemset(c.w.status, 0, sizeof(*c.w.status)));
            launch();
        },
        [&] {
            return check_fill(what, c.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.out, kBytes, kSentinelByte) +
                   check_signal(what, c.signal, kSentinelSignal);
        });
}

/* A local operand in shared memory never reaches the transport: an explicit
 * shared value is rejected without classification and a generic one after
 * classifying as shared, at every scope and in every build. `combined` is the
 * expected status of put_signal, whose signal preflight also runs:
 * INVALID_VALUE, or NOT_SUPPORTED while the route is unavailable. */
template <threadgroup Scope, LocalSpace Local>
int shared_local_cases(const route_context &c, const char *local, int combined) {
    const op ops[] = {op::put,           op::put_nbi_flush, op::get,
                      op::get_nbi_quiet, op::put_signal,    op::put_signal_nbi};
    const char *names[] = {"put", "put_nbi", "get", "get_nbi", "put_signal", "put_signal_nbi"};
    int failures = 0;
    for (int i = 0; i < 6; i++) {
        char what[192];
        snprintf(what, sizeof(what), "Network %s-scope %s local %s rejected", scope_name<Scope>(),
                 local, names[i]);
        const op which = ops[i];
        const bool signals = which == op::put_signal || which == op::put_signal_nbi;
        const int expected = signals ? combined : NVSHMEMX_ERROR_INVALID_VALUE;
        failures += rejected_case(c, what, expected, [&] {
            unsigned char *local_operand = is_get(which) ? c.out : c.source;
            launch_op<Network, Scope, Local, true>(
                which, c.remote, local_operand, static_cast<const unsigned char *>(c.source),
                c.signal, 1, NVSHMEM_SIGNAL_SET, kBytes, issuers_for<Scope>(), c.peer, c.w.status);
        });
    }
    return failures;
}

int memory_space_rejections(const route_context &c, int combined) {
    int failures = 0;
    failures += shared_local_cases<threadgroup::thread, kShared>(c, "shared", combined);
    failures += shared_local_cases<threadgroup::warp, kShared>(c, "shared", combined);
    failures += shared_local_cases<threadgroup::block, kShared>(c, "shared", combined);
    failures += shared_local_cases<threadgroup::thread, kGeneric>(c, "generic(shared)", combined);
    failures += shared_local_cases<threadgroup::warp, kGeneric>(c, "generic(shared)", combined);
    failures += shared_local_cases<threadgroup::block, kGeneric>(c, "generic(shared)", combined);
    return failures;
}

/* One argument error at Scope. Arguments are checked before the capability, so
 * a single preflight returns INVALID_VALUE in every build and job; `expected`
 * differs only for the combined cases described above. */
template <threadgroup Scope>
int argument_case(const route_context &c, const char *what, op which, unsigned char *remote,
                  unsigned char *local, uint64_t *signal, int sig_op, int pe,
                  int expected = NVSHMEMX_ERROR_INVALID_VALUE) {
    return rejected_case(c, what, expected, [&] {
        launch_op<Network, Scope, kGlobal>(which, remote, local,
                                           static_cast<const unsigned char *>(c.source), signal, 1,
                                           sig_op, kBytes, issuers_for<Scope>(), pe, c.w.status);
    });
}

int argument_rejections(const route_context &c, int combined) {
    constexpr auto kThread = threadgroup::thread;
    constexpr auto kWarp = threadgroup::warp;
    constexpr int kSet = NVSHMEM_SIGNAL_SET;
    int failures = 0;
    failures += argument_case<kBlock>(c, "Network put invalid PE", op::put, c.remote, c.source,
                                      c.signal, kSet, c.w.npes);
    failures += argument_case<kWarp>(c, "Network get negative PE", op::get, c.remote, c.out,
                                     c.signal, kSet, -1);
    failures += argument_case<kThread>(c, "Network put remote outside the heap", op::put,
                                       c.private_buffer, c.source, c.signal, kSet, c.peer);
    failures += argument_case<kBlock>(c, "Network put null local with bytes", op::put, c.remote,
                                      nullptr, c.signal, kSet, c.peer);
    failures += argument_case<kBlock>(c, "Network put_signal invalid op", op::put_signal, c.remote,
                                      c.source, c.signal, 0, c.peer, combined);
    failures += argument_case<kWarp>(c, "Network put_signal_nbi signal outside the heap",
                                     op::put_signal_nbi, c.remote, c.source, c.private_signal, kSet,
                                     c.peer, combined);
    failures += argument_case<kThread>(c, "Network signal_op invalid op", op::signal_op, c.remote,
                                       c.source, c.signal, 0, c.peer);
    failures += argument_case<kBlock>(c, "Network signal_op invalid PE", op::signal_op, c.remote,
                                      c.source, c.signal, kSet, c.w.npes);
    failures +=
        argument_case<kBlock>(c, "Network typed put element-count overflow",
                              op::typed_count_overflow, c.remote, c.source, c.signal, kSet, c.peer);
    failures += rejected_case(c, "Network warp-scope divergent PE", combined, [&] {
        CUDA_CHECK(cudaMemcpy(c.w.status, &combined, sizeof(combined), cudaMemcpyHostToDevice));
        divergent_warp_kernel<<<1, kThreads>>>(c.remote, c.source, kBytes, c.peer, c.w.npes,
                                               combined, c.w.status);
    });
    return failures;
}

/* The route is unavailable: every valid operation, ordering included, reports
 * NOT_SUPPORTED, never INTERNAL, and touches no buffer. */
template <threadgroup Scope>
int unavailable_scope_cases(const route_context &c) {
    const op ops[] = {op::put,        op::put_nbi_flush,  op::get,       op::get_nbi_quiet,
                      op::put_signal, op::put_signal_nbi, op::signal_op, op::quiet,
                      op::flush,      op::fence};
    const char *names[] = {"put",        "put_nbi",        "get",       "get_nbi",
                           "put_signal", "put_signal_nbi", "signal_op", "quiet",
                           "flush",      "fence"};
    int failures = 0;
    for (int i = 0; i < 10; i++) {
        char what[192];
        snprintf(what, sizeof(what), "Network %s-scope %s while the route is unavailable",
                 scope_name<Scope>(), names[i]);
        const op which = ops[i];
        failures += rejected_case(c, what, NVSHMEMX_ERROR_NOT_SUPPORTED, [&] {
            unsigned char *local_operand = is_get(which) ? c.out : c.source;
            launch_op<Network, Scope, kGlobal>(
                which, c.remote, local_operand, static_cast<const unsigned char *>(c.source),
                c.signal, 1, NVSHMEM_SIGNAL_SET, kBytes, issuers_for<Scope>(), c.peer, c.w.status);
        });
    }
    return failures;
}

int unavailable_cases(const route_context &c) {
    return unavailable_scope_cases<threadgroup::thread>(c) +
           unavailable_scope_cases<threadgroup::warp>(c) + unavailable_scope_cases<kBlock>(c);
}

/* A zero-length put runs the route's preflight and issues nothing. */
__global__ void probe_kernel(unsigned char *remote, unsigned char *source, int peer, int *status) {
    const int result = nvshmemx_put<uint8_t, AutoAddressing, Network, kBlock, kGlobal, kRemoteDest>(
        remote, source, 0, peer);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* Auto / Auto control: one positive data, signal, and ordering run per scope,
 * plus the direct comparison with the portable API. */
int auto_cases(const route_context &c, const compare_buffers &b) {
    int failures = 0;
    failures += scope_cases<AutoPolicy, kBlock, kGeneric>(c, "Auto", "generic(global)");
    failures += thread_warp_cases<AutoAddressing, AutoPolicy, kGeneric>(c.scoped(), "Auto");
    failures += auto_vs_stable_cases(c.scoped(), b, "Auto");
    return failures;
}

}  // namespace

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            printf("transport_specific_network: SKIP (requires at least two PEs)\n");
        }
        finalize_wrapper();
        return 0;
    }

    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    route_context c{};
    compare_buffers b{};
    c.remote = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.source = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.out = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.signal = static_cast<uint64_t *>(nvshmem_malloc(kMaxIssuers * sizeof(uint64_t)));
    b.ref_remote = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.ref_out = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.ref_signal = static_cast<uint64_t *>(nvshmem_malloc(sizeof(uint64_t)));
    int *status = nullptr;
    const bool local_ok = scratch && c.remote && c.source && c.out && c.signal && b.ref_remote &&
                          b.ref_out && b.ref_signal &&
                          cudaMalloc(&status, sizeof(*status)) == cudaSuccess &&
                          cudaMalloc(&c.private_buffer, kBytes) == cudaSuccess &&
                          cudaMalloc(&c.private_signal, sizeof(uint64_t)) == cudaSuccess;
    /* A partial symmetric allocation cannot be collectively freed. */
    if (!scratch || world_max(scratch, !local_ok) != 0) {
        fprintf(stderr, "PE %d: transport_specific_network setup failed\n", mype);
        nvshmem_global_exit(1);
        return 1;
    }
    c.w = world{mype, npes, scratch, status};
    c.active = true;
    c.peer = (mype + 1) % npes;
    c.previous = (mype + npes - 1) % npes;
    b.remote = c.remote;
    b.out = c.out;
    b.signal = c.signal;

    int failures = 0;

    /* Probe the route's capability without issuing work. */
    CUDA_CHECK(cudaMemset(status, 0xff, sizeof(*status)));
    probe_kernel<<<1, kThreads>>>(c.remote, c.source, c.peer, status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const int probe = read_status(status);
    const bool probe_failed = probe != NVSHMEMX_SUCCESS && probe != NVSHMEMX_ERROR_NOT_SUPPORTED;
    if (probe_failed) {
        fprintf(stderr, "PE %d: transport_specific_network capability probe returned %d\n", mype,
                probe);
    }
    const bool any_probe_failed = world_max(scratch, probe_failed) != 0;
    const bool unavailable = world_max(scratch, probe == NVSHMEMX_ERROR_NOT_SUPPORTED) != 0;
    if (any_probe_failed) {
        failures += 1;
        report(mype, "transport_specific_network", "Auto / Network capability probe", 1);
    } else {
        /* Memory-space and argument rejections run in every build and job. */
        const int combined =
            unavailable ? NVSHMEMX_ERROR_NOT_SUPPORTED : NVSHMEMX_ERROR_INVALID_VALUE;
        const int rejection_failures = world_max(
            scratch, memory_space_rejections(c, combined) + argument_rejections(c, combined));
        report(mype, "transport_specific_network", "Auto / Network rejections", rejection_failures);
        failures += rejection_failures;
    }
    if (any_probe_failed) {
        /* Reported above; the route's status is not trustworthy. */
    } else if (unavailable) {
        const int unavailable_failures = world_max(scratch, unavailable_cases(c));
        if (mype == 0) {
            printf(
                "transport_specific_network: SKIP Auto / Network (no device network transport: a "
                "P2P-only build, or a job NVSHMEM runs over LD/ST alone)\n");
        }
        if (unavailable_failures) {
            report(mype, "transport_specific_network", "Auto / Network unavailable status",
                   unavailable_failures);
        }
        failures += unavailable_failures;
    } else {
        /* Ring peers: one PE per node on the dedicated launcher. */
        int network_local_failures = network_cases(c, "Network");
        network_local_failures += typed_case<uint32_t>(c, "Network typed uint32 put/get_nbi");
        network_local_failures += typed_case<uint64_t>(c, "Network typed uint64 put/get_nbi");
        network_local_failures += zero_length_case(c, "Network zero-length null local");
        const int network_failures = world_max(scratch, network_local_failures);
        report(mype, "transport_specific_network", "Auto / Network", network_failures);
        failures += network_failures;

        /* The strict route also carries traffic to a P2P-mapped peer over the
         * network transport. Peers form a ring inside each TEAM_SHARED clique;
         * PEs without a mapped ring neighbour only join the barriers. */
        const int shared_npes = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
        const int shared_mype = nvshmem_team_my_pe(NVSHMEM_TEAM_SHARED);
        route_context mapped = c;
        mapped.peer = -1;
        mapped.previous = -1;
        if (shared_npes >= 2 && shared_mype >= 0) {
            mapped.peer = nvshmem_team_translate_pe(
                NVSHMEM_TEAM_SHARED, (shared_mype + 1) % shared_npes, NVSHMEM_TEAM_WORLD);
            mapped.previous = nvshmem_team_translate_pe(
                NVSHMEM_TEAM_SHARED, (shared_mype + shared_npes - 1) % shared_npes,
                NVSHMEM_TEAM_WORLD);
        }
        mapped.active = mapped.peer >= 0 && mapped.previous >= 0 &&
                        nvshmem_ptr(c.remote, mapped.peer) != nullptr &&
                        nvshmem_ptr(c.remote, mapped.previous) != nullptr;
        if (world_max(scratch, mapped.active) == 0) {
            if (mype == 0) {
                printf(
                    "transport_specific_network: Auto / Network to a P2P-mapped peer not "
                    "exercised (no PE has a mapped peer)\n");
            }
        } else {
            const int mapped_failures =
                world_max(scratch, network_cases(mapped, "Network to mapped peer"));
            report(mype, "transport_specific_network", "Auto / Network to P2P-mapped peers",
                   mapped_failures);
            failures += mapped_failures;
        }
    }

    const int auto_failures = world_max(scratch, auto_cases(c, b));
    report(mype, "transport_specific_network", "Auto / Auto", auto_failures);
    failures += auto_failures;

    nvshmem_barrier_all();
    nvshmem_free(c.remote);
    nvshmem_free(c.source);
    nvshmem_free(c.out);
    nvshmem_free(c.signal);
    nvshmem_free(b.ref_remote);
    nvshmem_free(b.ref_out);
    nvshmem_free(b.ref_signal);
    nvshmem_free(scratch);
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(c.private_buffer));
    CUDA_CHECK(cudaFree(c.private_signal));
    finalize_wrapper();
    return failures ? 1 : 0;
}
