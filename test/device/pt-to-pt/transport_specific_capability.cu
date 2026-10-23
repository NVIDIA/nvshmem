/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Functional coverage for nvshmemx_rma_capability. The source overload must
 * run exactly the put preflight and the dest overload exactly the get
 * preflight, so for every route and argument variant the kernel queries the
 * capability and then, with the same arguments, issues the operation: both
 * must return the same group-uniform status, and an operation that did not
 * succeed must leave every buffer unchanged. Capability-only runs check that
 * the query itself never moves data, whatever it returns.
 *
 * Every PE runs each case against its ring neighbour. A route whose
 * prerequisites are missing reports NOT_SUPPORTED for a valid call, which is
 * a per-case skip; on a PE whose peer has no direct mapping the Pointer routes
 * must report NOT_SUPPORTED. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_common.cuh"

using namespace ts_test;

namespace {

constexpr size_t kBytes = 4096;
using Pointer = nvshmemx::addressing::Pointer;
using CFT = nvshmemx::addressing::CFT;
using AutoAddressing = nvshmemx::addressing::Auto;
using SM = nvshmemx::transport_policy::SM;
using TMA = nvshmemx::transport_policy::TMA;
using Network = nvshmemx::transport_policy::Network;
using AutoPolicy = nvshmemx::transport_policy::Auto;
constexpr auto kThread = threadgroup::thread;
constexpr auto kWarp = threadgroup::warp;

/* Dynamic shared memory: [0, kRegistered) may be given to NVSHMEM and the
 * user payload area follows it, outside the registration. */
constexpr int kRegistered = 65536;
constexpr size_t kSmem = static_cast<size_t>(kRegistered) + kBytes;
constexpr int kNoRegistration = 0;
constexpr int kWarpThreads = 32;

/* A size and a remote offset that break the 16-byte TMA and CFT_HANDLE_TX_SIZE
 * rules while staying inside the buffers. */
constexpr size_t kUnalignedBytes = 24;
constexpr size_t kMisalignedOffset = 8;
constexpr size_t kSmallBytes = 64;

enum class direction { put, get };

/* How a case deviates from a valid call. */
enum class variant {
    valid,
    bad_pe,            /* every participant passes an out-of-range PE */
    divergent_pe,      /* only the group's last participant passes it */
    outside_heap,      /* remote operand outside the symmetric heap */
    null_local,        /* null local operand with a nonzero size */
    unaligned_size,    /* size not a 16-byte multiple */
    misaligned_remote, /* remote operand 8 bytes past a 16-byte boundary */
};

/* capability_only queries and issues nothing; the others query and then
 * issue the blocking or NBI operation, completed with the route quiet. */
enum class mode { capability_only, blocking, nbi };

struct kernel_args {
    variant which;
    mode run;
    bool in_shared;
    int registration;
    unsigned char *remote;  /* symmetric: put destination, or the peer's get source */
    unsigned char *local;   /* symmetric: put source or get destination */
    unsigned char *outside; /* device memory outside the symmetric heap */
    int peer;
    int npes;
    int *capability; /* per-thread status */
    int *operation;  /* per-thread status */
};

template <threadgroup Scope>
__host__ __device__ constexpr int participants() {
    return Scope == kThread ? 1 : Scope == kWarp ? kWarpThreads : kThreads;
}

__device__ unsigned char *smem_base() {
    extern __shared__ __align__(16) unsigned char smem[];
    return smem;
}

template <typename A, typename P, threadgroup Scope, direction Dir, LocalSpace Local>
__device__ int capability(const void *remote, const void *local, size_t bytes, int pe) {
    if constexpr (Dir == direction::put) {
        return nvshmemx_rma_capability<A, P, Scope, Local>(remote, local, bytes, pe);
    } else {
        return nvshmemx_rma_capability<A, P, Scope, local_dest(Local)>(remote, local, bytes, pe);
    }
}

template <typename A, typename P, threadgroup Scope, direction Dir, LocalSpace Local>
__device__ int operation(mode run, unsigned char *remote, unsigned char *local, size_t bytes,
                         int pe) {
    constexpr auto kLocalDest = local_dest(Local);
    int result;
    if constexpr (Dir == direction::put) {
        result = run == mode::nbi ? nvshmemx_put_nbi<uint8_t, A, P, Scope, Local, kRemoteDest>(
                                        remote, local, bytes, pe)
                                  : nvshmemx_put<uint8_t, A, P, Scope, Local, kRemoteDest>(
                                        remote, local, bytes, pe);
    } else {
        result = run == mode::nbi
                     ? nvshmemx_get_nbi<uint8_t, A, P, Scope, kRemoteSource, kLocalDest>(
                           local, remote, bytes, pe)
                     : nvshmemx_get<uint8_t, A, P, Scope, kRemoteSource, kLocalDest>(local, remote,
                                                                                     bytes, pe);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, Scope>();
    }
    return result;
}

/*
 * The participants (thread 0, warp 0, or the whole CTA) query the capability
 * and, unless the run is capability_only, issue the operation with the same
 * arguments. A shared local operand lives past the registration: a put copies
 * its source there first, and a get destination is filled with the sentinel
 * and copied back to `local` afterwards, so the host checks either placement
 * in the same buffer.
 */
template <typename A, typename P, threadgroup Scope, direction Dir, LocalSpace Local>
__global__ void capability_kernel(kernel_args a) {
    if (a.registration > 0) {
        nvshmemx_give_smem(smem_base(), a.registration);
    }
    unsigned char *payload = smem_base() + kRegistered;
    if (a.in_shared) {
        if constexpr (Dir == direction::put) {
            copy_block(payload, a.local, kBytes);
        } else {
            fill_block(payload, kBytes, kSentinelByte);
        }
    }
    __syncthreads();

    if (static_cast<int>(threadIdx.x) < participants<Scope>()) {
        unsigned char *remote = a.remote;
        unsigned char *local = a.in_shared ? payload : a.local;
        size_t bytes = kBytes;
        int pe = a.peer;
        const bool last = static_cast<int>(threadIdx.x) == participants<Scope>() - 1;
        switch (a.which) {
            case variant::valid:
                break;
            case variant::bad_pe:
                pe = a.npes;
                break;
            case variant::divergent_pe:
                pe = last ? a.npes : a.peer;
                break;
            case variant::outside_heap:
                remote = a.outside;
                break;
            case variant::null_local:
                local = nullptr;
                break;
            case variant::unaligned_size:
                bytes = kUnalignedBytes;
                break;
            case variant::misaligned_remote:
                remote += kMisalignedOffset;
                bytes = kSmallBytes;
                break;
        }
        a.capability[threadIdx.x] = capability<A, P, Scope, Dir, Local>(remote, local, bytes, pe);
        if (a.run != mode::capability_only) {
            a.operation[threadIdx.x] =
                operation<A, P, Scope, Dir, Local>(a.run, remote, local, bytes, pe);
        }
    }
    __syncthreads();

    if (Dir == direction::get && a.in_shared) {
        copy_block(a.local, payload, kBytes);
    }
    __syncthreads();
    if (a.registration > 0) {
        nvshmemx_release_smem();
    }
}

/* ---- Host side ------------------------------------------------------------ */

struct context {
    world w;
    int peer;
    int previous;
    bool peer_mapped;
    unsigned char *remote;  /* symmetric put destination */
    unsigned char *source;  /* symmetric put source and peer get source */
    unsigned char *out;     /* symmetric get destination */
    unsigned char *outside; /* cudaMalloc, outside the heap */
    int *capability;        /* device int[kThreads] */
    int *operation;         /* device int[kThreads] */
    int *flags;             /* symmetric int[2 * npes] */
};

/* The status a case must report, and whether NOT_SUPPORTED is an accepted
 * capability skip instead. */
struct expectation {
    int status;
    bool skip_if_unsupported;
};

constexpr expectation kSucceeds{NVSHMEMX_SUCCESS, true};
constexpr expectation kInvalid{NVSHMEMX_ERROR_INVALID_VALUE, false};
constexpr expectation kInvalidWhenAvailable{NVSHMEMX_ERROR_INVALID_VALUE, true};
constexpr expectation kUnsupported{NVSHMEMX_ERROR_NOT_SUPPORTED, false};

/* Each PE's flag, gathered on every PE. */
std::vector<int> gather_flags(const context &c, int local) {
    const int npes = c.w.npes;
    std::vector<int> host(npes, 0);
    host[c.w.mype] = local;
    CUDA_CHECK(cudaMemcpy(c.flags, host.data(), npes * sizeof(int), cudaMemcpyHostToDevice));
    if (nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, c.flags + npes, c.flags, npes) !=
        NVSHMEMX_SUCCESS) {
        fprintf(stderr, "PE %d: flag reduction failed\n", c.w.mype);
        nvshmem_global_exit(1);
    }
    CUDA_CHECK(cudaMemcpy(host.data(), c.flags + npes, npes * sizeof(int), cudaMemcpyDeviceToHost));
    return host;
}

const char *mode_name(mode run) {
    return run == mode::capability_only ? "capability only"
           : run == mode::blocking      ? "capability + blocking operation"
                                        : "capability + NBI operation";
}

/* Whether every participant reported the same status; stores it in *status. */
bool uniform(const std::vector<int> &statuses, int count, int *status) {
    *status = statuses[0];
    for (int i = 1; i < count; i++) {
        if (statuses[i] != *status) {
            return false;
        }
    }
    return true;
}

/* One run of a case. Returns the failure count (job-wide) and sets *skipped
 * when any PE took the NOT_SUPPORTED skip. */
template <typename A, typename P, threadgroup Scope, direction Dir, LocalSpace Local>
int run_mode(const context &c, const char *what, variant which, expectation expected,
             bool in_shared, int registration, mode run, bool *skipped) {
    constexpr int kCount = participants<Scope>();
    const bool is_put = Dir == direction::put;
    fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
    CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
    CUDA_CHECK(cudaMemset(c.out, kSentinelByte, kBytes));
    CUDA_CHECK(cudaMemset(c.outside, kSentinelByte, kBytes));
    CUDA_CHECK(cudaMemset(c.capability, 0xff, kThreads * sizeof(int)));
    CUDA_CHECK(cudaMemset(c.operation, 0xff, kThreads * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    const kernel_args args{which,
                           run,
                           in_shared,
                           registration,
                           is_put ? c.remote : c.source,
                           is_put ? c.source : c.out,
                           c.outside,
                           c.peer,
                           c.w.npes,
                           c.capability,
                           c.operation};
    auto kernel = capability_kernel<A, P, Scope, Dir, Local>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSmem)));
    kernel<<<1, kThreads, kSmem>>>(args);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    std::vector<int> capability(kThreads);
    std::vector<int> operation(kThreads);
    CUDA_CHECK(cudaMemcpy(capability.data(), c.capability, kThreads * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(operation.data(), c.operation, kThreads * sizeof(int), cudaMemcpyDeviceToHost));

    int failures = 0;
    int status = kStatusUnset;
    if (!uniform(capability, kCount, &status)) {
        fprintf(stderr, "PE %d: %s [%s]: capability status is not group-uniform\n", c.w.mype, what,
                mode_name(run));
        failures++;
    }
    if (run != mode::capability_only) {
        int op_status = kStatusUnset;
        if (!uniform(operation, kCount, &op_status)) {
            fprintf(stderr, "PE %d: %s [%s]: operation status is not group-uniform\n", c.w.mype,
                    what, mode_name(run));
            failures++;
        } else if (op_status != status) {
            fprintf(stderr, "PE %d: %s [%s]: capability returned %d but the operation %d\n",
                    c.w.mype, what, mode_name(run), status, op_status);
            failures++;
        }
    }
    bool local_skipped = false;
    if (status != expected.status) {
        if (status == NVSHMEMX_ERROR_NOT_SUPPORTED && expected.skip_if_unsupported) {
            local_skipped = true;
        } else {
            fprintf(stderr, "PE %d: %s [%s]: capability returned %d, expected %d\n", c.w.mype, what,
                    mode_name(run), status, expected.status);
            failures++;
        }
    }

    /* Data moved only where the operation ran and succeeded. */
    const int moved = run != mode::capability_only && operation[0] == NVSHMEMX_SUCCESS ? 1 : 0;
    const std::vector<int> moved_on = gather_flags(c, moved);
    if (is_put) {
        failures += moved_on[c.previous] ? check_payload(what, c.remote, kBytes, c.previous)
                                         : check_fill(what, c.remote, kBytes, kSentinelByte);
    } else {
        failures += moved ? check_payload(what, c.out, kBytes, c.peer)
                          : check_fill(what, c.out, kBytes, kSentinelByte);
    }
    failures += check_fill(what, c.outside, kBytes, kSentinelByte);
    failures += check_payload(what, c.source, kBytes, c.w.mype);

    *skipped = world_max(c.w.scratch, local_skipped) != 0;
    return world_max(c.w.scratch, failures != 0);
}

/* Runs a case capability-only and with each operation spelling. */
template <typename A, typename P, threadgroup Scope, direction Dir, LocalSpace Local>
int run_case(const context &c, const char *route, const char *label, variant which,
             expectation expected, bool in_shared = false, int registration = kNoRegistration) {
    char what[192];
    snprintf(what, sizeof(what), "%s %s-scope %s %s", route,
             Scope == kThread ? "thread"
             : Scope == kWarp ? "warp"
                              : "block",
             Dir == direction::put ? "put" : "get", label);
    int failures = 0;
    bool skipped = false;
    for (mode run : {mode::capability_only, mode::blocking, mode::nbi}) {
        bool run_skipped = false;
        failures += run_mode<A, P, Scope, Dir, Local>(c, what, which, expected, in_shared,
                                                      registration, run, &run_skipped);
        skipped |= run_skipped;
    }
    if (skipped && c.w.mype == 0) {
        printf("%s: SKIP on at least one PE (route capability not supported)\n", what);
    }
    return failures;
}

/* Argument problems every route rejects with INVALID_VALUE before any
 * capability check. When only one participant passes a bad PE, the others
 * reach the capability check, so the group agrees on the more severe
 * NOT_SUPPORTED where the route is unavailable: `divergent` says which. */
template <typename A, typename P, direction Dir, LocalSpace Local>
int argument_cases(const context &c, const char *route, bool in_shared, int registration,
                   expectation divergent) {
    int failures = 0;
    failures += run_case<A, P, kBlock, Dir, Local>(c, route, "invalid PE", variant::bad_pe,
                                                   kInvalid, in_shared, registration);
    failures += run_case<A, P, kBlock, Dir, Local>(c, route, "divergent PE", variant::divergent_pe,
                                                   divergent, in_shared, registration);
    failures += run_case<A, P, kBlock, Dir, Local>(
        c, route, "remote outside heap", variant::outside_heap, kInvalid, in_shared, registration);
    failures += run_case<A, P, kBlock, Dir, Local>(c, route, "null local", variant::null_local,
                                                   kInvalid, in_shared, registration);
    return failures;
}

/* Size and alignment rules of the TMA and CFT data paths. */
template <typename A, typename P, direction Dir, LocalSpace Local>
int alignment_cases(const context &c, const char *route, bool in_shared) {
    return run_case<A, P, kBlock, Dir, Local>(c, route, "unaligned size", variant::unaligned_size,
                                              kInvalidWhenAvailable, in_shared, kRegistered) +
           run_case<A, P, kBlock, Dir, Local>(c, route, "misaligned remote",
                                              variant::misaligned_remote, kInvalidWhenAvailable,
                                              in_shared, kRegistered);
}

/* On a Pointer route a peer without a direct mapping is a missing
 * capability, not a skip: a valid call, and every participant of a group in
 * which only one passes a bad argument, must report NOT_SUPPORTED. */
expectation pointer_valid(const context &c) { return c.peer_mapped ? kSucceeds : kUnsupported; }
expectation pointer_divergent(const context &c, expectation mapped) {
    return c.peer_mapped ? mapped : kUnsupported;
}

int pointer_sm_cases(const context &c) {
    const char *route = "Pointer/SM";
    const expectation valid = pointer_valid(c);
    int failures = 0;
    failures += run_case<Pointer, SM, kBlock, direction::put, kGlobal>(c, route, "global",
                                                                       variant::valid, valid);
    failures += run_case<Pointer, SM, kBlock, direction::put, kShared>(c, route, "shared",
                                                                       variant::valid, valid, true);
    failures += run_case<Pointer, SM, kBlock, direction::put, kGeneric>(
        c, route, "generic(shared)", variant::valid, valid, true);
    failures += run_case<Pointer, SM, kBlock, direction::get, kGlobal>(c, route, "global",
                                                                       variant::valid, valid);
    failures += run_case<Pointer, SM, kBlock, direction::get, kShared>(c, route, "shared",
                                                                       variant::valid, valid, true);
    failures += run_case<Pointer, SM, kBlock, direction::get, kGeneric>(c, route, "generic(global)",
                                                                        variant::valid, valid);
    failures += run_case<Pointer, SM, kWarp, direction::put, kGlobal>(c, route, "global",
                                                                      variant::valid, valid);
    failures += run_case<Pointer, SM, kWarp, direction::get, kShared>(
        c, route, "divergent PE", variant::divergent_pe, pointer_divergent(c, kInvalid), true);
    failures += run_case<Pointer, SM, kThread, direction::get, kGlobal>(c, route, "global",
                                                                        variant::valid, valid);
    failures += argument_cases<Pointer, SM, direction::put, kGlobal>(
        c, route, false, kNoRegistration, pointer_divergent(c, kInvalid));
    failures += argument_cases<Pointer, SM, direction::get, kGeneric>(
        c, route, false, kNoRegistration, pointer_divergent(c, kInvalid));
    return failures;
}

/* Pointer / TMA, reached through Pointer / TMA or the Auto / TMA alias. */
template <typename A>
int pointer_tma_cases(const context &c, const char *route) {
    const expectation valid = pointer_valid(c);
    int failures = 0;
    failures += run_case<A, TMA, kBlock, direction::put, kShared>(
        c, route, "shared", variant::valid, valid, true, kRegistered);
    failures += run_case<A, TMA, kBlock, direction::put, kGlobal>(
        c, route, "global", variant::valid, valid, false, kRegistered);
    failures += run_case<A, TMA, kBlock, direction::put, kGeneric>(
        c, route, "generic(global)", variant::valid, valid, false, kRegistered);
    failures += run_case<A, TMA, kBlock, direction::get, kShared>(
        c, route, "shared", variant::valid, valid, true, kRegistered);
    failures += run_case<A, TMA, kBlock, direction::get, kGlobal>(
        c, route, "global", variant::valid, valid, false, kRegistered);
    failures += run_case<A, TMA, kBlock, direction::get, kGeneric>(
        c, route, "generic(shared)", variant::valid, valid, true, kRegistered);
    /* Below block scope only a shared put source is legal, and a generic
     * one that is not in shared memory is INVALID_VALUE. */
    failures += run_case<A, TMA, kWarp, direction::put, kShared>(c, route, "shared", variant::valid,
                                                                 valid, true, kRegistered);
    failures += run_case<A, TMA, kThread, direction::put, kGeneric>(
        c, route, "generic(shared)", variant::valid, valid, true, kRegistered);
    failures += run_case<A, TMA, kWarp, direction::put, kGeneric>(
        c, route, "generic(global)", variant::valid,
        c.peer_mapped ? kInvalidWhenAvailable : kUnsupported, false, kRegistered);
    failures += run_case<A, TMA, kThread, direction::put, kGeneric>(
        c, route, "generic(global)", variant::valid,
        c.peer_mapped ? kInvalidWhenAvailable : kUnsupported, false, kRegistered);
    failures += run_case<A, TMA, kWarp, direction::put, kShared>(
        c, route, "divergent PE", variant::divergent_pe,
        pointer_divergent(c, kInvalidWhenAvailable), true, kRegistered);
    /* Without registered shared memory the route is unavailable. */
    failures += run_case<A, TMA, kBlock, direction::put, kShared>(
        c, route, "shared without registration", variant::valid, kUnsupported, true,
        kNoRegistration);
    failures += run_case<A, TMA, kBlock, direction::get, kGlobal>(
        c, route, "global without registration", variant::valid, kUnsupported, false,
        kNoRegistration);
    failures += argument_cases<A, TMA, direction::put, kShared>(
        c, route, true, kRegistered, pointer_divergent(c, kInvalidWhenAvailable));
    failures += argument_cases<A, TMA, direction::get, kGlobal>(
        c, route, false, kRegistered, pointer_divergent(c, kInvalidWhenAvailable));
    failures += alignment_cases<A, TMA, direction::put, kShared>(c, route, true);
    failures += alignment_cases<A, TMA, direction::get, kGlobal>(c, route, false);
    return failures;
}

/* CFT / TMA, block scope only. In a build without CFT handles every valid
 * call is NOT_SUPPORTED. */
int cft_tma_cases(const context &c) {
    const char *route = "CFT/TMA";
    int failures = 0;
    failures += run_case<CFT, TMA, kBlock, direction::put, kGlobal>(
        c, route, "global", variant::valid, kSucceeds, false, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::put, kShared>(
        c, route, "shared", variant::valid, kSucceeds, true, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::put, kGeneric>(
        c, route, "generic(global)", variant::valid, kSucceeds, false, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::get, kGlobal>(
        c, route, "global", variant::valid, kSucceeds, false, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::get, kShared>(
        c, route, "shared", variant::valid, kSucceeds, true, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::get, kGeneric>(
        c, route, "generic(shared)", variant::valid, kSucceeds, true, kRegistered);
    failures += run_case<CFT, TMA, kBlock, direction::put, kGlobal>(
        c, route, "global without registration", variant::valid, kUnsupported, false,
        kNoRegistration);
    failures += run_case<CFT, TMA, kBlock, direction::get, kShared>(
        c, route, "shared without registration", variant::valid, kUnsupported, true,
        kNoRegistration);
    failures += argument_cases<CFT, TMA, direction::put, kGlobal>(c, route, false, kRegistered,
                                                                  kInvalidWhenAvailable);
    failures += argument_cases<CFT, TMA, direction::get, kShared>(c, route, true, kRegistered,
                                                                  kInvalidWhenAvailable);
    failures += alignment_cases<CFT, TMA, direction::put, kGlobal>(c, route, false);
    failures += alignment_cases<CFT, TMA, direction::get, kShared>(c, route, true);
    return failures;
}

/* Auto / Network. A shared local operand is INVALID_VALUE in every build and
 * job. */
template <typename A>
int network_cases(const context &c, const char *route) {
    int failures = 0;
    failures += run_case<A, Network, kBlock, direction::put, kGlobal>(c, route, "global",
                                                                      variant::valid, kSucceeds);
    failures += run_case<A, Network, kBlock, direction::put, kGeneric>(c, route, "generic(global)",
                                                                       variant::valid, kSucceeds);
    failures += run_case<A, Network, kBlock, direction::get, kGlobal>(c, route, "global",
                                                                      variant::valid, kSucceeds);
    failures += run_case<A, Network, kWarp, direction::get, kGeneric>(c, route, "generic(global)",
                                                                      variant::valid, kSucceeds);
    failures += run_case<A, Network, kThread, direction::put, kGlobal>(c, route, "global",
                                                                       variant::valid, kSucceeds);
    failures += run_case<A, Network, kBlock, direction::put, kShared>(
        c, route, "shared", variant::valid, kInvalid, true);
    failures += run_case<A, Network, kBlock, direction::get, kShared>(
        c, route, "shared", variant::valid, kInvalid, true);
    failures += run_case<A, Network, kWarp, direction::put, kGeneric>(
        c, route, "generic(shared)", variant::valid, kInvalid, true);
    failures += run_case<A, Network, kBlock, direction::get, kGeneric>(
        c, route, "generic(shared)", variant::valid, kInvalid, true);
    failures += run_case<A, Network, kWarp, direction::put, kGlobal>(
        c, route, "divergent PE", variant::divergent_pe, kInvalidWhenAvailable);
    failures += argument_cases<A, Network, direction::put, kGlobal>(
        c, route, false, kNoRegistration, kInvalidWhenAvailable);
    failures += argument_cases<A, Network, direction::get, kGlobal>(
        c, route, false, kNoRegistration, kInvalidWhenAvailable);
    return failures;
}

/* Auto / Auto and the pairs that lower to it. The portable route keeps its
 * fallback, so a valid call always succeeds. */
template <typename A, typename P>
int automatic_cases(const context &c, const char *route, bool full) {
    int failures = 0;
    failures += run_case<A, P, kBlock, direction::put, kGeneric>(c, route, "generic(global)",
                                                                 variant::valid, kSucceeds);
    failures += run_case<A, P, kBlock, direction::get, kGeneric>(c, route, "generic(global)",
                                                                 variant::valid, kSucceeds);
    failures += run_case<A, P, kBlock, direction::put, kGeneric>(c, route, "invalid PE",
                                                                 variant::bad_pe, kInvalid);
    if (full) {
        failures += run_case<A, P, kBlock, direction::put, kGlobal>(c, route, "global",
                                                                    variant::valid, kSucceeds);
        failures += run_case<A, P, kWarp, direction::get, kGlobal>(c, route, "global",
                                                                   variant::valid, kSucceeds);
        failures += run_case<A, P, kThread, direction::put, kGeneric>(c, route, "generic(global)",
                                                                      variant::valid, kSucceeds);
        failures += run_case<A, P, kWarp, direction::put, kGeneric>(
            c, route, "divergent PE", variant::divergent_pe, kInvalid);
        failures += argument_cases<A, P, direction::put, kGeneric>(c, route, false, kNoRegistration,
                                                                   kInvalid);
        failures += argument_cases<A, P, direction::get, kGlobal>(c, route, false, kNoRegistration,
                                                                  kInvalid);
    }
    return failures;
}

void report_route(const context &c, const char *route, int failures) {
    report(c.w.mype, "transport_specific_capability", route, world_max(c.w.scratch, failures));
}

}  // namespace

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            printf("transport_specific_capability: SKIP (requires at least two PEs)\n");
        }
        finalize_wrapper();
        return 0;
    }

    constexpr size_t kAlignment = 16;
    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    context c{};
    c.remote = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.source = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.out = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.flags = static_cast<int *>(nvshmem_malloc(2 * npes * sizeof(int)));
    int *status = nullptr;
    const bool local_ok = scratch && c.remote && c.source && c.out && c.flags &&
                          cudaMalloc(&status, sizeof(*status)) == cudaSuccess &&
                          cudaMalloc(&c.outside, kBytes) == cudaSuccess &&
                          cudaMalloc(&c.capability, kThreads * sizeof(int)) == cudaSuccess &&
                          cudaMalloc(&c.operation, kThreads * sizeof(int)) == cudaSuccess;
    /* A partial symmetric allocation cannot be collectively freed. */
    if (!scratch || world_max(scratch, !local_ok) != 0) {
        fprintf(stderr, "PE %d: transport_specific_capability setup failed\n", mype);
        nvshmem_global_exit(1);
        return 1;
    }
    c.w = world{mype, npes, scratch, status};
    c.peer = (mype + 1) % npes;
    c.previous = (mype + npes - 1) % npes;
    c.peer_mapped = nvshmem_ptr(c.remote, c.peer) != nullptr;
    if (world_max(scratch, !c.peer_mapped) != 0 && mype == 0) {
        printf(
            "transport_specific_capability: a ring peer has no direct mapping; the Pointer "
            "routes must report NOT_SUPPORTED there\n");
    }

    int failures = 0;
    int route_failures = pointer_sm_cases(c);
    report_route(c, "Pointer / SM", route_failures);
    failures += route_failures;

    route_failures = pointer_tma_cases<Pointer>(c, "Pointer/TMA");
    route_failures += pointer_tma_cases<AutoAddressing>(c, "Auto/TMA");
    report_route(c, "Pointer / TMA and Auto / TMA", route_failures);
    failures += route_failures;

    route_failures = cft_tma_cases(c);
    report_route(c, "CFT / TMA", route_failures);
    failures += route_failures;

    route_failures = network_cases<AutoAddressing>(c, "Auto/Network");
    report_route(c, "Auto / Network", route_failures);
    failures += route_failures;

    route_failures = automatic_cases<AutoAddressing, AutoPolicy>(c, "Auto/Auto", true);
    route_failures += automatic_cases<Pointer, AutoPolicy>(c, "Pointer/Auto", false);
    route_failures += automatic_cases<CFT, AutoPolicy>(c, "CFT/Auto", false);
    route_failures += automatic_cases<AutoAddressing, SM>(c, "Auto/SM", false);
    report_route(c, "Auto / Auto and its aliases", route_failures);
    failures += route_failures;

    nvshmem_barrier_all();
    nvshmem_free(c.remote);
    nvshmem_free(c.source);
    nvshmem_free(c.out);
    nvshmem_free(c.flags);
    nvshmem_free(scratch);
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(c.outside));
    CUDA_CHECK(cudaFree(c.capability));
    CUDA_CHECK(cudaFree(c.operation));
    finalize_wrapper();
    return failures ? 1 : 0;
}
