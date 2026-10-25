/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Shared scaffolding for the transport-policy route perftests: route
 * descriptors, route availability through nvshmemx_rma_capability, job-wide
 * status agreement, and a size-sweep driver that reports through the standard
 * perftest tables. The `stable` pseudo-route issues the same operations
 * through the stable device API (nvshmem_putmem, nvshmemx_putmem_block,
 * nvshmemx_quiet_block, ...) inside the same kernels, so its rows are the
 * baseline for every route. */

#ifndef TRANSPORT_SPECIFIC_PERFTEST_CUH
#define TRANSPORT_SPECIFIC_PERFTEST_CUH

#ifndef NVSHMEMX_ENABLE_TRANSPORT_POLICY
#error "Define NVSHMEMX_ENABLE_TRANSPORT_POLICY before including transport_specific_perftest.cuh"
#endif

#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <type_traits>
#include <vector>

#include "utils.h"

namespace transport_specific {

namespace Addressing = nvshmemx::addressing;
namespace Policy = nvshmemx::transport_policy;
namespace Memory = nvshmemx::memory_space;

using Scope = nvshmemx::threadgroup;
constexpr auto kBlock = Scope::block;
/* The TMA alignment and size granule, which is also the CFT handle transfer
 * size (CFT_HANDLE_TX_SIZE). */
constexpr size_t kAlignment = 16;
/* Headroom for the kernels' static shared memory when sizing shared payloads. */
constexpr size_t kStaticSmemReserve = 1024;
/* Status slots start out unset so a block that never published is visible. */
constexpr int kStatusFill = 0xff;

enum class route { auto_auto, pointer_sm, pointer_tma, cft_tma, auto_network, stable, count };
enum class placement { global, shared };
enum class rma_operation { put, get };
enum class rma_nbi { no, yes };
enum class completion { none, quiet, flush, fence };
enum class outcome { success, skipped, failed };

/* option: the NVSHMEM_TRANSPORT_SPECIFIC_ROUTE name; key: the subjob prefix;
 * requirement: what the route needs when its capability is missing. */
struct route_info {
    const char *option;
    const char *key;
    const char *label;
    const char *requirement;
};

constexpr route_info kRouteInfo[] = {
    {"auto", "auto_auto", "Auto / Auto", "the portable route is always available"},
    {"sm", "pointer_sm", "Pointer / SM", "needs a directly mapped peer"},
    {"tma", "pointer_tma", "Pointer / TMA",
     "needs sm_90+, NVSHMEM_TMA_POLICY=ENABLE, registered shared memory, and a directly "
     "mapped peer"},
    {"cft", "cft_tma", "CFT / TMA",
     "needs a CFT-enabled build on a capable GPU, NVSHMEM_ENABLE_LOGICAL_ENDPOINT=1, "
     "NVSHMEM_TMA_POLICY=ENABLE, and registered shared memory"},
    {"network", "auto_network", "Auto / Network",
     "needs a device network transport (a job not run over P2P alone)"},
    {"stable", "stable_api", "Stable API", "the stable device API is always available"},
};
static_assert(sizeof(kRouteInfo) / sizeof(kRouteInfo[0]) == static_cast<size_t>(route::count),
              "route descriptor table is incomplete");

inline const route_info &info(route r) { return kRouteInfo[static_cast<int>(r)]; }
constexpr unsigned route_bit(route r) { return 1u << static_cast<int>(r); }
constexpr unsigned kAllRoutes = (1u << static_cast<int>(route::count)) - 1;

inline const char *placement_key(placement p) {
    return p == placement::shared ? "shared" : "global";
}

/* registers_smem: the route consumes a CTA shared-memory registration.
 * granule: payload sizes and per-block slices must be multiples of this.
 * shared_local: the route accepts a CTA shared-memory local operand. */
template <route R>
struct route_traits;

template <>
struct route_traits<route::auto_auto> {
    using addressing = Addressing::Auto;
    using policy = Policy::Auto;
    static constexpr bool registers_smem = false;
    static constexpr size_t granule = 1;
    static constexpr bool shared_local = false;
};

template <>
struct route_traits<route::pointer_sm> {
    using addressing = Addressing::Pointer;
    using policy = Policy::SM;
    static constexpr bool registers_smem = false;
    static constexpr size_t granule = 1;
    static constexpr bool shared_local = true;
};

template <>
struct route_traits<route::pointer_tma> {
    using addressing = Addressing::Pointer;
    using policy = Policy::TMA;
    static constexpr bool registers_smem = true;
    static constexpr size_t granule = kAlignment;
    static constexpr bool shared_local = true;
};

template <>
struct route_traits<route::cft_tma> {
    using addressing = Addressing::CFT;
    using policy = Policy::TMA;
    static constexpr bool registers_smem = true;
    static constexpr size_t granule = kAlignment;
    static constexpr bool shared_local = true;
};

template <>
struct route_traits<route::auto_network> {
    using addressing = Addressing::Auto;
    using policy = Policy::Network;
    static constexpr bool registers_smem = false;
    static constexpr size_t granule = 1;
    static constexpr bool shared_local = false;
};

/* The stable pseudo-route's addressing and policy: the call helpers below
 * issue the stable device API instead of a transport-policy entry point. */
struct stable_api {};

template <>
struct route_traits<route::stable> {
    using addressing = stable_api;
    using policy = stable_api;
    static constexpr bool registers_smem = false;
    static constexpr size_t granule = 1;
    static constexpr bool shared_local = false;
};

template <typename A>
constexpr bool is_stable = std::is_same<A, stable_api>::value;

/* Whether route R accepts the operation at scope S with local placement P
 * (the routes' static scope rules): CFT / TMA is block-only and Pointer / TMA
 * below block scope moves only puts from a shared source. */
template <route R, Scope S, placement P, rma_operation Op>
constexpr bool scope_legal =
    S == Scope::block ||
    (R != route::cft_tma &&
     (R != route::pointer_tma || (Op == rma_operation::put && P == placement::shared)));

inline const char *scope_key(Scope s) {
    return s == Scope::thread ? "thread" : s == Scope::warp ? "warp" : "block";
}

/* Threads per CTA for a single-group latency kernel at scope S. */
constexpr int scope_threads(Scope s, int block_threads) {
    return s == Scope::thread ? 1 : s == Scope::warp ? 32 : block_threads;
}

/* Auto / Auto keeps generic local classification; every strict route gets
 * the explicit memory space that matches where the local operand lives. The
 * local operand is a put source or, via local_dest(), a get destination. */
template <route R, placement P>
constexpr Memory::source local_space = R == route::auto_auto    ? Memory::source::generic
                                       : P == placement::shared ? Memory::source::shared
                                                                : Memory::source::global;

__host__ __device__ constexpr Memory::dest local_dest(Memory::source local) {
    return local == Memory::source::shared   ? Memory::dest::shared
           : local == Memory::source::global ? Memory::dest::global
                                             : Memory::dest::generic;
}

/* The stable device API at scope S: the thread, _warp, or _block form. */
template <Scope S, rma_operation Op, rma_nbi Nbi>
__device__ void stable_rma(uint8_t *remote, uint8_t *local, size_t bytes, int pe) {
    constexpr bool kPut = Op == rma_operation::put;
    constexpr bool kNbi = Nbi == rma_nbi::yes;
    if constexpr (S == Scope::thread) {
        if constexpr (kPut) {
            kNbi ? nvshmem_putmem_nbi(remote, local, bytes, pe)
                 : nvshmem_putmem(remote, local, bytes, pe);
        } else {
            kNbi ? nvshmem_getmem_nbi(local, remote, bytes, pe)
                 : nvshmem_getmem(local, remote, bytes, pe);
        }
    } else if constexpr (S == Scope::warp) {
        if constexpr (kPut) {
            kNbi ? nvshmemx_putmem_nbi_warp(remote, local, bytes, pe)
                 : nvshmemx_putmem_warp(remote, local, bytes, pe);
        } else {
            kNbi ? nvshmemx_getmem_nbi_warp(local, remote, bytes, pe)
                 : nvshmemx_getmem_warp(local, remote, bytes, pe);
        }
    } else {
        if constexpr (kPut) {
            kNbi ? nvshmemx_putmem_nbi_block(remote, local, bytes, pe)
                 : nvshmemx_putmem_block(remote, local, bytes, pe);
        } else {
            kNbi ? nvshmemx_getmem_nbi_block(local, remote, bytes, pe)
                 : nvshmemx_getmem_block(local, remote, bytes, pe);
        }
    }
}

template <rma_operation Op, rma_nbi Nbi>
struct rma_call {
    template <typename A, typename P, Memory::source Local, Scope S = kBlock>
    __device__ static int invoke(uint8_t *remote, uint8_t *local, size_t bytes, int pe) {
        constexpr bool kNbi = Nbi == rma_nbi::yes;
        if constexpr (is_stable<A>) {
            stable_rma<S, Op, Nbi>(remote, local, bytes, pe);
            return NVSHMEMX_SUCCESS;
        } else if constexpr (Op == rma_operation::put && kNbi) {
            return nvshmemx_put_nbi<uint8_t, A, P, S, Local, Memory::dest::global>(remote, local,
                                                                                   bytes, pe);
        } else if constexpr (Op == rma_operation::put) {
            return nvshmemx_put<uint8_t, A, P, S, Local, Memory::dest::global>(remote, local, bytes,
                                                                               pe);
        } else if constexpr (kNbi) {
            return nvshmemx_get_nbi<uint8_t, A, P, S, Memory::source::global, local_dest(Local)>(
                local, remote, bytes, pe);
        } else {
            return nvshmemx_get<uint8_t, A, P, S, Memory::source::global, local_dest(Local)>(
                local, remote, bytes, pe);
        }
    }
};

template <Scope S, rma_nbi Nbi>
__device__ void stable_put_signal(uint8_t *remote, uint8_t *local, size_t bytes, uint64_t *sig_addr,
                                  uint64_t signal, int sig_op, int pe) {
    constexpr bool kNbi = Nbi == rma_nbi::yes;
    if constexpr (S == Scope::thread) {
        kNbi ? nvshmem_putmem_signal_nbi(remote, local, bytes, sig_addr, signal, sig_op, pe)
             : nvshmem_putmem_signal(remote, local, bytes, sig_addr, signal, sig_op, pe);
    } else if constexpr (S == Scope::warp) {
        kNbi ? nvshmemx_putmem_signal_nbi_warp(remote, local, bytes, sig_addr, signal, sig_op, pe)
             : nvshmemx_putmem_signal_warp(remote, local, bytes, sig_addr, signal, sig_op, pe);
    } else {
        kNbi ? nvshmemx_putmem_signal_nbi_block(remote, local, bytes, sig_addr, signal, sig_op, pe)
             : nvshmemx_putmem_signal_block(remote, local, bytes, sig_addr, signal, sig_op, pe);
    }
}

template <rma_nbi Nbi>
struct put_signal_call {
    template <typename A, typename P, Memory::source Local, Scope S = kBlock>
    __device__ static int invoke(uint8_t *remote, uint8_t *local, size_t bytes, uint64_t *sig_addr,
                                 uint64_t signal, int sig_op, int pe) {
        if constexpr (is_stable<A>) {
            stable_put_signal<S, Nbi>(remote, local, bytes, sig_addr, signal, sig_op, pe);
            return NVSHMEMX_SUCCESS;
        } else if constexpr (Nbi == rma_nbi::yes) {
            return nvshmemx_put_signal_nbi<uint8_t, A, P, S, Local, Memory::dest::global>(
                remote, local, bytes, sig_addr, signal, sig_op, pe);
        } else {
            return nvshmemx_put_signal<uint8_t, A, P, S, Local, Memory::dest::global>(
                remote, local, bytes, sig_addr, signal, sig_op, pe);
        }
    }
};

/* The stable nvshmemx_signal_op is a single-thread call; at warp and block
 * scope the leader issues it and the group synchronizes, as Auto / Auto does. */
template <typename A, typename P, Scope S = kBlock>
__device__ int signal_op_call(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    if constexpr (is_stable<A>) {
        if constexpr (S == Scope::thread) {
            nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
        } else if constexpr (S == Scope::warp) {
            if ((threadIdx.x % warpSize) == 0) {
                nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
            }
            __syncwarp();
        } else {
            if (threadIdx.x == 0) {
                nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
            }
            __syncthreads();
        }
        return NVSHMEMX_SUCCESS;
    } else {
        return nvshmemx_signal_op<A, P, S>(sig_addr, signal, sig_op, pe);
    }
}

template <completion C>
struct completion_call {
    template <typename A, typename P, Scope S = kBlock>
    __device__ static int invoke() {
        if constexpr (C == completion::none) {
            return NVSHMEMX_SUCCESS;
        } else if constexpr (is_stable<A>) {
            if constexpr (C == completion::quiet) {
                S == Scope::thread ? nvshmem_quiet()
                : S == Scope::warp ? nvshmemx_quiet_warp()
                                   : nvshmemx_quiet_block();
            } else if constexpr (C == completion::flush) {
                S == Scope::thread ? nvshmemx_flush()
                : S == Scope::warp ? nvshmemx_flush_warp()
                                   : nvshmemx_flush_block();
            } else {
                S == Scope::thread ? nvshmem_fence()
                : S == Scope::warp ? nvshmemx_fence_warp()
                                   : nvshmemx_fence_block();
            }
            return NVSHMEMX_SUCCESS;
        } else if constexpr (C == completion::quiet) {
            return nvshmemx_quiet<A, P, S>();
        } else if constexpr (C == completion::flush) {
            return nvshmemx_flush<A, P, S>();
        } else {
            return nvshmemx_fence<A, P, S>();
        }
    }
};

inline const char *completion_key(completion c) {
    switch (c) {
        case completion::quiet:
            return "quiet";
        case completion::flush:
            return "flush";
        case completion::fence:
            return "fence";
        default:
            return "none";
    }
}

/* Dynamic shared memory is laid out as [registration][payload]; the payload
 * therefore never aliases NVSHMEM's barrier or staging area. */
template <route R>
__device__ unsigned char *acquire_smem(int registration) {
    extern __shared__ __align__(16) unsigned char transport_specific_smem[];
    if constexpr (route_traits<R>::registers_smem) {
        nvshmemx_give_smem(transport_specific_smem, registration);
        __syncthreads();
        return transport_specific_smem + registration;
    } else {
        return transport_specific_smem;
    }
}

template <route R>
__device__ void release_smem() {
    if constexpr (route_traits<R>::registers_smem) {
        __syncthreads();
        nvshmemx_release_smem();
        __syncthreads();
    }
}

/* Block-uniform early exit so no thread skips a block-scoped call its peers
 * still make. */
__device__ inline bool block_failed(int result) {
    return __syncthreads_or(result != NVSHMEMX_SUCCESS) != 0;
}

__device__ inline void publish_block_status(int result, int *statuses) {
    __shared__ int block_status;
    if (threadIdx.x == 0) {
        block_status = NVSHMEMX_SUCCESS;
    }
    __syncthreads();
    if (result != NVSHMEMX_SUCCESS) {
        block_status = result;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        statuses[blockIdx.x] = block_status;
    }
}

__device__ inline void block_slice(size_t bytes, size_t granule, size_t *offset, size_t *length) {
    const size_t units = bytes / granule;
    const size_t blocks = gridDim.x;
    const size_t block = blockIdx.x;
    const size_t base = units / blocks;
    const size_t remainder = units % blocks;
    *offset = (block * base + (block < remainder ? block : remainder)) * granule;
    *length = (base + (block < remainder ? 1 : 0)) * granule;
}

/* The route's own answer for one operation in the benchmark's launch shape
 * (registration, block size, and local operand placement): exactly the
 * preflight the operation runs, without issuing work. */
template <route R, placement P, rma_operation Op, Scope S = kBlock>
__global__ void rma_capability_kernel(unsigned char *remote, unsigned char *local_global,
                                      size_t bytes, int registration, int peer, int *statuses) {
    using traits = route_traits<R>;
    using A = typename traits::addressing;
    using Pol = typename traits::policy;
    constexpr Memory::source kLocal = local_space<R, P>;

    unsigned char *shared = acquire_smem<R>(registration);
    unsigned char *local = P == placement::shared ? shared : local_global;
    int result;
    if constexpr (is_stable<A>) {
        result = NVSHMEMX_SUCCESS;
    } else if constexpr (Op == rma_operation::put) {
        result = nvshmemx_rma_capability<A, Pol, S, kLocal>(remote, local, bytes, peer);
    } else {
        result = nvshmemx_rma_capability<A, Pol, S, local_dest(kLocal)>(remote, local, bytes, peer);
    }
    release_smem<R>();
    publish_block_status(result, statuses);
}

/* NVSHMEM_TRANSPORT_SPECIFIC_ROUTE is a comma-separated subset of
 * auto,sm,tma,cft,network,stable (cft is CFT / TMA; stable is the stable
 * device API baseline), or "all" (the default). */
inline bool parse_route_selection(unsigned *mask) {
    const char *value = getenv("NVSHMEM_TRANSPORT_SPECIFIC_ROUTE");
    if (value == nullptr || strcmp(value, "all") == 0) {
        *mask = kAllRoutes;
        return true;
    }
    unsigned selected = 0;
    std::string list(value);
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = std::min(list.find(',', begin), list.size());
        const std::string name = list.substr(begin, end - begin);
        bool known = name == "all";
        if (known) {
            selected |= kAllRoutes;
        }
        for (int r = 0; r < static_cast<int>(route::count) && !known; r++) {
            if (name == kRouteInfo[r].option) {
                selected |= route_bit(static_cast<route>(r));
                known = true;
            }
        }
        if (!known) {
            return false;
        }
        begin = end + 1;
    }
    *mask = selected;
    return selected != 0;
}

/* TMA and CFT need a TMA-enabled runtime. Respect an explicit user policy so
 * that NVSHMEM_TMA_POLICY=DISABLE still produces a TMA/CFT SKIP. */
inline void default_tma_policy(unsigned mask) {
    if (mask & (route_bit(route::pointer_tma) | route_bit(route::cft_tma))) {
        setenv("NVSHMEM_TMA_POLICY", "ENABLE", 0);
    }
}

struct context {
    const char *benchmark;
    unsigned selection;
    int mype;
    int npes;
    int peer;
    int *state;
    int *statuses;
    int max_blocks;
    int registration;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaDeviceProp properties;

    bool selected(route r) const { return (selection & route_bit(r)) != 0; }
};

/* Job-wide OR. Every PE must call this in the same order. */
inline bool any_pe(context &ctx, bool local) {
    const int value = local ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(ctx.state, &value, sizeof(value), cudaMemcpyHostToDevice));
    if (nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, ctx.state + 1, ctx.state, 1) != 0) {
        fprintf(stderr, "PE %d: %s: status reduction failed\n", ctx.mype, ctx.benchmark);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    int global = 0;
    CUDA_CHECK(cudaMemcpy(&global, ctx.state + 1, sizeof(global), cudaMemcpyDeviceToHost));
    return global != 0;
}

inline int read_status(const int *statuses, int count) {
    std::vector<int> host(count);
    CUDA_CHECK(
        cudaMemcpy(host.data(), statuses, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int status : host) {
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
    }
    return NVSHMEMX_SUCCESS;
}

/* Returns false (after printing the reason) when the job cannot run the
 * benchmark at all; the caller then exits with the returned code. */
inline bool setup_context(context &ctx, const char *benchmark, int max_blocks, int *exit_code) {
    ctx.benchmark = benchmark;
    ctx.mype = nvshmem_my_pe();
    ctx.npes = nvshmem_n_pes();
    ctx.peer = (ctx.mype + 1) % (ctx.npes > 0 ? ctx.npes : 1);
    ctx.max_blocks = max_blocks > 0 ? max_blocks : 1;
    ctx.registration = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
    *exit_code = 0;

    if (!parse_route_selection(&ctx.selection)) {
        if (ctx.mype == 0) {
            fprintf(stderr,
                    "%s: NVSHMEM_TRANSPORT_SPECIFIC_ROUTE must be a comma-separated list of "
                    "auto, sm, tma, cft, network, stable, or all\n",
                    benchmark);
        }
        *exit_code = 1;
        return false;
    }
    if (ctx.npes != 2) {
        if (ctx.mype == 0) {
            printf("%s: SKIP (requires exactly two PEs)\n", benchmark);
        }
        return false;
    }
    if (min_size == 0 || min_size > max_size || step_factor <= 1 || iters == 0 ||
        repetitions == 0) {
        if (ctx.mype == 0) {
            fprintf(stderr, "%s: invalid size or iteration arguments\n", benchmark);
        }
        *exit_code = 1;
        return false;
    }

    ctx.state = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    if (ctx.state == nullptr) {
        fprintf(stderr, "PE %d: %s: state allocation failed\n", ctx.mype, benchmark);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    int device = 0;
    const bool failed =
        cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&ctx.properties, device) != cudaSuccess ||
        cudaMalloc(&ctx.statuses, static_cast<size_t>(ctx.max_blocks) * sizeof(int)) !=
            cudaSuccess ||
        cudaEventCreate(&ctx.start) != cudaSuccess || cudaEventCreate(&ctx.stop) != cudaSuccess;
    if (any_pe(ctx, failed)) {
        fprintf(stderr, "PE %d: %s: setup failed\n", ctx.mype, benchmark);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    return true;
}

inline void teardown_context(context &ctx) {
    nvshmem_barrier_all();
    nvshmem_free(ctx.state);
    CUDA_CHECK(cudaFree(ctx.statuses));
    CUDA_CHECK(cudaEventDestroy(ctx.start));
    CUDA_CHECK(cudaEventDestroy(ctx.stop));
}

inline size_t registration_bytes(const context &ctx, route r) {
    return (r == route::pointer_tma || r == route::cft_tma) ? static_cast<size_t>(ctx.registration)
                                                            : 0;
}

/* Per-block shared payload capacity after the route's registration. */
inline size_t shared_capacity(const context &ctx, route r) {
    const size_t optin = ctx.properties.sharedMemPerBlockOptin;
    const size_t reserved = registration_bytes(ctx, r) + kStaticSmemReserve;
    return optin > reserved ? (optin - reserved) & ~(kAlignment - 1) : 0;
}

/* A shared payload must fit next to the registration; this is a benchmark
 * resource limit rather than a route capability. */
inline bool placement_available(context &ctx, route r, placement p) {
    const bool gap = p == placement::shared && shared_capacity(ctx, r) < kAlignment;
    if (!any_pe(ctx, gap)) {
        return true;
    }
    if (ctx.mype == 0) {
        printf(
            "%s: SKIP route=%s local=%s (insufficient dynamic shared memory for a "
            "shared-memory payload)\n",
            ctx.benchmark, info(r).label, placement_key(p));
    }
    return false;
}

/*
 * Asks nvshmemx_rma_capability, on every issuing PE, whether route R can run
 * operation Op at scope S with the row's local placement, block size, and registration
 * for a granule-sized payload. NOT_SUPPORTED on any issuer skips the row;
 * any other non-success answer means the benchmark built invalid arguments
 * and fails it.
 */
template <route R, placement P, rma_operation Op, Scope S = kBlock>
outcome rma_available(context &ctx, const std::string &subjob, unsigned char *remote,
                      unsigned char *local, int threads, bool issuer) {
    const int registration = static_cast<int>(registration_bytes(ctx, R));
    const size_t bytes = route_traits<R>::granule;
    int status = NVSHMEMX_SUCCESS;
    if (issuer) {
        const size_t smem = registration + (P == placement::shared ? bytes : 0);
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((rma_capability_kernel<R, P, Op, S>), smem);
        CUDA_CHECK(cudaMemset(ctx.statuses, kStatusFill, sizeof(int)));
        rma_capability_kernel<R, P, Op, S>
            <<<1, threads, smem>>>(remote, local, bytes, registration, ctx.peer, ctx.statuses);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        status = read_status(ctx.statuses, 1);
    }
    const bool unsupported = status == NVSHMEMX_ERROR_NOT_SUPPORTED;
    const bool invalid = status != NVSHMEMX_SUCCESS && !unsupported;
    if (invalid) {
        fprintf(stderr, "PE %d: %s: FAIL %s capability status=%d\n", ctx.mype, ctx.benchmark,
                subjob.c_str(), status);
    }
    if (any_pe(ctx, invalid)) {
        return outcome::failed;
    }
    if (any_pe(ctx, unsupported)) {
        if (unsupported) {
            printf("%s: SKIP %s on PE %d (route unavailable: %s)\n", ctx.benchmark, subjob.c_str(),
                   ctx.mype, info(R).requirement);
        }
        return outcome::skipped;
    }
    return outcome::success;
}

template <placement P>
using placement_constant = std::integral_constant<placement, P>;

/* Calls fn(placement_constant<P>{}) for each local placement of R that fits
 * and returns the summed failure count. */
template <route R, typename Fn>
int for_each_placement(context &ctx, Fn &&fn) {
    int failures = 0;
    if (placement_available(ctx, R, placement::global)) {
        failures += fn(placement_constant<placement::global>{});
    }
    if constexpr (route_traits<R>::shared_local) {
        if (placement_available(ctx, R, placement::shared)) {
            failures += fn(placement_constant<placement::shared>{});
        }
    }
    return failures;
}

inline int failed(outcome o) { return o == outcome::failed ? 1 : 0; }

/* Whether `-s/--scope` (default: all scopes) selects scope s. */
inline bool scope_selected(Scope s) {
    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            return s == Scope::thread;
        case NVSHMEM_WARP:
            return s == Scope::warp;
        case NVSHMEM_BLOCK:
            return s == Scope::block;
        default:
            return true;
    }
}

template <Scope S>
using scope_constant = std::integral_constant<Scope, S>;

/* Calls fn(scope_constant<S>{}) for each scope the user selected and returns
 * the summed failure count. */
template <typename Fn>
int for_each_scope(Fn &&fn) {
    int failures = 0;
    if (scope_selected(Scope::thread)) {
        failures += fn(scope_constant<Scope::thread>{});
    }
    if (scope_selected(Scope::warp)) {
        failures += fn(scope_constant<Scope::warp>{});
    }
    if (scope_selected(Scope::block)) {
        failures += fn(scope_constant<Scope::block>{});
    }
    return failures;
}

/* Runs Bench::run<R> for every route the benchmark measures and the user
 * selected; rows the route cannot serve report a SKIP. Returns the number of
 * failed rows. */
template <route R, typename Bench>
int run_route(context &ctx, Bench &bench) {
    if constexpr ((Bench::kRoutes & route_bit(R)) != 0) {
        if (ctx.selected(R)) {
            return bench.template run<R>(ctx);
        }
    }
    return 0;
}

template <typename Bench>
int run_selected_routes(context &ctx, Bench &bench) {
    int failures = 0;
    failures += run_route<route::stable>(ctx, bench);
    failures += run_route<route::auto_auto>(ctx, bench);
    failures += run_route<route::pointer_sm>(ctx, bench);
    failures += run_route<route::pointer_tma>(ctx, bench);
    failures += run_route<route::cft_tma>(ctx, bench);
    failures += run_route<route::auto_network>(ctx, bench);
    return failures;
}

struct sweep_spec {
    std::string subjob;
    const char *output_var;
    const char *units;
    char plus_minus;
    size_t granule;
    size_t max_blocks;
    size_t per_block_limit;
};

inline sweep_spec make_spec(route r, std::string variant, placement p, const char *output_var,
                            const char *units, char plus_minus, size_t granule, size_t max_blocks,
                            const context &ctx) {
    return {std::string(info(r).key) + "_" + variant + "_" + placement_key(p),
            output_var,
            units,
            plus_minus,
            granule,
            max_blocks,
            p == placement::shared ? shared_capacity(ctx, r) : SIZE_MAX};
}

/* One timed launch. Launch(bytes, grid, iterations) enqueues the kernel(s) on
 * the default stream and returns the number of status slots written, or 0
 * when this PE does not issue in this row. */
template <typename Launch>
int timed_launch(context &ctx, Launch &launch, size_t bytes, int grid, size_t iterations,
                 float *milliseconds) {
    CUDA_CHECK(
        cudaMemset(ctx.statuses, kStatusFill, static_cast<size_t>(ctx.max_blocks) * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_CHECK(cudaEventRecord(ctx.start));
    const int slots = launch(bytes, grid, iterations);
    CUDA_CHECK(cudaEventRecord(ctx.stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(ctx.stop));
    CUDA_CHECK(cudaEventElapsedTime(milliseconds, ctx.start, ctx.stop));
    return slots > 0 ? read_status(ctx.statuses, slots) : NVSHMEMX_SUCCESS;
}

/* Sweeps min_size..max_size, measuring warmup plus `repetitions` timed runs
 * per size and printing one standard perftest table on PE 0. Metric(bytes,
 * milliseconds) converts PE 0's timing into the reported value. Any nonzero
 * route status on any PE stops the sweep and fails the row. */
template <typename Launch, typename Metric>
outcome run_sweep(context &ctx, const sweep_spec &spec, Launch launch, Metric metric) {
    std::vector<uint64_t> sizes;
    std::vector<double> means;
    std::vector<perf_stats_t> stats;
    bool failed = false;
    bool unqualified = false;

    for (size_t bytes = min_size; bytes <= max_size; bytes *= step_factor) {
        const size_t units = bytes / spec.granule;
        if (bytes % spec.granule != 0 || units == 0) {
            unqualified = true;
        } else {
            const int grid = static_cast<int>(units < spec.max_blocks ? units : spec.max_blocks);
            const size_t per_block = (units + grid - 1) / grid * spec.granule;
            if (per_block > spec.per_block_limit) {
                break;
            }

            float milliseconds = 0.0f;
            const size_t warmups = warmup_iters > 0 ? warmup_iters : 1;
            int status = timed_launch(ctx, launch, bytes, grid, warmups, &milliseconds);
            perf_stats_t entry{};
            for (size_t rep = 0; rep < repetitions && !any_pe(ctx, status != NVSHMEMX_SUCCESS);
                 rep++) {
                status = timed_launch(ctx, launch, bytes, grid, iters, &milliseconds);
                perf_stats_add(entry, metric(bytes, milliseconds));
            }
            if (any_pe(ctx, status != NVSHMEMX_SUCCESS)) {
                if (status != NVSHMEMX_SUCCESS) {
                    fprintf(stderr, "PE %d: %s: FAIL %s size=%zu status=%d\n", ctx.mype,
                            ctx.benchmark, spec.subjob.c_str(), bytes, status);
                }
                failed = true;
                break;
            }
            sizes.push_back(bytes);
            means.push_back(entry.mean);
            stats.push_back(entry);
        }
        if (bytes > max_size / step_factor) {
            break;
        }
    }

    if (ctx.mype == 0) {
        if (unqualified) {
            printf("%s: %s measures only %zu-byte multiples\n", ctx.benchmark, spec.subjob.c_str(),
                   spec.granule);
        }
        if (!sizes.empty()) {
            print_basic_table(ctx.benchmark, spec.subjob.c_str(), spec.output_var, spec.units,
                              spec.plus_minus, sizes.data(), means.data(),
                              static_cast<int>(sizes.size()), stats.data());
        } else if (!failed) {
            printf("%s: SKIP %s (no qualified sizes in [%zu, %zu])\n", ctx.benchmark,
                   spec.subjob.c_str(), min_size, max_size);
        }
    }
    return failed ? outcome::failed : outcome::success;
}

/* Symmetric remote target and local global operand shared by the benchmarks. */
struct payload_buffers {
    static constexpr unsigned kRoutes = kAllRoutes;
    static constexpr int kPayloadFill = 0xa5;
    unsigned char *remote = nullptr;
    unsigned char *local = nullptr;

    int max_blocks() const { return 1; }

    void setup(context &ctx) {
        remote = static_cast<unsigned char *>(nvshmem_align(kAlignment, max_size));
        local = static_cast<unsigned char *>(nvshmem_align(kAlignment, max_size));
        if (remote == nullptr || local == nullptr) {
            fprintf(stderr, "PE %d: %s: payload allocation failed\n", ctx.mype, ctx.benchmark);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMemset(remote, 0, max_size));
        CUDA_CHECK(cudaMemset(local, kPayloadFill, max_size));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void teardown() {
        nvshmem_free(remote);
        nvshmem_free(local);
    }
};

/* Common main(): Bench provides max_blocks(), setup(ctx), teardown(), and
 * run<R>(ctx). Returns nonzero when any route row failed. */
template <typename Bench>
int run_benchmark(int argc, char **argv, const char *name, Bench &bench) {
    read_args(argc, argv);
    unsigned selection = 0;
    if (parse_route_selection(&selection)) {
        default_tma_policy(selection);
    }
    init_wrapper(&argc, &argv);

    context ctx{};
    int exit_code = 0;
    if (setup_context(ctx, name, bench.max_blocks(), &exit_code)) {
        bench.setup(ctx);
        nvshmem_barrier_all();
        exit_code = run_selected_routes(ctx, bench) != 0 ? 1 : 0;
        nvshmem_barrier_all();
        bench.teardown();
        teardown_context(ctx);
    }
    finalize_wrapper();
    return exit_code;
}

constexpr double kMillisecondsPerSecond = 1.0e3;
constexpr double kBytesPerGigabyte = 1.0e9;
constexpr double kMicrosecondsPerMillisecond = 1.0e3;

inline double bandwidth_gbps(size_t bytes, size_t iterations, float milliseconds) {
    return milliseconds > 0.0f ? static_cast<double>(bytes) * static_cast<double>(iterations) /
                                     (static_cast<double>(milliseconds) / kMillisecondsPerSecond) /
                                     kBytesPerGigabyte
                               : 0.0;
}

inline double latency_us(size_t iterations, float milliseconds) {
    return static_cast<double>(milliseconds) * kMicrosecondsPerMillisecond /
           static_cast<double>(iterations);
}

}  // namespace transport_specific

#endif  // TRANSPORT_SPECIFIC_PERFTEST_CUH
