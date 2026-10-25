/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Transport-policy route NBI source-reuse benchmark. Each iteration rewrites
 * the local source, issues a put_nbi, and then waits at a route flush (source
 * reuse) or quiet (remote completion) before the next rewrite. A final quiet
 * retires the last write before shared memory is released.
 *
 * The put_nbi_quiet_plus_auto rows measure the cost of route-specific
 * ordering when a kernel uses more than one route: each iteration moves half
 * of the payload through the measured route and half through Auto / Auto
 * (from a global source), then issues a quiet for each route. Compare them
 * with the single-route put_nbi_quiet rows of the same size. A row the route
 * cannot serve, as nvshmemx_rma_capability reports, is a SKIP. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_perftest.cuh"

namespace ts = transport_specific;
using ts::completion;
using ts::placement;
using ts::rma_nbi;
using ts::rma_operation;
using ts::route;

namespace {

constexpr const char *kBenchmark = "shmem_transport_specific_flush_bench";
constexpr int kThreads = 128;
/* The plus_auto rows split each payload into two halves. */
constexpr size_t kPlusAutoParts = 2;

template <route R, placement P, completion C, bool PlusAuto>
__global__ void source_reuse_batch(unsigned char *remote, unsigned char *local_global, size_t bytes,
                                   int registration, int peer, size_t iterations, int *statuses) {
    using traits = ts::route_traits<R>;
    using A = typename traits::addressing;
    using Pol = typename traits::policy;
    using AutoA = ts::route_traits<route::auto_auto>::addressing;
    using AutoP = ts::route_traits<route::auto_auto>::policy;

    unsigned char *shared = ts::acquire_smem<R>(registration);
    unsigned char *source = P == placement::shared ? shared : local_global;
    /* With PlusAuto the measured route moves [0, half) and Auto / Auto moves
     * [half, bytes) from the global source. */
    const size_t half = PlusAuto ? bytes / kPlusAutoParts : bytes;

    int result = NVSHMEMX_SUCCESS;
    for (size_t i = 0; i < iterations; i++) {
        for (size_t offset = threadIdx.x; offset < bytes; offset += blockDim.x) {
            (offset < half ? source : local_global)[offset] =
                static_cast<unsigned char>(i + offset);
        }
        __syncthreads();
        result =
            ts::rma_call<rma_operation::put, rma_nbi::yes>::template invoke<A, Pol,
                                                                            ts::local_space<R, P>>(
                remote, source, half, peer);
        if constexpr (PlusAuto) {
            if (result == NVSHMEMX_SUCCESS) {
                result = ts::rma_call<rma_operation::put, rma_nbi::yes>::template invoke<
                    AutoA, AutoP, nvshmemx::memory_space::source::generic>(
                    remote + half, local_global + half, bytes - half, peer);
            }
        }
        if (result == NVSHMEMX_SUCCESS) {
            result = ts::completion_call<C>::template invoke<A, Pol>();
        }
        if constexpr (PlusAuto) {
            if (result == NVSHMEMX_SUCCESS) {
                result = ts::completion_call<C>::template invoke<AutoA, AutoP>();
            }
        }
        if (ts::block_failed(result)) {
            break;
        }
    }
    if (!ts::block_failed(result)) {
        result = ts::completion_call<completion::quiet>::template invoke<A, Pol>();
    }
    if constexpr (PlusAuto) {
        if (!ts::block_failed(result)) {
            result = ts::completion_call<completion::quiet>::template invoke<AutoA, AutoP>();
        }
    }
    ts::release_smem<R>();
    ts::publish_block_status(result, statuses);
}

struct flush_bench : ts::payload_buffers {
    template <route R, placement P, completion C, bool PlusAuto = false>
    ts::outcome run_variant(ts::context &ctx) {
        /* Each half must stay a multiple of the measured route's granule. */
        const ts::sweep_spec spec = ts::make_spec(
            R, std::string("put_nbi_") + ts::completion_key(C) + (PlusAuto ? "_plus_auto" : ""), P,
            "BW", "GB/sec", '+', ts::route_traits<R>::granule * (PlusAuto ? kPlusAutoParts : 1), 1,
            ctx);
        const int registration = static_cast<int>(ts::registration_bytes(ctx, R));
        const ts::outcome available = ts::rma_available<R, P, rma_operation::put>(
            ctx, spec.subjob, remote, local, kThreads, ctx.mype == 0);
        if (available != ts::outcome::success) {
            return available;
        }

        auto launch = [&](size_t bytes, int, size_t iterations) {
            if (ctx.mype != 0) {
                return 0;
            }
            const size_t smem = registration + (P == placement::shared ? bytes : 0);
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((source_reuse_batch<R, P, C, PlusAuto>), smem);
            source_reuse_batch<R, P, C, PlusAuto><<<1, kThreads, smem>>>(
                remote, local, bytes, registration, ctx.peer, iterations, ctx.statuses);
            return 1;
        };
        auto metric = [](size_t bytes, float milliseconds) {
            return ts::bandwidth_gbps(bytes, iters, milliseconds);
        };
        return ts::run_sweep(ctx, spec, launch, metric);
    }

    template <route R>
    int run(ts::context &ctx) {
        return ts::for_each_placement<R>(ctx, [&](auto placement_tag) {
            constexpr placement P = decltype(placement_tag)::value;
            int failures = ts::failed(run_variant<R, P, completion::quiet>(ctx));
            failures += ts::failed(run_variant<R, P, completion::flush>(ctx));
            if constexpr (R != route::auto_auto && R != route::stable) {
                failures += ts::failed(run_variant<R, P, completion::quiet, true>(ctx));
            }
            return failures;
        });
    }
};

}  // namespace

int main(int argc, char **argv) {
    flush_bench bench;
    return ts::run_benchmark(argc, argv, kBenchmark, bench);
}
