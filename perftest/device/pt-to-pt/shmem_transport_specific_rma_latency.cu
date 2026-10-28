/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_perftest.cuh"

namespace ts = transport_specific;
using ts::completion;
using ts::placement;
using ts::rma_nbi;
using ts::rma_operation;
using ts::route;

namespace {

constexpr const char *kBenchmark = "shmem_transport_specific_rma_latency";
constexpr int kThreads = 128;

template <route R, placement P, rma_operation Op, rma_nbi Nbi, ts::Scope S>
__global__ void latency_batch(unsigned char *remote, unsigned char *local_global, size_t bytes,
                              int registration, int peer, size_t iterations, int *statuses) {
    using traits = ts::route_traits<R>;
    using A = typename traits::addressing;
    using Pol = typename traits::policy;

    unsigned char *shared = ts::acquire_smem<R>(registration);
    unsigned char *local = P == placement::shared ? shared : local_global;

    int result = NVSHMEMX_SUCCESS;
    for (size_t i = 0; i < iterations; i++) {
        result = ts::rma_call<Op, Nbi>::template invoke<A, Pol, ts::local_space<R, P>, S>(
            remote, local, bytes, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = ts::completion_call<completion::quiet>::template invoke<A, Pol, S>();
        }
        if (ts::block_failed(result)) {
            break;
        }
    }
    ts::release_smem<R>();
    ts::publish_block_status(result, statuses);
}

struct rma_latency : ts::payload_buffers {
    template <route R, placement P, rma_operation Op, rma_nbi Nbi, ts::Scope S>
    ts::outcome run_variant(ts::context &ctx) {
        constexpr int kScopeThreads = ts::scope_threads(S, kThreads);
        const std::string variant = std::string(Op == rma_operation::put ? "put" : "get") + "_" +
                                    (Nbi == rma_nbi::yes ? "nbi" : "blocking") + "_quiet_" +
                                    ts::scope_key(S);
        const ts::sweep_spec spec = ts::make_spec(R, variant, P, "latency", "us", '-',
                                                  ts::route_traits<R>::granule, 1, ctx);
        const int registration = static_cast<int>(ts::registration_bytes(ctx, R));
        const ts::outcome available = ts::rma_available<R, P, Op, S>(
            ctx, spec.subjob, remote, local, kScopeThreads, ctx.mype == 0);
        if (available != ts::outcome::success) {
            return available;
        }

        auto launch = [&](size_t bytes, int, size_t iterations) {
            if (ctx.mype != 0) {
                return 0;
            }
            const size_t smem = registration + (P == placement::shared ? bytes : 0);
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((latency_batch<R, P, Op, Nbi, S>), smem);
            latency_batch<R, P, Op, Nbi, S><<<1, kScopeThreads, smem>>>(
                remote, local, bytes, registration, ctx.peer, iterations, ctx.statuses);
            return 1;
        };
        auto metric = [](size_t, float milliseconds) {
            return ts::latency_us(iters, milliseconds);
        };
        return ts::run_sweep(ctx, spec, launch, metric);
    }

    template <route R, placement P, rma_operation Op, rma_nbi Nbi, ts::Scope S>
    int run_legal(ts::context &ctx) {
        if constexpr (ts::scope_legal<R, S, P, Op>) {
            return ts::failed(run_variant<R, P, Op, Nbi, S>(ctx));
        } else {
            return 0;
        }
    }

    template <route R>
    int run(ts::context &ctx) {
        return ts::for_each_placement<R>(ctx, [&](auto placement_tag) {
            constexpr placement P = decltype(placement_tag)::value;
            return ts::for_each_scope([&](auto scope_tag) {
                constexpr ts::Scope S = decltype(scope_tag)::value;
                return run_legal<R, P, rma_operation::put, rma_nbi::no, S>(ctx) +
                       run_legal<R, P, rma_operation::put, rma_nbi::yes, S>(ctx) +
                       run_legal<R, P, rma_operation::get, rma_nbi::no, S>(ctx) +
                       run_legal<R, P, rma_operation::get, rma_nbi::yes, S>(ctx);
            });
        });
    }
};

}  // namespace

int main(int argc, char **argv) {
    rma_latency bench;
    return ts::run_benchmark(argc, argv, kBenchmark, bench);
}
