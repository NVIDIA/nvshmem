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

constexpr const char *kBenchmark = "shmem_transport_specific_rma_bw";

template <route R, placement P, rma_operation Op, rma_nbi Nbi>
__global__ void rma_batch(unsigned char *remote, unsigned char *local_global, size_t bytes,
                          int registration, int peer, size_t iterations, int *statuses) {
    using traits = ts::route_traits<R>;
    using A = typename traits::addressing;
    using Pol = typename traits::policy;
    constexpr completion kCompletion =
        (Op == rma_operation::put || Nbi == rma_nbi::yes) ? completion::quiet : completion::none;

    unsigned char *shared = ts::acquire_smem<R>(registration);
    size_t offset = 0;
    size_t length = 0;
    ts::block_slice(bytes, traits::granule, &offset, &length);
    unsigned char *local = P == placement::shared ? shared : local_global + offset;

    int result = NVSHMEMX_SUCCESS;
    for (size_t i = 0; i < iterations; i++) {
        result = ts::rma_call<Op, Nbi>::template invoke<A, Pol, ts::local_space<R, P>>(
            remote + offset, local, length, peer);
        if (ts::block_failed(result)) {
            break;
        }
    }
    if (!ts::block_failed(result)) {
        result = ts::completion_call<kCompletion>::template invoke<A, Pol>();
    }
    ts::release_smem<R>();
    ts::publish_block_status(result, statuses);
}

struct rma_bw : ts::payload_buffers {
    int max_blocks() const { return static_cast<int>(num_blocks); }

    template <route R, placement P, rma_operation Op, rma_nbi Nbi>
    ts::outcome run_variant(ts::context &ctx) {
        const std::string variant = std::string(Op == rma_operation::put ? "put" : "get") + "_" +
                                    (Nbi == rma_nbi::yes ? "nbi" : "blocking") + "_" +
                                    (bidirectional ? "bidir" : "unidir");
        const ts::sweep_spec spec =
            ts::make_spec(R, variant, P, "BW", "GB/sec", '+', ts::route_traits<R>::granule,
                          static_cast<size_t>(ctx.max_blocks), ctx);
        const int registration = static_cast<int>(ts::registration_bytes(ctx, R));
        const bool issuer = bidirectional || ctx.mype == 0;
        const int threads = static_cast<int>(threads_per_block);
        const ts::outcome available =
            ts::rma_available<R, P, Op>(ctx, spec.subjob, remote, local, threads, issuer);
        if (available != ts::outcome::success) {
            return available;
        }

        auto launch = [&](size_t bytes, int grid, size_t iterations) {
            if (!issuer) {
                return 0;
            }
            const size_t units = bytes / spec.granule;
            const size_t payload =
                P == placement::shared ? (units + grid - 1) / grid * spec.granule : 0;
            const size_t smem = registration + payload;
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((rma_batch<R, P, Op, Nbi>), smem);
            rma_batch<R, P, Op, Nbi><<<grid, threads, smem>>>(remote, local, bytes, registration,
                                                              ctx.peer, iterations, ctx.statuses);
            return grid;
        };
        /* Bidirectional reports both PEs' bandwidth combined, as shmem_put_bw does. */
        auto metric = [&](size_t bytes, float milliseconds) {
            const double bw = ts::bandwidth_gbps(bytes, iters, milliseconds);
            return bidirectional ? ts::world_sum(ctx, bw) : bw;
        };
        return ts::run_sweep(ctx, spec, launch, metric);
    }

    template <route R>
    int run(ts::context &ctx) {
        return ts::for_each_placement<R>(ctx, [&](auto placement_tag) {
            constexpr placement P = decltype(placement_tag)::value;
            return ts::failed(run_variant<R, P, rma_operation::put, rma_nbi::no>(ctx)) +
                   ts::failed(run_variant<R, P, rma_operation::put, rma_nbi::yes>(ctx)) +
                   ts::failed(run_variant<R, P, rma_operation::get, rma_nbi::no>(ctx)) +
                   ts::failed(run_variant<R, P, rma_operation::get, rma_nbi::yes>(ctx));
        });
    }
};

}  // namespace

int main(int argc, char **argv) {
    rma_bw bench;
    return ts::run_benchmark(argc, argv, kBenchmark, bench);
}
