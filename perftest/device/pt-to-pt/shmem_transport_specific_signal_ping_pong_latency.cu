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

constexpr const char *kBenchmark = "shmem_transport_specific_signal_ping_pong_latency";
constexpr int kThreads = 128;
constexpr uint64_t kAddIncrement = 1;
constexpr uint64_t kProbeValue = 1;

template <int SigOp>
__device__ uint64_t signal_value(uint64_t iteration) {
    return SigOp == NVSHMEM_SIGNAL_SET ? iteration : kAddIncrement;
}

__device__ void block_wait_until(uint64_t *flag, uint64_t value) {
    if (threadIdx.x == 0) {
        nvshmem_signal_wait_until(flag, NVSHMEM_CMP_EQ, value);
    }
    __syncthreads();
}

template <route R, placement P, rma_nbi Nbi, int SigOp, ts::Scope S>
__device__ int send(unsigned char *remote, unsigned char *local, size_t bytes, uint64_t *signal,
                    uint64_t value, int peer) {
    using traits = ts::route_traits<R>;
    using A = typename traits::addressing;
    using Pol = typename traits::policy;
    constexpr completion kCompletion = Nbi == rma_nbi::yes ? completion::quiet : completion::none;

    const int result = ts::put_signal_call<Nbi>::template invoke<A, Pol, ts::local_space<R, P>, S>(
        remote, local, bytes, signal, value, SigOp, peer);
    if (result != NVSHMEMX_SUCCESS) {
        return result;
    }
    return ts::completion_call<kCompletion>::template invoke<A, Pol, S>();
}

template <route R, int SigOp, ts::Scope S>
__device__ int acknowledge(uint64_t *ack, uint64_t value, int peer) {
    using traits = ts::route_traits<R>;
    return ts::signal_op_call<typename traits::addressing, typename traits::policy, S>(ack, value,
                                                                                       SigOp, peer);
}

template <route R, placement P, rma_nbi Nbi, int SigOp, ts::Scope S>
__global__ void signal_probe(unsigned char *remote, unsigned char *local_global, size_t bytes,
                             uint64_t *signal, uint64_t *ack, int registration, int peer,
                             int *statuses) {
    unsigned char *shared = ts::acquire_smem<R>(registration);
    unsigned char *local = P == placement::shared ? shared : local_global;
    int result = send<R, P, Nbi, SigOp, S>(remote, local, bytes, signal, kProbeValue, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = acknowledge<R, SigOp, S>(ack, kProbeValue, peer);
    }
    ts::release_smem<R>();
    ts::publish_block_status(result, statuses);
}

template <route R, placement P, rma_nbi Nbi, int SigOp, ts::Scope S>
__global__ void signal_ping_pong(unsigned char *remote, unsigned char *local_global, size_t bytes,
                                 uint64_t *signal, uint64_t *ack, int registration, int mype,
                                 size_t iterations, int *statuses) {
    unsigned char *shared = ts::acquire_smem<R>(registration);
    unsigned char *local = P == placement::shared ? shared : local_global;
    const int peer = !mype;

    int result = NVSHMEMX_SUCCESS;
    for (uint64_t i = 1; i <= iterations; i++) {
        const uint64_t value = signal_value<SigOp>(i);
        if (mype == 0) {
            result = send<R, P, Nbi, SigOp, S>(remote, local, bytes, signal, value, peer);
            if (ts::block_failed(result)) {
                break;
            }
            block_wait_until(ack, i);
        } else {
            block_wait_until(signal, i);
            result = acknowledge<R, SigOp, S>(ack, value, peer);
            if (ts::block_failed(result)) {
                break;
            }
        }
    }
    /* The peer no longer writes this flag, so rearm it for the next launch. */
    if (threadIdx.x == 0 && result == NVSHMEMX_SUCCESS) {
        *(mype == 0 ? ack : signal) = 0;
    }
    ts::release_smem<R>();
    ts::publish_block_status(result, statuses);
}

struct signal_ping_pong_bench : ts::payload_buffers {
    uint64_t *signal = nullptr;
    uint64_t *ack = nullptr;

    void setup(ts::context &ctx) {
        ts::payload_buffers::setup(ctx);
        signal = static_cast<uint64_t *>(nvshmem_align(ts::kAlignment, ts::kAlignment));
        ack = static_cast<uint64_t *>(nvshmem_align(ts::kAlignment, ts::kAlignment));
        if (signal == nullptr || ack == nullptr) {
            fprintf(stderr, "PE %d: %s: signal allocation failed\n", ctx.mype, ctx.benchmark);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        reset_flags();
    }

    void teardown() {
        nvshmem_free(signal);
        nvshmem_free(ack);
        ts::payload_buffers::teardown();
    }

    void reset_flags() {
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(*signal)));
        CUDA_CHECK(cudaMemset(ack, 0, sizeof(*ack)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
    }

    template <route R, placement P, rma_nbi Nbi, int SigOp, ts::Scope S>
    ts::outcome probe_signal(ts::context &ctx, const std::string &subjob, int registration) {
        const size_t probe_bytes = ts::route_traits<R>::granule;
        const size_t probe_smem = registration + (P == placement::shared ? probe_bytes : 0);
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((signal_probe<R, P, Nbi, SigOp, S>), probe_smem);

        reset_flags();
        CUDA_CHECK(cudaMemset(ctx.statuses, ts::kStatusFill, sizeof(int)));
        signal_probe<R, P, Nbi, SigOp, S><<<1, ts::scope_threads(S, kThreads), probe_smem>>>(
            remote, local, probe_bytes, signal, ack, registration, ctx.peer, ctx.statuses);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const int status = ts::read_status(ctx.statuses, 1);
        nvshmem_barrier_all();
        reset_flags();

        const bool unsupported = status == NVSHMEMX_ERROR_NOT_SUPPORTED;
        const bool invalid = status != NVSHMEMX_SUCCESS && !unsupported;
        if (invalid) {
            fprintf(stderr, "PE %d: %s: FAIL %s probe status=%d\n", ctx.mype, ctx.benchmark,
                    subjob.c_str(), status);
        }
        if (ts::any_pe(ctx, invalid)) {
            return ts::outcome::failed;
        }
        if (ts::any_pe(ctx, unsupported)) {
            if (unsupported) {
                printf("%s: SKIP %s on PE %d (the route's %s signal is unavailable)\n",
                       ctx.benchmark, subjob.c_str(), ctx.mype,
                       SigOp == NVSHMEM_SIGNAL_SET ? "SET" : "ADD");
            }
            return ts::outcome::skipped;
        }
        return ts::outcome::success;
    }

    template <route R, placement P, rma_nbi Nbi, int SigOp, ts::Scope S>
    ts::outcome run_variant(ts::context &ctx) {
        constexpr int kScopeThreads = ts::scope_threads(S, kThreads);
        const std::string variant =
            std::string("put_signal_") + (Nbi == rma_nbi::yes ? "nbi_quiet" : "blocking") + "_" +
            (SigOp == NVSHMEM_SIGNAL_SET ? "set" : "add") + "_signal_op_" + ts::scope_key(S);
        const ts::sweep_spec spec = ts::make_spec(R, variant, P, "latency", "us", '-',
                                                  ts::route_traits<R>::granule, 1, ctx);
        const int registration = static_cast<int>(ts::registration_bytes(ctx, R));
        ts::outcome available = ts::rma_available<R, P, rma_operation::put, S>(
            ctx, spec.subjob, remote, local, kScopeThreads, ctx.mype == 0);
        if (available == ts::outcome::success) {
            available = probe_signal<R, P, Nbi, SigOp, S>(ctx, spec.subjob, registration);
        }
        if (available != ts::outcome::success) {
            return available;
        }

        auto launch = [&](size_t bytes, int, size_t iterations) {
            const size_t smem = registration + (P == placement::shared ? bytes : 0);
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((signal_ping_pong<R, P, Nbi, SigOp, S>), smem);
            signal_ping_pong<R, P, Nbi, SigOp, S>
                <<<1, kScopeThreads, smem>>>(remote, local, bytes, signal, ack, registration,
                                             ctx.mype, iterations, ctx.statuses);
            return 1;
        };
        auto metric = [](size_t, float milliseconds) {
            return ts::latency_us(iters, milliseconds);
        };
        const ts::outcome result = ts::run_sweep(ctx, spec, launch, metric);
        if (result == ts::outcome::failed) {
            reset_flags();
        }
        return result;
    }

    template <route R, placement P, int SigOp, ts::Scope S>
    int run_signal_op(ts::context &ctx) {
        if constexpr (ts::scope_legal<R, S, P, rma_operation::put>) {
            return ts::failed(run_variant<R, P, rma_nbi::no, SigOp, S>(ctx)) +
                   ts::failed(run_variant<R, P, rma_nbi::yes, SigOp, S>(ctx));
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
                return run_signal_op<R, P, NVSHMEM_SIGNAL_SET, S>(ctx) +
                       run_signal_op<R, P, NVSHMEM_SIGNAL_ADD, S>(ctx);
            });
        });
    }
};

}  // namespace

int main(int argc, char **argv) {
    signal_ping_pong_bench bench;
    return ts::run_benchmark(argc, argv, kBenchmark, bench);
}
