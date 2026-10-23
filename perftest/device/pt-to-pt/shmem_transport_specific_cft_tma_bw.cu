/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* CFT / TMA completion-mode sweep: block-scoped put, get, and put_signal (SET,
 * and ADD where the fabric atomic capability exists), each blocking or NBI
 * followed by one CFT quiet, flush, or fence per operation. The CFT get and
 * put_signal paths complete before they return, so NBI rows are labelled by
 * the completion they use rather than claiming deferred completion.
 * nvshmemx_rma_capability decides whether the route can serve each row, and
 * an unavailable route or signal is a SKIP. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_perftest.cuh"

namespace ts = transport_specific;
using ts::completion;
using ts::placement;
using ts::rma_nbi;
using ts::rma_operation;
using ts::route;

namespace {

constexpr const char *kBenchmark = "shmem_transport_specific_cft_tma_bw";
constexpr int kThreads = 128;
constexpr route kRoute = route::cft_tma;
/* Adding zero probes the fabric atomic capability without changing the word. */
constexpr uint64_t kAddProbeValue = 0;
constexpr uint64_t kAddIncrement = 1;

using cft_traits = ts::route_traits<kRoute>;
using CftAddressing = cft_traits::addressing;
using CftPolicy = cft_traits::policy;

enum class cft_operation { put, get, put_signal_set, put_signal_add };

template <cft_operation Op, rma_nbi Nbi>
struct cft_call;

template <rma_nbi Nbi>
struct cft_call<cft_operation::put, Nbi> {
    template <ts::Memory::source Local>
    __device__ static int invoke(unsigned char *remote, unsigned char *local, size_t bytes,
                                 uint64_t *, uint64_t, int pe) {
        return ts::rma_call<rma_operation::put, Nbi>::template invoke<CftAddressing, CftPolicy,
                                                                      Local>(remote, local, bytes,
                                                                             pe);
    }
};

template <rma_nbi Nbi>
struct cft_call<cft_operation::get, Nbi> {
    template <ts::Memory::source Local>
    __device__ static int invoke(unsigned char *remote, unsigned char *local, size_t bytes,
                                 uint64_t *, uint64_t, int pe) {
        return ts::rma_call<rma_operation::get, Nbi>::template invoke<CftAddressing, CftPolicy,
                                                                      Local>(remote, local, bytes,
                                                                             pe);
    }
};

template <rma_nbi Nbi>
struct cft_call<cft_operation::put_signal_set, Nbi> {
    template <ts::Memory::source Local>
    __device__ static int invoke(unsigned char *remote, unsigned char *local, size_t bytes,
                                 uint64_t *signal, uint64_t iteration, int pe) {
        return ts::put_signal_call<Nbi>::template invoke<CftAddressing, CftPolicy, Local>(
            remote, local, bytes, signal, iteration, NVSHMEM_SIGNAL_SET, pe);
    }
};

template <rma_nbi Nbi>
struct cft_call<cft_operation::put_signal_add, Nbi> {
    template <ts::Memory::source Local>
    __device__ static int invoke(unsigned char *remote, unsigned char *local, size_t bytes,
                                 uint64_t *signal, uint64_t, int pe) {
        return ts::put_signal_call<Nbi>::template invoke<CftAddressing, CftPolicy, Local>(
            remote, local, bytes, signal, kAddIncrement, NVSHMEM_SIGNAL_ADD, pe);
    }
};

template <placement P, cft_operation Op, rma_nbi Nbi, completion C>
__global__ void cft_batch(unsigned char *remote, unsigned char *local_global, size_t bytes,
                          uint64_t *signal, int registration, int peer, size_t iterations,
                          int *statuses) {
    unsigned char *shared = ts::acquire_smem<kRoute>(registration);
    unsigned char *local = P == placement::shared ? shared : local_global;

    int result = NVSHMEMX_SUCCESS;
    for (size_t i = 0; i < iterations; i++) {
        result = cft_call<Op, Nbi>::template invoke<ts::local_space<kRoute, P>>(
            remote, local, bytes, signal, i + 1, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = ts::completion_call<C>::template invoke<CftAddressing, CftPolicy>();
        }
        if (ts::block_failed(result)) {
            break;
        }
    }
    ts::release_smem<kRoute>();
    ts::publish_block_status(result, statuses);
}

__global__ void add_probe(uint64_t *signal, int registration, int peer, int *statuses) {
    ts::acquire_smem<kRoute>(registration);
    const int result = nvshmemx_signal_op<CftAddressing, CftPolicy, ts::kBlock>(
        signal, kAddProbeValue, NVSHMEM_SIGNAL_ADD, peer);
    ts::release_smem<kRoute>();
    ts::publish_block_status(result, statuses);
}

inline const char *operation_key(cft_operation op) {
    switch (op) {
        case cft_operation::put:
            return "put";
        case cft_operation::get:
            return "get";
        case cft_operation::put_signal_set:
            return "put_signal_set";
        default:
            return "put_signal_add";
    }
}

struct cft_tma_bw : ts::payload_buffers {
    static constexpr unsigned kRoutes = ts::route_bit(kRoute);
    uint64_t *signal = nullptr;
    bool add_supported = false;

    void setup(ts::context &ctx) {
        ts::payload_buffers::setup(ctx);
        /* CFT ADD updates a 16-byte block, so the signal word is 16-byte aligned. */
        signal = static_cast<uint64_t *>(nvshmem_align(ts::kAlignment, ts::kAlignment));
        if (signal == nullptr) {
            fprintf(stderr, "PE %d: %s: signal allocation failed\n", ctx.mype, ctx.benchmark);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMemset(signal, 0, sizeof(*signal)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void teardown() {
        nvshmem_free(signal);
        ts::payload_buffers::teardown();
    }

    /* Whether PE 0 can issue signal ADD; NOT_SUPPORTED skips the ADD rows. */
    ts::outcome probe_add(ts::context &ctx) {
        const int registration = static_cast<int>(ts::registration_bytes(ctx, kRoute));
        int status = NVSHMEMX_SUCCESS;
        if (ctx.mype == 0) {
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(add_probe, registration);
            CUDA_CHECK(cudaMemset(ctx.statuses, ts::kStatusFill, sizeof(int)));
            add_probe<<<1, kThreads, registration>>>(signal, registration, ctx.peer, ctx.statuses);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            status = ts::read_status(ctx.statuses, 1);
        }
        const bool unsupported = status == NVSHMEMX_ERROR_NOT_SUPPORTED;
        const bool invalid = status != NVSHMEMX_SUCCESS && !unsupported;
        if (invalid) {
            fprintf(stderr, "PE %d: %s: FAIL signal ADD probe status=%d\n", ctx.mype, ctx.benchmark,
                    status);
        }
        if (ts::any_pe(ctx, invalid)) {
            return ts::outcome::failed;
        }
        if (ts::any_pe(ctx, unsupported)) {
            if (ctx.mype == 0) {
                printf(
                    "%s: SKIP put_signal_add rows (signal ADD unavailable: needs the route and the "
                    "fabric atomic capability)\n",
                    ctx.benchmark);
            }
            return ts::outcome::skipped;
        }
        return ts::outcome::success;
    }

    template <placement P, cft_operation Op, rma_nbi Nbi, completion C>
    ts::outcome run_variant(ts::context &ctx) {
        const std::string variant = std::string(operation_key(Op)) + "_" +
                                    (Nbi == rma_nbi::yes ? "nbi_" : "blocking") +
                                    (Nbi == rma_nbi::yes ? ts::completion_key(C) : "");
        const ts::sweep_spec spec =
            ts::make_spec(kRoute, variant, P, "BW", "GB/sec", '+', cft_traits::granule, 1, ctx);
        const int registration = static_cast<int>(ts::registration_bytes(ctx, kRoute));
        constexpr rma_operation kRma =
            Op == cft_operation::get ? rma_operation::get : rma_operation::put;
        const ts::outcome available = ts::rma_available<kRoute, P, kRma>(
            ctx, spec.subjob, remote, local, kThreads, ctx.mype == 0);
        if (available != ts::outcome::success) {
            return available;
        }

        auto launch = [&](size_t bytes, int, size_t iterations) {
            if (ctx.mype != 0) {
                return 0;
            }
            const size_t smem = registration + (P == placement::shared ? bytes : 0);
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM((cft_batch<P, Op, Nbi, C>), smem);
            cft_batch<P, Op, Nbi, C><<<1, kThreads, smem>>>(
                remote, local, bytes, signal, registration, ctx.peer, iterations, ctx.statuses);
            return 1;
        };
        auto metric = [](size_t bytes, float milliseconds) {
            return ts::bandwidth_gbps(bytes, iters, milliseconds);
        };
        return ts::run_sweep(ctx, spec, launch, metric);
    }

    template <placement P, cft_operation Op>
    int run_operation(ts::context &ctx) {
        return ts::failed(run_variant<P, Op, rma_nbi::no, completion::none>(ctx)) +
               ts::failed(run_variant<P, Op, rma_nbi::yes, completion::quiet>(ctx)) +
               ts::failed(run_variant<P, Op, rma_nbi::yes, completion::flush>(ctx)) +
               ts::failed(run_variant<P, Op, rma_nbi::yes, completion::fence>(ctx));
    }

    template <route R>
    int run(ts::context &ctx) {
        int failures = 0;
        bool add_probed = false;
        failures += ts::for_each_placement<R>(ctx, [&](auto placement_tag) {
            constexpr placement P = decltype(placement_tag)::value;
            int row_failures = run_operation<P, cft_operation::put>(ctx) +
                               run_operation<P, cft_operation::get>(ctx) +
                               run_operation<P, cft_operation::put_signal_set>(ctx);
            if (!add_probed) {
                const ts::outcome add = probe_add(ctx);
                add_probed = true;
                add_supported = add == ts::outcome::success;
                row_failures += ts::failed(add);
            }
            if (add_supported) {
                row_failures += run_operation<P, cft_operation::put_signal_add>(ctx);
            }
            return row_failures;
        });
        return failures;
    }
};

}  // namespace

int main(int argc, char **argv) {
    cft_tma_bw bench;
    return ts::run_benchmark(argc, argv, kBenchmark, bench);
}
