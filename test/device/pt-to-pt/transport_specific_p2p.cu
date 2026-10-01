/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Functional coverage for the direct Pointer / SM route and the portable
 * Auto / Auto route. Auto / Auto always runs, so a topology without a direct
 * peer mapping still checks the portable route rather than skipping the
 * executable. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_common.cuh"

using namespace ts_test;

namespace {

constexpr size_t kBytes = 256;

/* Each case is rejected by preflight before any work is issued. */
enum class error_case {
    invalid_pe,         /* INVALID_VALUE */
    invalid_sig_op,     /* INVALID_VALUE */
    invalid_signal_op,  /* INVALID_VALUE */
    private_put_dest,   /* INVALID_VALUE: remote range outside the heap */
    private_get_source, /* INVALID_VALUE: remote range outside the heap */
    private_signal,     /* INVALID_VALUE: signal outside the heap */
    null_put_source,    /* INVALID_VALUE: null local operand with bytes */
    null_get_dest,      /* INVALID_VALUE: null local operand with bytes */
    count_overflow,     /* INVALID_VALUE: saturated byte count leaves the heap */
    unmapped_put,       /* NOT_SUPPORTED on Pointer / SM: peer has no direct mapping */
    unmapped_signal,    /* NOT_SUPPORTED on Pointer / SM: peer has no direct mapping */
};

struct buffers {
    unsigned char *remote; /* symmetric put destination */
    unsigned char *source; /* symmetric put source and peer get source */
    unsigned char *out;    /* symmetric get destination */
    unsigned char *private_buffer;
    uint64_t *signal;
    uint64_t *private_signal;
    /* symmetric targets of the portable-API reference transfers */
    unsigned char *ref_remote;
    unsigned char *ref_out;
    uint64_t *ref_signal;
};

template <typename A, typename P, LocalSpace Local, bool InShared, bool Nbi>
__global__ void put_kernel(unsigned char *remote, unsigned char *source, size_t bytes, int peer,
                           int *status) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *local = InShared ? smem : source;
    if (InShared) {
        copy_block(smem, source, bytes);
    }
    __syncthreads();

    int result;
    if constexpr (Nbi) {
        result =
            nvshmemx_put_nbi<uint8_t, A, P, kBlock, Local, kRemoteDest>(remote, local, bytes, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_flush<A, P, kBlock>();
        }
        if (result == NVSHMEMX_SUCCESS) {
            /* flush makes the source reusable; the peer must still see the original. */
            fill_block(local, bytes, kPoisonByte);
            __syncthreads();
            result = nvshmemx_quiet<A, P, kBlock>();
        }
    } else {
        result =
            nvshmemx_put<uint8_t, A, P, kBlock, Local, kRemoteDest>(remote, local, bytes, peer);
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

template <typename A, typename P, LocalSpace Local, bool InShared, bool Nbi>
__global__ void get_kernel(unsigned char *out, const unsigned char *remote_source, size_t bytes,
                           int peer, int *status) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *local = InShared ? smem : out;
    int result;
    if constexpr (Nbi) {
        result = nvshmemx_get_nbi<uint8_t, A, P, kBlock, kRemoteSource, local_dest(Local)>(
            local, remote_source, bytes, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_quiet<A, P, kBlock>();
        }
    } else {
        result = nvshmemx_get<uint8_t, A, P, kBlock, kRemoteSource, local_dest(Local)>(
            local, remote_source, bytes, peer);
    }
    __syncthreads();
    if (InShared && result == NVSHMEMX_SUCCESS) {
        copy_block(out, smem, bytes);
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

template <typename A, typename P, LocalSpace Local, bool Nbi>
__global__ void put_signal_kernel(unsigned char *remote, const unsigned char *source, size_t bytes,
                                  uint64_t *signal, uint64_t value, int sig_op, int peer,
                                  int *status) {
    int result;
    if constexpr (Nbi) {
        result = nvshmemx_put_signal_nbi<uint8_t, A, P, kBlock, Local, kRemoteDest>(
            remote, source, bytes, signal, value, sig_op, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_quiet<A, P, kBlock>();
        }
    } else {
        result = nvshmemx_put_signal<uint8_t, A, P, kBlock, Local, kRemoteDest>(
            remote, source, bytes, signal, value, sig_op, peer);
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

template <typename A, typename P>
__global__ void signal_op_kernel(uint64_t *signal, uint64_t value, int sig_op, int peer,
                                 int *status) {
    const int result = nvshmemx_signal_op<A, P, kBlock>(signal, value, sig_op, peer);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* fence orders the NBI payload before the following signal. */
template <typename A, typename P, LocalSpace Local>
__global__ void fence_kernel(unsigned char *remote, const unsigned char *source, size_t bytes,
                             uint64_t *signal, uint64_t value, int peer, int *status) {
    int result =
        nvshmemx_put_nbi<uint8_t, A, P, kBlock, Local, kRemoteDest>(remote, source, bytes, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_fence<A, P, kBlock>();
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_signal_op<A, P, kBlock>(signal, value, NVSHMEM_SIGNAL_SET, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, kBlock>();
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

template <typename T, typename A, typename P, LocalSpace Local>
__global__ void typed_kernel(unsigned char *remote, unsigned char *source, unsigned char *out,
                             size_t bytes, int peer, int *status) {
    const size_t nelems = bytes / sizeof(T);
    int result = nvshmemx_put<T, A, P, kBlock, Local, kRemoteDest>(
        reinterpret_cast<T *>(remote), reinterpret_cast<const T *>(source), nelems, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<T, A, P, kBlock, kRemoteSource, local_dest(Local)>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(source), nelems, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, kBlock>();
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

using typed_kernel_t = void (*)(unsigned char *, unsigned char *, unsigned char *, size_t, int,
                                int *);

/* A zero-length transfer may name a null local operand: it succeeds and
 * nothing is modified. */
template <typename A, typename P>
__global__ void zero_length_kernel(unsigned char *remote, const unsigned char *remote_source,
                                   int peer, int *status) {
    constexpr auto kLocalDest = local_dest(kGeneric);
    int result =
        nvshmemx_put<uint8_t, A, P, kBlock, kGeneric, kRemoteDest>(remote, nullptr, 0, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_put_nbi<uint8_t, A, P, kBlock, kGeneric, kRemoteDest>(remote, nullptr, 0,
                                                                                peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get<uint8_t, A, P, kBlock, kRemoteSource, kLocalDest>(
            nullptr, remote_source, 0, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<uint8_t, A, P, kBlock, kRemoteSource, kLocalDest>(
            nullptr, remote_source, 0, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_quiet<A, P, kBlock>();
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

template <typename A, typename P, LocalSpace Local>
__global__ void error_kernel(error_case which, buffers b, size_t bytes, int peer, int npes,
                             int *status) {
    int result = NVSHMEMX_SUCCESS;
    switch (which) {
        case error_case::invalid_pe:
            result = nvshmemx_put<uint8_t, A, P, kBlock, Local, kRemoteDest>(b.remote, b.source,
                                                                             bytes, npes);
            break;
        case error_case::invalid_sig_op:
            result = nvshmemx_put_signal<uint8_t, A, P, kBlock, Local, kRemoteDest>(
                b.remote, b.source, bytes, b.signal, 1, NVSHMEM_SIGNAL_ADD + 100, peer);
            break;
        case error_case::invalid_signal_op:
            result = nvshmemx_signal_op<A, P, kBlock>(b.signal, 1, NVSHMEM_SIGNAL_SET + 100, peer);
            break;
        case error_case::private_put_dest:
            result = nvshmemx_put<uint8_t, A, P, kBlock, Local, kRemoteDest>(b.private_buffer,
                                                                             b.source, bytes, peer);
            break;
        case error_case::private_get_source:
            result = nvshmemx_get<uint8_t, A, P, kBlock, kRemoteSource, local_dest(Local)>(
                b.out, b.private_buffer, bytes, peer);
            break;
        case error_case::private_signal:
            result = nvshmemx_put_signal_nbi<uint8_t, A, P, kBlock, Local, kRemoteDest>(
                b.remote, b.source, bytes, b.private_signal, 1, NVSHMEM_SIGNAL_SET, peer);
            break;
        case error_case::null_put_source:
            result = nvshmemx_put<uint8_t, A, P, kBlock, Local, kRemoteDest>(b.remote, nullptr,
                                                                             bytes, peer);
            break;
        case error_case::null_get_dest:
            result = nvshmemx_get_nbi<uint8_t, A, P, kBlock, kRemoteSource, local_dest(Local)>(
                nullptr, b.source, bytes, peer);
            break;
        case error_case::count_overflow:
            result = nvshmemx_put<uint64_t, A, P, kBlock, Local, kRemoteDest>(
                reinterpret_cast<uint64_t *>(b.remote), reinterpret_cast<uint64_t *>(b.source),
                static_cast<size_t>(-1) / 2, peer);
            break;
        case error_case::unmapped_put:
            result = nvshmemx_put<uint8_t, A, P, kBlock, Local, kRemoteDest>(b.remote, b.source,
                                                                             bytes, peer);
            break;
        case error_case::unmapped_signal:
            result = nvshmemx_signal_op<A, P, kBlock>(b.signal, 1, NVSHMEM_SIGNAL_SET, peer);
            break;
    }
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/*
 * Group-uniform preflight: one participant per group passes an invalid
 * argument (an out-of-range PE, or an invalid signal opcode). The group must
 * agree before issuing, so every participant returns INVALID_VALUE and nothing
 * is written. Every warp (or the block) issues to its own slice.
 */
enum class divergent_op { put_pe, put_signal_op };

template <typename A, typename P, threadgroup Scope, LocalSpace Local>
__global__ void divergent_kernel(divergent_op which, buffers b, size_t bytes, int peer, int npes,
                                 int *status) {
    const int group = scope_group_index<Scope>();
    const int groups = Scope == threadgroup::warp ? blockDim.x / warpSize : 1;
    const bool last = Scope == threadgroup::warp ? threadIdx.x % warpSize == warpSize - 1
                                                 : threadIdx.x == blockDim.x - 1;
    const size_t slice = bytes / groups;
    unsigned char *remote = b.remote + group * slice;
    const unsigned char *source = b.source + group * slice;
    int result;
    if (which == divergent_op::put_pe) {
        result = nvshmemx_put<uint8_t, A, P, Scope, Local, kRemoteDest>(remote, source, slice,
                                                                        last ? npes : peer);
    } else {
        result = nvshmemx_put_signal<uint8_t, A, P, Scope, Local, kRemoteDest>(
            remote, source, slice, b.signal, 1,
            last ? NVSHMEM_SIGNAL_ADD + 100 : NVSHMEM_SIGNAL_SET, peer);
    }
    if (result != NVSHMEMX_ERROR_INVALID_VALUE) {
        atomicExch(status, result);
    }
}

struct route_context {
    world w;
    buffers b;
    bool active;
    int peer;
    int previous;

    scoped_target scoped() const {
        return scoped_target{w,        active, peer,     previous, b.remote,
                             b.source, b.out,  b.signal, kBytes};
    }
};

template <typename A, typename P, LocalSpace Local, bool InShared, bool Nbi>
int put_case(const route_context &c, const char *what) {
    const size_t smem = InShared ? kBytes : 0;
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, 0, kBytes));
        },
        [&] {
            put_kernel<A, P, Local, InShared, Nbi>
                <<<1, kThreads, smem>>>(c.b.remote, c.b.source, kBytes, c.peer, c.w.status);
        },
        [&] { return check_payload(what, c.b.remote, kBytes, c.previous); });
}

template <typename A, typename P, LocalSpace Local, bool InShared, bool Nbi>
int get_case(const route_context &c, const char *what) {
    const size_t smem = InShared ? kBytes : 0;
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.out, 0, kBytes));
        },
        [&] {
            get_kernel<A, P, Local, InShared, Nbi>
                <<<1, kThreads, smem>>>(c.b.out, c.b.source, kBytes, c.peer, c.w.status);
        },
        [&] { return check_payload(what, c.b.out, kBytes, c.peer); });
}

template <typename A, typename P, LocalSpace Local, bool Nbi>
int put_signal_case(const route_context &c, const char *what, int sig_op) {
    const uint64_t initial = sig_op == NVSHMEM_SIGNAL_ADD ? kSignalBase : 0;
    const uint64_t value =
        sig_op == NVSHMEM_SIGNAL_ADD ? static_cast<uint64_t>(c.w.mype) + 1 : kSignalBase + c.w.mype;
    const uint64_t expected = sig_op == NVSHMEM_SIGNAL_ADD
                                  ? kSignalBase + static_cast<uint64_t>(c.previous) + 1
                                  : kSignalBase + static_cast<uint64_t>(c.previous);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, 0, kBytes));
            set_signal(c.b.signal, initial);
        },
        [&] {
            put_signal_kernel<A, P, Local, Nbi><<<1, kThreads>>>(
                c.b.remote, c.b.source, kBytes, c.b.signal, value, sig_op, c.peer, c.w.status);
        },
        [&] {
            return check_payload(what, c.b.remote, kBytes, c.previous) +
                   check_signal_and_wait(what, c.b.signal, expected, c.w.status);
        });
}

template <typename A, typename P>
int signal_op_case(const route_context &c, const char *what, int sig_op) {
    const uint64_t initial = sig_op == NVSHMEM_SIGNAL_ADD ? kSignalBase : 0;
    const uint64_t value =
        sig_op == NVSHMEM_SIGNAL_ADD ? static_cast<uint64_t>(c.w.mype) + 1 : kSignalBase + c.w.mype;
    const uint64_t expected = sig_op == NVSHMEM_SIGNAL_ADD
                                  ? kSignalBase + static_cast<uint64_t>(c.previous) + 1
                                  : kSignalBase + static_cast<uint64_t>(c.previous);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS, [&] { set_signal(c.b.signal, initial); },
        [&] {
            signal_op_kernel<A, P><<<1, kThreads>>>(c.b.signal, value, sig_op, c.peer, c.w.status);
        },
        [&] { return check_signal_and_wait(what, c.b.signal, expected, c.w.status); });
}

template <typename A, typename P, LocalSpace Local>
int fence_case(const route_context &c, const char *what) {
    const uint64_t expected = kSignalBase + static_cast<uint64_t>(c.previous);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, 0, kBytes));
            set_signal(c.b.signal, 0);
        },
        [&] {
            fence_kernel<A, P, Local><<<1, kThreads>>>(c.b.remote, c.b.source, kBytes, c.b.signal,
                                                       kSignalBase + c.w.mype, c.peer, c.w.status);
        },
        [&] {
            return check_signal_and_wait(what, c.b.signal, expected, c.w.status) +
                   check_payload(what, c.b.remote, kBytes, c.previous);
        });
}

int typed_case(const route_context &c, typed_kernel_t kernel, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, 0, kBytes));
            CUDA_CHECK(cudaMemset(c.b.out, 0, kBytes));
        },
        [&] {
            kernel<<<1, kThreads>>>(c.b.remote, c.b.source, c.b.out, kBytes, c.peer, c.w.status);
        },
        [&] {
            return check_payload(what, c.b.remote, kBytes, c.previous) +
                   check_payload(what, c.b.out, kBytes, c.peer);
        });
}

int zero_length_case(const route_context &c, const char *what,
                     void (*kernel)(unsigned char *, const unsigned char *, int, int *)) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            CUDA_CHECK(cudaMemset(c.b.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.b.source, kSentinelByte, kBytes));
        },
        [&] { kernel<<<1, kThreads>>>(c.b.remote, c.b.source, c.peer, c.w.status); },
        [&] {
            return check_fill(what, c.b.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.b.source, kBytes, kSentinelByte);
        });
}

/* A rejected call must not issue work: every PE's destination, get buffer,
 * and signal keep their sentinels. */
template <typename A, typename P, LocalSpace Local>
int error_case_run(const route_context &c, const char *what, error_case which,
                   int expected = NVSHMEMX_ERROR_INVALID_VALUE) {
    return run_case(
        c.w, c.active, what, expected,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.b.out, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.b.private_buffer, kSentinelByte, kBytes));
            set_signal(c.b.signal, kSentinelSignal);
            set_signal(c.b.private_signal, kSentinelSignal);
        },
        [&] {
            error_kernel<A, P, Local>
                <<<1, kThreads>>>(which, c.b, kBytes, c.peer, c.w.npes, c.w.status);
        },
        [&] {
            return check_fill(what, c.b.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.b.out, kBytes, kSentinelByte) +
                   check_fill(what, c.b.private_buffer, kBytes, kSentinelByte) +
                   check_signal(what, c.b.signal, kSentinelSignal) +
                   check_signal(what, c.b.private_signal, kSentinelSignal);
        });
}

template <typename A, typename P, LocalSpace Local>
int error_cases(const route_context &c) {
    int failures = 0;
    failures += error_case_run<A, P, Local>(c, "invalid PE", error_case::invalid_pe);
    failures += error_case_run<A, P, Local>(c, "put_signal invalid op", error_case::invalid_sig_op);
    failures +=
        error_case_run<A, P, Local>(c, "signal_op invalid op", error_case::invalid_signal_op);
    failures +=
        error_case_run<A, P, Local>(c, "non-symmetric put dest", error_case::private_put_dest);
    failures +=
        error_case_run<A, P, Local>(c, "non-symmetric get source", error_case::private_get_source);
    failures += error_case_run<A, P, Local>(c, "non-symmetric signal", error_case::private_signal);
    failures += error_case_run<A, P, Local>(c, "null put source", error_case::null_put_source);
    failures += error_case_run<A, P, Local>(c, "null get destination", error_case::null_get_dest);
    failures +=
        error_case_run<A, P, Local>(c, "element count overflow", error_case::count_overflow);
    return failures;
}

template <typename A, typename P, LocalSpace Local>
int common_cases(const route_context &c) {
    int failures = 0;
    failures += put_signal_case<A, P, Local, false>(c, "put_signal SET", NVSHMEM_SIGNAL_SET);
    failures += put_signal_case<A, P, Local, false>(c, "put_signal ADD", NVSHMEM_SIGNAL_ADD);
    failures += put_signal_case<A, P, Local, true>(c, "put_signal_nbi SET", NVSHMEM_SIGNAL_SET);
    failures += put_signal_case<A, P, Local, true>(c, "put_signal_nbi ADD", NVSHMEM_SIGNAL_ADD);
    failures += signal_op_case<A, P>(c, "signal_op SET", NVSHMEM_SIGNAL_SET);
    failures += signal_op_case<A, P>(c, "signal_op ADD", NVSHMEM_SIGNAL_ADD);
    failures += fence_case<A, P, Local>(c, "put_nbi+fence+signal_op");
    failures += typed_case(c, typed_kernel<uint8_t, A, P, Local>, "typed uint8 put/get_nbi");
    failures += typed_case(c, typed_kernel<uint32_t, A, P, Local>, "typed uint32 put/get_nbi");
    failures += typed_case(c, typed_kernel<uint64_t, A, P, Local>, "typed uint64 put/get_nbi");
    failures += error_cases<A, P, Local>(c);
    return failures;
}

template <typename A, typename P, threadgroup Scope, LocalSpace Local>
int divergent_case(const route_context &c, const char *what, divergent_op which) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_ERROR_INVALID_VALUE,
        [&] {
            fill_payload<<<1, kThreads>>>(c.b.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.b.remote, kSentinelByte, kBytes));
            set_signal(c.b.signal, kSentinelSignal);
        },
        [&] {
            const int expected = NVSHMEMX_ERROR_INVALID_VALUE;
            CUDA_CHECK(cudaMemcpy(c.w.status, &expected, sizeof(expected), cudaMemcpyHostToDevice));
            divergent_kernel<A, P, Scope, Local>
                <<<1, kThreads>>>(which, c.b, kBytes, c.peer, c.w.npes, c.w.status);
        },
        [&] {
            return check_fill(what, c.b.remote, kBytes, kSentinelByte) +
                   check_signal(what, c.b.signal, kSentinelSignal);
        });
}

template <typename A, typename P, LocalSpace Local>
int divergent_cases(const route_context &c) {
    int failures = 0;
    failures += divergent_case<A, P, threadgroup::warp, Local>(c, "warp-scope divergent PE",
                                                               divergent_op::put_pe);
    failures += divergent_case<A, P, threadgroup::block, Local>(c, "block-scope divergent PE",
                                                                divergent_op::put_pe);
    failures += divergent_case<A, P, threadgroup::warp, Local>(c, "warp-scope divergent sig_op",
                                                               divergent_op::put_signal_op);
    failures += divergent_case<A, P, threadgroup::block, Local>(c, "block-scope divergent sig_op",
                                                                divergent_op::put_signal_op);
    return failures;
}

/* On PEs whose ring neighbour has no direct mapping, the explicit Pointer / SM
 * route reports the missing capability instead of falling back. */
int pointer_sm_unmapped_cases(const route_context &c) {
    using A = nvshmemx::addressing::Pointer;
    using P = nvshmemx::transport_policy::SM;
    return error_case_run<A, P, kGlobal>(c, "SM put to unmapped peer", error_case::unmapped_put,
                                         NVSHMEMX_ERROR_NOT_SUPPORTED) +
           error_case_run<A, P, kGlobal>(c, "SM signal_op to unmapped peer",
                                         error_case::unmapped_signal, NVSHMEMX_ERROR_NOT_SUPPORTED);
}

int pointer_sm_cases(const route_context &c) {
    using A = nvshmemx::addressing::Pointer;
    using P = nvshmemx::transport_policy::SM;
    int failures = 0;
    failures += put_case<A, P, kShared, true, false>(c, "SM put Shared");
    failures += put_case<A, P, kGlobal, false, false>(c, "SM put Global");
    failures += put_case<A, P, kGeneric, true, false>(c, "SM put generic(shared)");
    failures += put_case<A, P, kGeneric, false, false>(c, "SM put generic(global)");
    failures += put_case<A, P, kShared, true, true>(c, "SM put_nbi Shared+flush reuse");
    failures += put_case<A, P, kGlobal, false, true>(c, "SM put_nbi Global+flush reuse");
    failures += put_case<A, P, kGeneric, false, true>(c, "SM put_nbi generic+flush reuse");
    failures += get_case<A, P, kShared, true, false>(c, "SM get Shared");
    failures += get_case<A, P, kGlobal, false, false>(c, "SM get Global");
    failures += get_case<A, P, kGeneric, true, false>(c, "SM get generic(shared)");
    failures += get_case<A, P, kGeneric, false, false>(c, "SM get generic(global)");
    failures += get_case<A, P, kShared, true, true>(c, "SM get_nbi Shared+quiet");
    failures += get_case<A, P, kGlobal, false, true>(c, "SM get_nbi Global+quiet");
    failures += get_case<A, P, kGeneric, false, true>(c, "SM get_nbi generic+quiet");
    failures += common_cases<A, P, kGlobal>(c);
    failures += divergent_cases<A, P, kGlobal>(c);
    failures += zero_length_case(c, "SM zero-length null local", zero_length_kernel<A, P>);
    failures += thread_warp_cases<A, P, kGlobal>(c.scoped(), "SM");
    /* Auto / Auto keeps the portable zero-length behavior; a direct peer keeps
     * the null local operand off the network transport. */
    failures += zero_length_case(
        c, "Auto zero-length null local",
        zero_length_kernel<nvshmemx::addressing::Auto, nvshmemx::transport_policy::Auto>);
    return failures;
}

/* A shared-memory local operand is unregistered, so like the stable API the
 * portable route can move it only to a directly mapped peer; the network
 * transports require a registered local buffer. */
int auto_auto_shared_cases(const route_context &c) {
    using A = nvshmemx::addressing::Auto;
    using P = nvshmemx::transport_policy::Auto;
    int failures = 0;
    failures += put_case<A, P, kShared, true, false>(c, "Auto put Shared");
    failures += put_case<A, P, kShared, true, true>(c, "Auto put_nbi Shared+flush reuse");
    failures += get_case<A, P, kShared, true, false>(c, "Auto get Shared");
    failures += get_case<A, P, kShared, true, true>(c, "Auto get_nbi Shared+quiet");
    return failures;
}

int auto_auto_cases(const route_context &c) {
    using A = nvshmemx::addressing::Auto;
    using P = nvshmemx::transport_policy::Auto;
    int failures = 0;
    failures += put_case<A, P, kGeneric, false, false>(c, "Auto put");
    failures += put_case<A, P, kGeneric, false, true>(c, "Auto put_nbi+flush reuse");
    failures += get_case<A, P, kGeneric, false, false>(c, "Auto get");
    failures += get_case<A, P, kGeneric, false, true>(c, "Auto get_nbi+quiet");
    /* Explicit local memory spaces are legal and use the portable dispatch. */
    failures += put_case<A, P, kGlobal, false, false>(c, "Auto put Global");
    failures += put_case<A, P, kGlobal, false, true>(c, "Auto put_nbi Global+flush reuse");
    failures += get_case<A, P, kGlobal, false, false>(c, "Auto get Global");
    failures += get_case<A, P, kGlobal, false, true>(c, "Auto get_nbi Global+quiet");
    /* Thread and warp scopes never select the block-only CFT path. */
    failures += thread_warp_cases<A, P, kGeneric>(c.scoped(), "Auto");
    failures += common_cases<A, P, kGeneric>(c);
    failures += divergent_cases<A, P, kGeneric>(c);
    failures += auto_vs_stable_cases(c.scoped(),
                                     compare_buffers{c.b.remote, c.b.out, c.b.signal,
                                                     c.b.ref_remote, c.b.ref_out, c.b.ref_signal},
                                     "Auto");
    return failures;
}

}  // namespace

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            printf("transport_specific_p2p: SKIP (requires at least two PEs)\n");
        }
        finalize_wrapper();
        return 0;
    }

    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    buffers b{};
    b.remote = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.source = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.out = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.signal = static_cast<uint64_t *>(nvshmem_malloc(sizeof(uint64_t)));
    b.ref_remote = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.ref_out = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    b.ref_signal = static_cast<uint64_t *>(nvshmem_malloc(sizeof(uint64_t)));
    int *status = nullptr;
    const bool local_ok = scratch && b.remote && b.source && b.out && b.signal && b.ref_remote &&
                          b.ref_out && b.ref_signal &&
                          cudaMalloc(&status, sizeof(*status)) == cudaSuccess &&
                          cudaMalloc(&b.private_buffer, kBytes) == cudaSuccess &&
                          cudaMalloc(&b.private_signal, sizeof(uint64_t)) == cudaSuccess;
    /* A partial symmetric allocation cannot be collectively freed. */
    if (!scratch || world_max(scratch, !local_ok) != 0) {
        fprintf(stderr, "PE %d: transport_specific_p2p setup failed\n", mype);
        nvshmem_global_exit(1);
        return 1;
    }

    const world w{mype, npes, scratch, status};
    int failures = 0;

    /* Pointer / SM runs inside each direct-P2P clique of TEAM_SHARED; ranks
     * without a mapped ring neighbour only join the barriers. */
    const int shared_npes = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
    const int shared_mype = nvshmem_team_my_pe(NVSHMEM_TEAM_SHARED);
    int strict_peer = -1;
    int strict_previous = -1;
    if (shared_npes >= 2 && shared_mype >= 0) {
        strict_peer = nvshmem_team_translate_pe(
            NVSHMEM_TEAM_SHARED, (shared_mype + 1) % shared_npes, NVSHMEM_TEAM_WORLD);
        strict_previous = nvshmem_team_translate_pe(
            NVSHMEM_TEAM_SHARED, (shared_mype + shared_npes - 1) % shared_npes, NVSHMEM_TEAM_WORLD);
    }
    const bool strict_active = strict_peer >= 0 && strict_previous >= 0 &&
                               nvshmem_ptr(b.remote, strict_peer) != nullptr &&
                               nvshmem_ptr(b.remote, strict_previous) != nullptr;
    const route_context sm{w, b, strict_active, strict_peer, strict_previous};
    const int sm_failures = world_max(scratch, pointer_sm_cases(sm));
    if (world_max(scratch, strict_active) == 0) {
        if (mype == 0) {
            printf("transport_specific_p2p: SKIP Pointer / SM (no direct peer clique)\n");
        }
    } else {
        report(mype, "transport_specific_p2p", "Pointer / SM", sm_failures);
    }
    failures += sm_failures;

    const route_context portable{w, b, true, (mype + 1) % npes, (mype + npes - 1) % npes};

    const bool unmapped = nvshmem_ptr(b.remote, portable.peer) == nullptr;
    const route_context sm_unmapped{w, b, unmapped, portable.peer, portable.previous};
    const int unmapped_failures = world_max(scratch, pointer_sm_unmapped_cases(sm_unmapped));
    if (world_max(scratch, unmapped) == 0) {
        if (mype == 0) {
            printf(
                "transport_specific_p2p: Pointer / SM unmapped-peer status not exercised "
                "(every ring peer is mapped)\n");
        }
    } else {
        report(mype, "transport_specific_p2p", "Pointer / SM unmapped peer", unmapped_failures);
    }
    failures += unmapped_failures;

    const int auto_failures = world_max(scratch, auto_auto_cases(portable));
    report(mype, "transport_specific_p2p", "Auto / Auto", auto_failures);
    failures += auto_failures;

    /* Every PE runs these or none does, so no PE waits on an unsent ring peer. */
    const bool mapped_ring =
        world_max(scratch, nvshmem_ptr(b.remote, portable.peer) == nullptr ||
                               nvshmem_ptr(b.remote, portable.previous) == nullptr) == 0;
    const route_context auto_shared{w, b, mapped_ring, portable.peer, portable.previous};
    const int auto_shared_failures = world_max(scratch, auto_auto_shared_cases(auto_shared));
    if (!mapped_ring) {
        if (mype == 0) {
            printf(
                "transport_specific_p2p: SKIP Auto / Auto shared local operand (no directly "
                "mapped ring peer)\n");
        }
    } else {
        report(mype, "transport_specific_p2p", "Auto / Auto shared local", auto_shared_failures);
    }
    failures += auto_shared_failures;

    nvshmem_barrier_all();
    nvshmem_free(b.remote);
    nvshmem_free(b.source);
    nvshmem_free(b.out);
    nvshmem_free(b.signal);
    nvshmem_free(b.ref_remote);
    nvshmem_free(b.ref_out);
    nvshmem_free(b.ref_signal);
    nvshmem_free(scratch);
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(b.private_buffer));
    CUDA_CHECK(cudaFree(b.private_signal));
    finalize_wrapper();
    return failures ? 1 : 0;
}
