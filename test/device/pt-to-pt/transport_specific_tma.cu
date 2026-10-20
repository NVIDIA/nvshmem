/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Functional coverage for the strict Pointer / TMA route. The route never
 * falls back to another data path, so once its prerequisites hold (an sm_90+
 * device, NVSHMEM_TMA_POLICY=ENABLE, registered CTA shared memory, and a
 * directly mapped peer) every positive case must succeed. When they do not,
 * the route reports NOT_SUPPORTED and the executable is a clean skip. Each
 * kernel keeps its shared memory registered until its final completion call.
 *
 * Signalled cases launch a second CTA that receives the previous PE's payload
 * and signal concurrently and checks the payload as soon as the signal lands.
 * TMA registration is per CTA: that receiver neither registers shared memory
 * nor issues TMA work. Gets check their result in the kernel right after the
 * completion call that makes it visible. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_common.cuh"

using namespace ts_test;

namespace {

constexpr size_t kBytes = 4096;
using Pointer = nvshmemx::addressing::Pointer;
using TMA = nvshmemx::transport_policy::TMA;

/* Dynamic shared memory: [0, kRegistered) may be given to NVSHMEM and the
 * user payload area follows it. */
constexpr int kRegistered = 65536; /* nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED) */
constexpr size_t kSmem = static_cast<size_t>(kRegistered) + kBytes;

enum class boundary { quiet, flush, fence };

template <threadgroup Scope, boundary B>
__device__ int complete() {
    if constexpr (B == boundary::quiet) {
        return nvshmemx_quiet<Pointer, TMA, Scope>();
    } else if constexpr (B == boundary::flush) {
        return nvshmemx_flush<Pointer, TMA, Scope>();
    } else {
        return nvshmemx_fence<Pointer, TMA, Scope>();
    }
}

__device__ unsigned char *smem_base() {
    extern __shared__ __align__(16) unsigned char smem[];
    return smem;
}

/* Registers `registered` bytes (none when 0) and returns the payload area. */
__device__ unsigned char *register_smem(int registered) {
    if (registered > 0) {
        nvshmemx_give_smem(smem_base(), registered);
    }
    __syncthreads();
    return smem_base() + kRegistered;
}

__device__ void release_smem(int registered) {
    __syncthreads();
    if (registered > 0) {
        nvshmemx_release_smem();
    }
    __syncthreads();
}

/* Every kernel launched with kSmem needs the dynamic shared-memory opt-in. */
template <typename Kernel, typename... Args>
void launch_blocks(int blocks, Kernel kernel, Args... args) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSmem)));
    kernel<<<blocks, kThreads, kSmem>>>(args...);
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, Args... args) {
    launch_blocks(1, kernel, args...);
}

/* Whether the whole block holds `pe`'s payload in `buffer`; block-uniform. */
__device__ bool block_payload_matches(const unsigned char *buffer, size_t bytes, int pe) {
    return !__syncthreads_or(!payload_matches(buffer, bytes, pe, 0, threadIdx.x, blockDim.x));
}

/* A zero-length block put from registered shared memory: it runs the route's
 * capability checks and issues nothing. */
__global__ void probe_kernel(unsigned char *remote, int peer, int *status) {
    unsigned char *payload = register_smem(kRegistered);
    const int result =
        nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(remote, payload, 0, peer);
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* ---- Block scope ---------------------------------------------------------- */

template <LocalSpace Local, bool InShared, bool Nbi>
__global__ void put_kernel(unsigned char *remote, unsigned char *source, size_t bytes,
                           int registered, int peer, int *status) {
    unsigned char *payload = register_smem(registered);
    unsigned char *local = InShared ? payload : source;
    if (InShared) {
        copy_block(payload, source, bytes);
    }
    __syncthreads();

    int result;
    if constexpr (Nbi) {
        result = nvshmemx_put_nbi<uint8_t, Pointer, TMA, kBlock, Local, kRemoteDest>(remote, local,
                                                                                     bytes, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, boundary::flush>();
        }
        if (result == NVSHMEMX_SUCCESS) {
            /* flush makes the source reusable; the peer must still see the original. */
            fill_block(local, bytes, kPoisonByte);
            __syncthreads();
            result = complete<kBlock, boundary::quiet>();
        }
    } else {
        result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, Local, kRemoteDest>(remote, local,
                                                                                 bytes, peer);
    }
    release_smem(registered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* NBI gets issue two outstanding transfers before one completion call, which
 * must complete both. Shared-destination NBI gets are deferred on the CTA's
 * transaction mbarrier until that call. */
template <LocalSpace Local, bool InShared, bool Nbi, boundary B>
__global__ void get_kernel(unsigned char *out, const unsigned char *remote_source, size_t bytes,
                           int peer, int *status) {
    constexpr auto kLocalDest = local_dest(Local);
    unsigned char *payload = register_smem(kRegistered);
    unsigned char *local = InShared ? payload : out;
    if (InShared) {
        fill_block(payload, bytes, 0);
    }
    __syncthreads();

    int result;
    if constexpr (Nbi) {
        const size_t half = bytes / 2;
        result = nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, kLocalDest>(
            local, remote_source, half, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, kLocalDest>(
                local + half, remote_source + half, bytes - half, peer);
        }
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, B>();
        }
    } else {
        result = nvshmemx_get<uint8_t, Pointer, TMA, kBlock, kRemoteSource, kLocalDest>(
            local, remote_source, bytes, peer);
    }
    __syncthreads();
    if (result == NVSHMEMX_SUCCESS && !block_payload_matches(local, bytes, peer)) {
        result = kDataMismatch;
    }
    if (InShared && result == NVSHMEMX_SUCCESS) {
        copy_block(out, payload, bytes);
    }
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* Block 0 sends to `peer`; block 1 receives from `previous` concurrently. */
template <LocalSpace Local, bool InShared, bool Nbi>
__global__ void put_signal_kernel(unsigned char *remote, const unsigned char *source, size_t bytes,
                                  uint64_t *signal, uint64_t value, int sig_op, int peer,
                                  uint64_t expected, int previous, int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, expected, remote, bytes, previous, status);
        return;
    }
    unsigned char *payload = register_smem(kRegistered);
    const unsigned char *local = InShared ? payload : source;
    if (InShared) {
        copy_block(payload, source, bytes);
    }
    __syncthreads();

    int result;
    if constexpr (Nbi) {
        result = nvshmemx_put_signal_nbi<uint8_t, Pointer, TMA, kBlock, Local, kRemoteDest>(
            remote, local, bytes, signal, value, sig_op, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, boundary::quiet>();
        }
    } else {
        result = nvshmemx_put_signal<uint8_t, Pointer, TMA, kBlock, Local, kRemoteDest>(
            remote, local, bytes, signal, value, sig_op, peer);
    }
    release_smem(kRegistered);
    report_sender(status, result);
}

/* The standalone signal needs no TMA state; block 1 waits for the previous
 * PE's signal while block 0 sends. */
__global__ void signal_op_kernel(uint64_t *signal, uint64_t value, int sig_op, int peer,
                                 uint64_t expected, int previous, int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, expected, nullptr, 0, previous, status);
        return;
    }
    const int result = nvshmemx_signal_op<Pointer, TMA, kBlock>(signal, value, sig_op, peer);
    report_sender(status, result);
}

/* fence orders the NBI payload before the following signal; block 1 checks
 * that order on the receiving side while the peer sends. */
__global__ void fence_kernel(unsigned char *remote, const unsigned char *source, size_t bytes,
                             uint64_t *signal, uint64_t value, int peer, uint64_t expected,
                             int previous, int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, expected, remote, bytes, previous, status);
        return;
    }
    unsigned char *payload = register_smem(kRegistered);
    copy_block(payload, source, bytes);
    __syncthreads();
    int result = nvshmemx_put_nbi<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
        remote, payload, bytes, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<kBlock, boundary::fence>();
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_signal_op<Pointer, TMA, kBlock>(signal, value, NVSHMEM_SIGNAL_SET, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<kBlock, boundary::quiet>();
    }
    release_smem(kRegistered);
    report_sender(status, result);
}

template <typename T>
__global__ void typed_kernel(unsigned char *remote, unsigned char *source, unsigned char *out,
                             size_t bytes, int peer, int *status) {
    unsigned char *payload = register_smem(kRegistered);
    (void)payload;
    const size_t nelems = bytes / sizeof(T);
    int result = nvshmemx_put<T, Pointer, TMA, kBlock, kGlobal, kRemoteDest>(
        reinterpret_cast<T *>(remote), reinterpret_cast<const T *>(source), nelems, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<T, Pointer, TMA, kBlock, kRemoteSource, local_dest(kGlobal)>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(source), nelems, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<kBlock, boundary::quiet>();
    }
    if (result == NVSHMEMX_SUCCESS && !block_payload_matches(out, bytes, peer)) {
        result = kDataMismatch;
    }
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* A zero-length transfer may name a null local operand at any legal scope:
 * it succeeds once the route is available and modifies nothing. */
__global__ void zero_length_kernel(unsigned char *remote, const unsigned char *remote_source,
                                   int peer, int *status) {
    constexpr auto kLocalDest = local_dest(kGeneric);
    register_smem(kRegistered);
    int result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kGeneric, kRemoteDest>(remote, nullptr,
                                                                                    0, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, kLocalDest>(
            nullptr, remote_source, 0, peer);
    }
    if (result == NVSHMEMX_SUCCESS && threadIdx.x == 0) {
        result =
            nvshmemx_put_nbi<uint8_t, Pointer, TMA, threadgroup::thread, kGeneric, kRemoteDest>(
                remote, nullptr, 0, peer);
    }
    if (__syncthreads_or(result != NVSHMEMX_SUCCESS)) {
        result = result != NVSHMEMX_SUCCESS ? result : NVSHMEMX_ERROR_INTERNAL;
    } else {
        result = complete<kBlock, boundary::quiet>();
    }
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* ---- Thread and warp scope ------------------------------------------------ */

/*
 * Each of the first `issuers` threads or warps of the CTA moves its own slice
 * from the CTA's shared payload, so the shared-source put family runs with
 * several independent concurrent issuers.
 */
enum class scoped_op {
    put_quiet,           /* put, then quiet */
    put_nbi_flush,       /* put_nbi + flush, poison the source slice, quiet */
    put_nbi_fence,       /* put_nbi + fence + signal_op ADD 1 + quiet */
    put_signal,          /* put_signal ADD 1 */
    put_signal_nbi_quiet /* put_signal_nbi ADD 1 + quiet */
};

template <threadgroup Scope, LocalSpace Local>
__device__ int scoped_put(scoped_op which, unsigned char *remote, unsigned char *local,
                          uint64_t *signal, size_t bytes, int peer) {
    int result;
    switch (which) {
        case scoped_op::put_quiet:
            result = nvshmemx_put<uint8_t, Pointer, TMA, Scope, Local, kRemoteDest>(remote, local,
                                                                                    bytes, peer);
            break;
        case scoped_op::put_signal:
            return nvshmemx_put_signal<uint8_t, Pointer, TMA, Scope, Local, kRemoteDest>(
                remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
        case scoped_op::put_signal_nbi_quiet:
            result = nvshmemx_put_signal_nbi<uint8_t, Pointer, TMA, Scope, Local, kRemoteDest>(
                remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            break;
        default:
            result = nvshmemx_put_nbi<uint8_t, Pointer, TMA, Scope, Local, kRemoteDest>(
                remote, local, bytes, peer);
            break;
    }
    if (result == NVSHMEMX_SUCCESS && which == scoped_op::put_nbi_flush) {
        result = complete<Scope, boundary::flush>();
        if (result == NVSHMEMX_SUCCESS) {
            fill_group<Scope>(local, bytes, kPoisonByte);
        }
    } else if (result == NVSHMEMX_SUCCESS && which == scoped_op::put_nbi_fence) {
        result = complete<Scope, boundary::fence>();
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_signal_op<Pointer, TMA, Scope>(signal, 1, NVSHMEM_SIGNAL_ADD, peer);
        }
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<Scope, boundary::quiet>();
    }
    return result;
}

constexpr bool scoped_signals(scoped_op which) {
    return which == scoped_op::put_nbi_fence || which == scoped_op::put_signal ||
           which == scoped_op::put_signal_nbi_quiet;
}

/* Reports only failures; the caller zeroes *status. Block 0 issues; for a
 * signalled op, block 1 receives the previous PE's payload and the signal
 * that all of its `issuers` groups add. */
template <threadgroup Scope, LocalSpace Local>
__global__ void scoped_kernel(scoped_op which, unsigned char *remote, const unsigned char *source,
                              uint64_t *signal, size_t bytes, int issuers, int peer, int previous,
                              int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, kSignalBase + static_cast<uint64_t>(issuers), remote, bytes,
                          previous, status);
        return;
    }
    unsigned char *payload = register_smem(kRegistered);
    copy_block(payload, source, bytes);
    __syncthreads();

    const int group = scope_group_index<Scope>();
    if (group < issuers) {
        const size_t slice = bytes / issuers;
        const size_t offset = static_cast<size_t>(group) * slice;
        const int result =
            scoped_put<Scope, Local>(which, remote + offset, payload + offset, signal, slice, peer);
        if (result != NVSHMEMX_SUCCESS) {
            atomicExch(status, result);
        }
    }
    release_smem(kRegistered);
}

/* Group-uniform preflight below block scope: the last lane of each warp names
 * an out-of-range PE, so every lane must return INVALID_VALUE and no warp may
 * write its slice. */
__global__ void divergent_warp_kernel(unsigned char *remote, const unsigned char *source,
                                      size_t bytes, int peer, int npes, int *status) {
    unsigned char *payload = register_smem(kRegistered);
    copy_block(payload, source, bytes);
    __syncthreads();

    const int group = scope_group_index<threadgroup::warp>();
    const size_t slice = bytes / (blockDim.x / warpSize);
    const bool last = threadIdx.x % warpSize == warpSize - 1;
    const int result = nvshmemx_put<uint8_t, Pointer, TMA, threadgroup::warp, kShared, kRemoteDest>(
        remote + group * slice, payload + group * slice, slice, last ? npes : peer);
    if (result != NVSHMEMX_ERROR_INVALID_VALUE) {
        atomicExch(status, result);
    }
    release_smem(kRegistered);
}

/* ---- Rejected calls ------------------------------------------------------- */

/* Each case is rejected by preflight before any work is issued. */
enum class error_case {
    invalid_pe,                         /* INVALID_VALUE */
    generic_global_thread_put,          /* INVALID_VALUE: generic source not in shared memory */
    generic_global_warp_put_signal_nbi, /* INVALID_VALUE: same, with a signal */
    unaligned_size_put,                 /* INVALID_VALUE: size not a 16-byte multiple */
    misaligned_put_dest,                /* INVALID_VALUE: remote not 16-byte aligned */
    misaligned_thread_source,           /* INVALID_VALUE: local not 16-byte aligned */
    unaligned_put_signal_nbi,           /* INVALID_VALUE: size not a 16-byte multiple */
    unaligned_shared_get_nbi,           /* INVALID_VALUE: size not a 16-byte multiple */
    misaligned_global_get,              /* INVALID_VALUE: local not 16-byte aligned */
    reserved_region_put,                /* INVALID_VALUE: source in NVSHMEM's barrier region */
    reserved_region_get_nbi,            /* INVALID_VALUE: dest in NVSHMEM's barrier region */
    unregistered_put,                   /* NOT_SUPPORTED: no registered shared memory */
    unregistered_warp_put_signal,       /* NOT_SUPPORTED: no registered shared memory */
    barriers_only_shared_get,           /* NOT_SUPPORTED: no room past the barrier region */
    barriers_only_global_put,           /* NOT_SUPPORTED: no staging tile */
};

__host__ __device__ constexpr int error_registration(error_case which) {
    return which == error_case::unregistered_put ||
                   which == error_case::unregistered_warp_put_signal
               ? 0
           : which == error_case::barriers_only_shared_get ||
                   which == error_case::barriers_only_global_put
               ? NVSHMEMI_SMEM_DATA_REGION_OFFSET
               : kRegistered;
}

__global__ void error_kernel(error_case which, unsigned char *remote, const unsigned char *source,
                             unsigned char *out, uint64_t *signal, int peer, int npes,
                             int *status) {
    const int registered = error_registration(which);
    unsigned char *payload = register_smem(registered);
    copy_block(payload, source, kBytes);
    __syncthreads();

    const bool lane0 = threadIdx.x == 0;
    const bool warp0 = threadIdx.x < warpSize;
    constexpr auto kWarp = threadgroup::warp;
    constexpr auto kThread = threadgroup::thread;
    int result = NVSHMEMX_SUCCESS;
    switch (which) {
        case error_case::invalid_pe:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kBytes, npes);
            break;
        case error_case::generic_global_thread_put:
            if (lane0) {
                result = nvshmemx_put<uint8_t, Pointer, TMA, kThread, kGeneric, kRemoteDest>(
                    remote, source, kBytes, peer);
            }
            break;
        case error_case::generic_global_warp_put_signal_nbi:
            if (warp0) {
                result =
                    nvshmemx_put_signal_nbi<uint8_t, Pointer, TMA, kWarp, kGeneric, kRemoteDest>(
                        remote, source, kBytes, signal, 1, NVSHMEM_SIGNAL_SET, peer);
            }
            break;
        case error_case::unaligned_size_put:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, 15, peer);
            break;
        case error_case::misaligned_put_dest:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kGlobal, kRemoteDest>(
                remote + 1, source, 64, peer);
            break;
        case error_case::misaligned_thread_source:
            if (lane0) {
                result = nvshmemx_put<uint8_t, Pointer, TMA, kThread, kShared, kRemoteDest>(
                    remote, payload + 8, 64, peer);
            }
            break;
        case error_case::unaligned_put_signal_nbi:
            result = nvshmemx_put_signal_nbi<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, 15, signal, 1, NVSHMEM_SIGNAL_SET, peer);
            break;
        case error_case::unaligned_shared_get_nbi:
            result =
                nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, local_dest(kShared)>(
                    payload, source, 15, peer);
            break;
        case error_case::misaligned_global_get:
            result =
                nvshmemx_get<uint8_t, Pointer, TMA, kBlock, kRemoteSource, local_dest(kGlobal)>(
                    out + 1, source, 64, peer);
            break;
        case error_case::reserved_region_put:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
                remote, smem_base(), 64, peer);
            break;
        case error_case::reserved_region_get_nbi:
            result =
                nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, local_dest(kShared)>(
                    smem_base(), source, 64, peer);
            break;
        case error_case::unregistered_put:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kBytes, peer);
            break;
        case error_case::unregistered_warp_put_signal:
            if (warp0) {
                result = nvshmemx_put_signal<uint8_t, Pointer, TMA, kWarp, kShared, kRemoteDest>(
                    remote, payload, kBytes, signal, 1, NVSHMEM_SIGNAL_SET, peer);
            }
            break;
        case error_case::barriers_only_shared_get:
            result =
                nvshmemx_get<uint8_t, Pointer, TMA, kBlock, kRemoteSource, local_dest(kShared)>(
                    payload, source, kBytes, peer);
            break;
        case error_case::barriers_only_global_put:
            result = nvshmemx_put<uint8_t, Pointer, TMA, kBlock, kGlobal, kRemoteDest>(
                remote, source, kBytes, peer);
            break;
    }
    release_smem(registered);
    if (lane0) {
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
};

template <LocalSpace Local, bool InShared, bool Nbi>
int put_case(const route_context &c, const char *what, int registered = kRegistered) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
        },
        [&] {
            launch(put_kernel<Local, InShared, Nbi>, c.remote, c.source, kBytes, registered, c.peer,
                   c.w.status);
        },
        [&] { return check_payload(what, c.remote, kBytes, c.previous); });
}

template <LocalSpace Local, bool InShared, bool Nbi, boundary B = boundary::quiet>
int get_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.out, 0, kBytes));
        },
        [&] {
            launch(get_kernel<Local, InShared, Nbi, B>, c.out,
                   static_cast<const unsigned char *>(c.source), kBytes, c.peer, c.w.status);
        },
        [&] { return check_payload(what, c.out, kBytes, c.peer); });
}

template <LocalSpace Local, bool InShared, bool Nbi>
int put_signal_case(const route_context &c, const char *what, int sig_op) {
    const bool add = sig_op == NVSHMEM_SIGNAL_ADD;
    const uint64_t value = add ? static_cast<uint64_t>(c.w.mype) + 1 : kSignalBase + c.w.mype;
    const uint64_t expected = kSignalBase + static_cast<uint64_t>(c.previous) + (add ? 1 : 0);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            set_signal(c.signal, add ? kSignalBase : 0);
        },
        [&] {
            launch_blocks(2, put_signal_kernel<Local, InShared, Nbi>, c.remote,
                          static_cast<const unsigned char *>(c.source), kBytes, c.signal, value,
                          sig_op, c.peer, expected, c.previous, c.w.status);
        },
        [&] {
            return check_payload(what, c.remote, kBytes, c.previous) +
                   check_signal_and_wait(what, c.signal, expected, c.w.status);
        });
}

int signal_op_case(const route_context &c, const char *what, int sig_op) {
    const bool add = sig_op == NVSHMEM_SIGNAL_ADD;
    const uint64_t value = add ? static_cast<uint64_t>(c.w.mype) + 1 : kSignalBase + c.w.mype;
    const uint64_t expected = kSignalBase + static_cast<uint64_t>(c.previous) + (add ? 1 : 0);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS, [&] { set_signal(c.signal, add ? kSignalBase : 0); },
        [&] {
            signal_op_kernel<<<2, kThreads>>>(c.signal, value, sig_op, c.peer, expected, c.previous,
                                              c.w.status);
        },
        [&] { return check_signal_and_wait(what, c.signal, expected, c.w.status); });
}

int fence_case(const route_context &c, const char *what) {
    const uint64_t expected = kSignalBase + static_cast<uint64_t>(c.previous);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            set_signal(c.signal, 0);
        },
        [&] {
            launch_blocks(2, fence_kernel, c.remote, static_cast<const unsigned char *>(c.source),
                          kBytes, c.signal, kSignalBase + c.w.mype, c.peer, expected, c.previous,
                          c.w.status);
        },
        [&] {
            return check_signal_and_wait(what, c.signal, expected, c.w.status) +
                   check_payload(what, c.remote, kBytes, c.previous);
        });
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
        [&] { launch(typed_kernel<T>, c.remote, c.source, c.out, kBytes, c.peer, c.w.status); },
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
        },
        [&] {
            launch(zero_length_kernel, c.remote, static_cast<const unsigned char *>(c.source),
                   c.peer, c.w.status);
        },
        [&] {
            return check_fill(what, c.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.source, kBytes, kSentinelByte);
        });
}

template <threadgroup Scope, LocalSpace Local>
int scoped_case(const route_context &c, const char *what, scoped_op which, int issuers) {
    const bool signals = scoped_signals(which);
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            set_signal(c.signal, kSignalBase);
        },
        [&] {
            CUDA_CHECK(cudaMemset(c.w.status, 0, sizeof(*c.w.status)));
            launch_blocks(signals ? 2 : 1, scoped_kernel<Scope, Local>, which, c.remote,
                          static_cast<const unsigned char *>(c.source), c.signal, kBytes, issuers,
                          c.peer, c.previous, c.w.status);
        },
        [&] {
            int failures = check_payload(what, c.remote, kBytes, c.previous);
            if (signals) {
                failures += check_signal_and_wait(
                    what, c.signal, kSignalBase + static_cast<uint64_t>(issuers), c.w.status);
            }
            return failures;
        });
}

template <LocalSpace Local>
int thread_warp_cases(const route_context &c, const char *local) {
    const scoped_op ops[] = {scoped_op::put_quiet, scoped_op::put_nbi_flush,
                             scoped_op::put_nbi_fence, scoped_op::put_signal,
                             scoped_op::put_signal_nbi_quiet};
    const char *names[] = {"put+quiet", "put_nbi+flush reuse+quiet",
                           "put_nbi+fence+signal_op ADD+quiet", "put_signal ADD",
                           "put_signal_nbi ADD+quiet"};
    int failures = 0;
    for (int i = 0; i < 5; i++) {
        char what[160];
        snprintf(what, sizeof(what), "TMA thread-scope %s %s, %d issuers", local, names[i],
                 kThreads);
        failures += scoped_case<threadgroup::thread, Local>(c, what, ops[i], kThreads);
        snprintf(what, sizeof(what), "TMA warp-scope %s %s, %d issuers", local, names[i],
                 kThreads / 32);
        failures += scoped_case<threadgroup::warp, Local>(c, what, ops[i], kThreads / 32);
    }
    return failures;
}

int divergent_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_ERROR_INVALID_VALUE,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
        },
        [&] {
            const int expected = NVSHMEMX_ERROR_INVALID_VALUE;
            CUDA_CHECK(cudaMemcpy(c.w.status, &expected, sizeof(expected), cudaMemcpyHostToDevice));
            launch(divergent_warp_kernel, c.remote, static_cast<const unsigned char *>(c.source),
                   kBytes, c.peer, c.w.npes, c.w.status);
        },
        [&] { return check_fill(what, c.remote, kBytes, kSentinelByte); });
}

/* A rejected call must not issue work: every PE's destination, get buffer,
 * and signal keep their sentinels. */
int error_case_run(const route_context &c, const char *what, error_case which, int expected) {
    return run_case(
        c.w, c.active, what, expected,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.out, kSentinelByte, kBytes));
            set_signal(c.signal, kSentinelSignal);
        },
        [&] {
            launch(error_kernel, which, c.remote, static_cast<const unsigned char *>(c.source),
                   c.out, c.signal, c.peer, c.w.npes, c.w.status);
        },
        [&] {
            return check_fill(what, c.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.out, kBytes, kSentinelByte) +
                   check_signal(what, c.signal, kSentinelSignal);
        });
}

int error_cases(const route_context &c) {
    constexpr int kInvalid = NVSHMEMX_ERROR_INVALID_VALUE;
    constexpr int kNotSupported = NVSHMEMX_ERROR_NOT_SUPPORTED;
    const struct {
        error_case which;
        const char *what;
        int expected;
    } cases[] = {
        {error_case::invalid_pe, "TMA put invalid PE", kInvalid},
        {error_case::generic_global_thread_put, "TMA thread-scope put generic(global) source",
         kInvalid},
        {error_case::generic_global_warp_put_signal_nbi,
         "TMA warp-scope put_signal_nbi generic(global) source", kInvalid},
        {error_case::unaligned_size_put, "TMA put unaligned size", kInvalid},
        {error_case::misaligned_put_dest, "TMA put misaligned destination", kInvalid},
        {error_case::misaligned_thread_source, "TMA thread-scope put misaligned shared source",
         kInvalid},
        {error_case::unaligned_put_signal_nbi, "TMA put_signal_nbi unaligned size", kInvalid},
        {error_case::unaligned_shared_get_nbi, "TMA shared get_nbi unaligned size", kInvalid},
        {error_case::misaligned_global_get, "TMA get misaligned global destination", kInvalid},
        {error_case::reserved_region_put, "TMA put from the reserved barrier region", kInvalid},
        {error_case::reserved_region_get_nbi, "TMA get_nbi into the reserved barrier region",
         kInvalid},
        {error_case::unregistered_put, "TMA put without registered shared memory", kNotSupported},
        {error_case::unregistered_warp_put_signal,
         "TMA warp-scope put_signal without registered shared memory", kNotSupported},
        {error_case::barriers_only_shared_get, "TMA shared get with a barrier-only registration",
         kNotSupported},
        {error_case::barriers_only_global_put, "TMA global put with a barrier-only registration",
         kNotSupported},
    };
    int failures = 0;
    for (const auto &e : cases) {
        failures += error_case_run(c, e.what, e.which, e.expected);
    }
    return failures;
}

int tma_cases(const route_context &c) {
    int failures = 0;
    failures += put_case<kShared, true, false>(c, "TMA put Shared");
    failures += put_case<kGlobal, false, false>(c, "TMA put Global");
    failures += put_case<kGeneric, true, false>(c, "TMA put generic(shared)");
    failures += put_case<kGeneric, false, false>(c, "TMA put generic(global)");
    failures += put_case<kShared, true, true>(c, "TMA put_nbi Shared+flush reuse");
    failures += put_case<kGlobal, false, true>(c, "TMA put_nbi Global+flush reuse");
    failures += put_case<kGeneric, true, true>(c, "TMA put_nbi generic(shared)+flush reuse");
    failures += put_case<kGeneric, false, true>(c, "TMA put_nbi generic(global)+flush reuse");
    /* A put from a shared source needs only the barrier region. */
    failures += put_case<kShared, true, false>(c, "TMA put Shared, barrier-only registration",
                                               NVSHMEMI_SMEM_DATA_REGION_OFFSET);

    failures += get_case<kShared, true, false>(c, "TMA get Shared");
    failures += get_case<kGlobal, false, false>(c, "TMA get Global");
    failures += get_case<kGeneric, true, false>(c, "TMA get generic(shared)");
    failures += get_case<kGeneric, false, false>(c, "TMA get generic(global)");
    failures += get_case<kGlobal, false, true>(c, "TMA 2x get_nbi Global+quiet");
    failures += get_case<kGeneric, false, true>(c, "TMA 2x get_nbi generic(global)+quiet");
    failures +=
        get_case<kShared, true, true, boundary::quiet>(c, "TMA 2x get_nbi Shared deferred+quiet");
    failures +=
        get_case<kShared, true, true, boundary::flush>(c, "TMA 2x get_nbi Shared deferred+flush");
    failures +=
        get_case<kShared, true, true, boundary::fence>(c, "TMA 2x get_nbi Shared deferred+fence");
    failures += get_case<kGeneric, true, true, boundary::quiet>(
        c, "TMA 2x get_nbi generic(shared) deferred+quiet");

    failures +=
        put_signal_case<kShared, true, false>(c, "TMA put_signal Shared SET", NVSHMEM_SIGNAL_SET);
    failures +=
        put_signal_case<kShared, true, false>(c, "TMA put_signal Shared ADD", NVSHMEM_SIGNAL_ADD);
    failures +=
        put_signal_case<kGlobal, false, false>(c, "TMA put_signal Global ADD", NVSHMEM_SIGNAL_ADD);
    failures += put_signal_case<kShared, true, true>(c, "TMA put_signal_nbi Shared SET",
                                                     NVSHMEM_SIGNAL_SET);
    failures += put_signal_case<kShared, true, true>(c, "TMA put_signal_nbi Shared ADD",
                                                     NVSHMEM_SIGNAL_ADD);
    failures += put_signal_case<kGlobal, false, true>(c, "TMA put_signal_nbi Global SET",
                                                      NVSHMEM_SIGNAL_SET);
    failures += put_signal_case<kGeneric, true, true>(c, "TMA put_signal_nbi generic(shared) ADD",
                                                      NVSHMEM_SIGNAL_ADD);
    failures += signal_op_case(c, "TMA signal_op SET", NVSHMEM_SIGNAL_SET);
    failures += signal_op_case(c, "TMA signal_op ADD", NVSHMEM_SIGNAL_ADD);
    failures += fence_case(c, "TMA put_nbi Shared+fence+signal_op");
    failures += typed_case<uint32_t>(c, "TMA typed uint32 put/get_nbi");
    failures += typed_case<uint64_t>(c, "TMA typed uint64 put/get_nbi");
    failures += zero_length_case(c, "TMA zero-length null local");

    failures += thread_warp_cases<kShared>(c, "Shared");
    failures += thread_warp_cases<kGeneric>(c, "generic(shared)");
    failures += divergent_case(c, "TMA warp-scope divergent PE");

    failures += error_cases(c);
    return failures;
}

/* The route is unavailable on this job: an argument error still reports
 * INVALID_VALUE and a valid put NOT_SUPPORTED, never INTERNAL, and neither
 * touches any buffer. */
int unavailable_cases(const route_context &c) {
    return error_case_run(c, "TMA put invalid PE while the route is unavailable",
                          error_case::invalid_pe, NVSHMEMX_ERROR_INVALID_VALUE) +
           error_case_run(c, "TMA put while the route is unavailable", error_case::unregistered_put,
                          NVSHMEMX_ERROR_NOT_SUPPORTED);
}

}  // namespace

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            printf("transport_specific_tma: SKIP (requires at least two PEs)\n");
        }
        finalize_wrapper();
        return 0;
    }

    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    route_context c{};
    c.remote = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.source = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.out = static_cast<unsigned char *>(nvshmem_align(16, kBytes));
    c.signal = static_cast<uint64_t *>(nvshmem_malloc(sizeof(uint64_t)));
    int *status = nullptr;
    const bool local_ok = scratch && c.remote && c.source && c.out && c.signal &&
                          cudaMalloc(&status, sizeof(*status)) == cudaSuccess;
    /* A partial symmetric allocation cannot be collectively freed. */
    if (!scratch || world_max(scratch, !local_ok) != 0) {
        fprintf(stderr, "PE %d: transport_specific_tma setup failed\n", mype);
        nvshmem_global_exit(1);
        return 1;
    }
    c.w = world{mype, npes, scratch, status};

    /* Run inside each direct-P2P clique of TEAM_SHARED; ranks without a mapped
     * ring neighbour only join the barriers. */
    const int shared_npes = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
    const int shared_mype = nvshmem_team_my_pe(NVSHMEM_TEAM_SHARED);
    c.peer = -1;
    c.previous = -1;
    if (shared_npes >= 2 && shared_mype >= 0) {
        c.peer = nvshmem_team_translate_pe(NVSHMEM_TEAM_SHARED, (shared_mype + 1) % shared_npes,
                                           NVSHMEM_TEAM_WORLD);
        c.previous = nvshmem_team_translate_pe(
            NVSHMEM_TEAM_SHARED, (shared_mype + shared_npes - 1) % shared_npes, NVSHMEM_TEAM_WORLD);
    }
    c.active = c.peer >= 0 && c.previous >= 0 && nvshmem_ptr(c.remote, c.peer) != nullptr &&
               nvshmem_ptr(c.remote, c.previous) != nullptr;

    int failures = 0;
    if (world_max(scratch, c.active) == 0) {
        if (mype == 0) {
            printf("transport_specific_tma: SKIP (no direct peer clique)\n");
        }
    } else {
        /* Probe the route's capability (sm_90+, NVSHMEM_TMA_POLICY, and shared
         * memory registration) without issuing work. */
        int probe = NVSHMEMX_SUCCESS;
        if (c.active) {
            CUDA_CHECK(cudaMemset(status, 0xff, sizeof(*status)));
            launch(probe_kernel, c.remote, c.peer, status);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            probe = read_status(status);
        }
        const bool probe_failed =
            probe != NVSHMEMX_SUCCESS && probe != NVSHMEMX_ERROR_NOT_SUPPORTED;
        if (probe_failed) {
            fprintf(stderr, "PE %d: transport_specific_tma capability probe returned %d\n", mype,
                    probe);
        }
        if (world_max(scratch, probe_failed) != 0) {
            failures = 1;
            report(mype, "transport_specific_tma", "Pointer / TMA capability probe", failures);
        } else if (world_max(scratch, probe == NVSHMEMX_ERROR_NOT_SUPPORTED) != 0) {
            failures = world_max(scratch, unavailable_cases(c));
            if (mype == 0) {
                printf(
                    "transport_specific_tma: SKIP Pointer / TMA (route unavailable: needs sm_90+, "
                    "NVSHMEM_TMA_POLICY=ENABLE, and registered shared memory)\n");
            }
            if (failures) {
                report(mype, "transport_specific_tma", "Pointer / TMA unavailable status",
                       failures);
            }
        } else {
            failures = world_max(scratch, tma_cases(c));
            report(mype, "transport_specific_tma", "Pointer / TMA", failures);
        }
    }

    nvshmem_barrier_all();
    nvshmem_free(c.remote);
    nvshmem_free(c.source);
    nvshmem_free(c.out);
    nvshmem_free(c.signal);
    nvshmem_free(scratch);
    CUDA_CHECK(cudaFree(status));
    finalize_wrapper();
    return failures ? 1 : 0;
}
