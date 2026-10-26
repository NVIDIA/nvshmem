/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Functional coverage for the strict CFT / TMA route. The route never falls
 * back to another data path, so once its prerequisites hold (a CFT-enabled
 * build on a capable GPU, NVSHMEM_ENABLE_LOGICAL_ENDPOINT=1 with
 * NVSHMEM_TMA_POLICY=ENABLE, registered CTA shared memory, and a valid logical
 * endpoint for the peer) every positive case must succeed. When they do not,
 * the route reports NOT_SUPPORTED and the executable is a clean skip. Each
 * kernel keeps its shared memory registered until its final completion call.
 *
 * Signalled cases launch a second CTA that receives the previous PE's payload
 * and signal concurrently and checks the payload as soon as the signal lands.
 * Registration is per CTA: that receiver neither registers shared memory nor
 * issues CFT work. Gets check their result in the kernel right after the call
 * that completes them. Signal ADD needs the fabric atomic capability; where it
 * is missing, ADD must report NOT_SUPPORTED without touching the signal. */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include "transport_specific_common.cuh"

using namespace ts_test;

namespace {

constexpr size_t kBytes = 4096;
using CFT = nvshmemx::addressing::CFT;
using TMA = nvshmemx::transport_policy::TMA;
using Pointer = nvshmemx::addressing::Pointer;

/* The mixed-slot kernel runs eight warps, so a stable warp-scope handle copy
 * from warps 5 and 6 borrows the TMA barrier slots the deferred Pointer / TMA
 * get used before it moved to the last two. */
constexpr int kMixedThreads = 256;

/* Dynamic shared memory: [0, kRegistered) may be given to NVSHMEM and the
 * user payload area follows it, outside the registration. */
constexpr int kRegistered = 65536;
constexpr size_t kSmem = static_cast<size_t>(kRegistered) + kBytes;

/* A size and an offset that break the CFT_HANDLE_TX_SIZE rules while staying
 * inside the buffers. */
constexpr size_t kUnalignedBytes = 24;
constexpr size_t kMisalignment = 8;
constexpr size_t kSmallBytes = 64;
constexpr int kInvalidSigOp = 7;

enum class boundary { quiet, flush, fence };

template <threadgroup Scope, boundary B>
__device__ int complete() {
    if constexpr (B == boundary::quiet) {
        return nvshmemx_quiet<CFT, TMA, Scope>();
    } else if constexpr (B == boundary::flush) {
        return nvshmemx_flush<CFT, TMA, Scope>();
    } else {
        return nvshmemx_fence<CFT, TMA, Scope>();
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

template <typename Kernel, typename... Args>
void launch_shape(int blocks, int threads, Kernel kernel, Args... args) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSmem)));
    kernel<<<blocks, threads, kSmem>>>(args...);
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, Args... args) {
    launch_shape(1, kThreads, kernel, args...);
}

__device__ bool block_payload_matches(const unsigned char *buffer, size_t bytes, int pe) {
    return !__syncthreads_or(!payload_matches(buffer, bytes, pe, 0, threadIdx.x, blockDim.x));
}

/* A zero-length block put from the payload area: it runs the route's
 * capability checks and issues nothing. */
__global__ void probe_kernel(unsigned char *remote, int peer, int *status) {
    unsigned char *payload = register_smem(kRegistered);
    const int result =
        nvshmemx_put<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(remote, payload, 0, peer);
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* Adding zero probes the fabric atomic capability without changing the word. */
__global__ void add_probe_kernel(uint64_t *signal, int peer, int *status) {
    register_smem(kRegistered);
    const int result = nvshmemx_signal_op<CFT, TMA, kBlock>(signal, 0, NVSHMEM_SIGNAL_ADD, peer);
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* ---- Block scope ---------------------------------------------------------- */

template <LocalSpace Local, bool InShared, bool Nbi>
__global__ void put_kernel(unsigned char *remote, unsigned char *source, size_t bytes, int peer,
                           int *status) {
    unsigned char *payload = register_smem(kRegistered);
    unsigned char *local = InShared ? payload : source;
    if (InShared) {
        copy_block(payload, source, bytes);
    }
    __syncthreads();

    int result;
    if constexpr (Nbi) {
        result = nvshmemx_put_nbi<uint8_t, CFT, TMA, kBlock, Local, kRemoteDest>(remote, local,
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
        result =
            nvshmemx_put<uint8_t, CFT, TMA, kBlock, Local, kRemoteDest>(remote, local, bytes, peer);
    }
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* NBI gets issue two transfers before one completion call. */
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
        result = nvshmemx_get_nbi<uint8_t, CFT, TMA, kBlock, kRemoteSource, kLocalDest>(
            local, remote_source, half, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = nvshmemx_get_nbi<uint8_t, CFT, TMA, kBlock, kRemoteSource, kLocalDest>(
                local + half, remote_source + half, bytes - half, peer);
        }
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, B>();
        }
    } else {
        result = nvshmemx_get<uint8_t, CFT, TMA, kBlock, kRemoteSource, kLocalDest>(
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
        result = nvshmemx_put_signal_nbi<uint8_t, CFT, TMA, kBlock, Local, kRemoteDest>(
            remote, local, bytes, signal, value, sig_op, peer);
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, boundary::quiet>();
        }
    } else {
        result = nvshmemx_put_signal<uint8_t, CFT, TMA, kBlock, Local, kRemoteDest>(
            remote, local, bytes, signal, value, sig_op, peer);
    }
    release_smem(kRegistered);
    report_sender(status, result);
}

__global__ void signal_op_kernel(uint64_t *signal, uint64_t value, int sig_op, int peer,
                                 uint64_t expected, int previous, int *status) {
    if (blockIdx.x == 1) {
        receive_and_check(signal, expected, nullptr, 0, previous, status);
        return;
    }
    register_smem(kRegistered);
    const int result = nvshmemx_signal_op<CFT, TMA, kBlock>(signal, value, sig_op, peer);
    release_smem(kRegistered);
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
    register_smem(kRegistered);
    int result = nvshmemx_put_nbi<uint8_t, CFT, TMA, kBlock, kGlobal, kRemoteDest>(remote, source,
                                                                                   bytes, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<kBlock, boundary::fence>();
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_signal_op<CFT, TMA, kBlock>(signal, value, NVSHMEM_SIGNAL_SET, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = complete<kBlock, boundary::quiet>();
    }
    release_smem(kRegistered);
    report_sender(status, result);
}

__global__ void typed_kernel(unsigned char *remote, unsigned char *source, unsigned char *out,
                             size_t bytes, int peer, int *status) {
    register_smem(kRegistered);
    const size_t nelems = bytes / sizeof(uint64_t);
    int result = nvshmemx_put<uint64_t, CFT, TMA, kBlock, kGlobal, kRemoteDest>(
        reinterpret_cast<uint64_t *>(remote), reinterpret_cast<const uint64_t *>(source), nelems,
        peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get<uint64_t, CFT, TMA, kBlock, kRemoteSource, local_dest(kGlobal)>(
            reinterpret_cast<uint64_t *>(out), reinterpret_cast<const uint64_t *>(source), nelems,
            peer);
    }
    if (result == NVSHMEMX_SUCCESS && !block_payload_matches(out, bytes, peer)) {
        result = kDataMismatch;
    }
    release_smem(kRegistered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* A zero-length transfer may name a null local operand and a misaligned
 * remote address: it succeeds once the route is available and modifies
 * nothing. */
__global__ void zero_length_kernel(unsigned char *remote, const unsigned char *remote_source,
                                   int peer, int *status) {
    constexpr auto kLocalDest = local_dest(kGeneric);
    register_smem(kRegistered);
    int result =
        nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGeneric, kRemoteDest>(remote, nullptr, 0, peer);
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get_nbi<uint8_t, CFT, TMA, kBlock, kRemoteSource, kLocalDest>(
            nullptr, remote_source, 0, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGeneric, kRemoteDest>(remote + 1, nullptr,
                                                                                0, peer);
    }
    if (result == NVSHMEMX_SUCCESS) {
        result = nvshmemx_get<uint8_t, CFT, TMA, kBlock, kRemoteSource, kLocalDest>(
            nullptr, remote_source + 1, 0, peer);
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

/* ---- Mixed routes ------------------------------------------------------- */

/* A stable warp-scope put reaches the CFT handle path for a mapped peer only
 * in a build that prioritizes logical endpoints; elsewhere it would take the
 * P2P path, so the kernel leaves it out. */
#ifdef NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT
constexpr bool kStableWarpCft = true;
#else
constexpr bool kStableWarpCft = false;
#endif

/*
 * A Pointer / TMA block get_nbi into shared memory stays pending on its
 * deferred-get barrier while the CTA issues a block-scope CFT / TMA put from a
 * global source and, where the build routes it through the handle path, every
 * warp issues a stable warp-scope put. A single-issuer handle copy from a
 * global source borrows the TMA barrier slot of its warp index, so warps 5
 * and 6 reach the slots the deferred get used before; the get must survive
 * and complete at the Pointer / TMA quiet. Reports only failures; the caller
 * zeroes *status.
 */
__global__ void mixed_kernel(unsigned char *remote, const unsigned char *source, unsigned char *out,
                             size_t bytes, int peer, int *status) {
    constexpr auto kSharedDest = local_dest(kShared);
    unsigned char *payload = register_smem(kRegistered);
    fill_block(payload, bytes, 0);
    __syncthreads();

    const int get_result =
        nvshmemx_get_nbi<uint8_t, Pointer, TMA, kBlock, kRemoteSource, kSharedDest>(payload, source,
                                                                                    bytes, peer);
    int result = get_result;
    if (get_result == NVSHMEMX_SUCCESS) {
        const size_t cft_bytes = kStableWarpCft ? bytes / 2 : bytes;
        result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGlobal, kRemoteDest>(remote, source,
                                                                               cft_bytes, peer);
        if (kStableWarpCft && result == NVSHMEMX_SUCCESS) {
            const size_t slice = (bytes - cft_bytes) / (blockDim.x / warpSize);
            const size_t offset = cft_bytes + (threadIdx.x / warpSize) * slice;
            nvshmemx_putmem_warp(remote + offset, source + offset, slice, peer);
            __syncthreads();
        }
        if (result == NVSHMEMX_SUCCESS) {
            result = complete<kBlock, boundary::quiet>();
        }
    }
    /* get_result is block-uniform: the whole CTA completes its pending get. */
    if (get_result == NVSHMEMX_SUCCESS) {
        const int drained = nvshmemx_quiet<Pointer, TMA, kBlock>();
        if (result == NVSHMEMX_SUCCESS) {
            result = drained;
        }
    }
    if (result == NVSHMEMX_SUCCESS) {
        if (!block_payload_matches(payload, bytes, peer)) {
            result = kDataMismatch;
        } else {
            copy_block(out, payload, bytes);
        }
    }
    if (result != NVSHMEMX_SUCCESS) {
        atomicExch(status, result);
    }
    release_smem(kRegistered);
}

/* ---- Rejected calls ------------------------------------------------------- */

/* Each case is rejected by preflight before any work is issued. */
enum class error_case {
    invalid_pe,                /* INVALID_VALUE */
    unaligned_size_put,        /* INVALID_VALUE: size not a CFT_HANDLE_TX_SIZE multiple */
    unaligned_size_get,        /* INVALID_VALUE: same, get */
    misaligned_remote_put,     /* INVALID_VALUE: remote heap offset not 16-byte aligned */
    misaligned_local_get_nbi,  /* INVALID_VALUE: local not 16-byte aligned */
    generic_local_put,         /* INVALID_VALUE: generic source in neither shared nor global */
    registered_region_put,     /* INVALID_VALUE: shared source inside the registration */
    registered_region_get,     /* INVALID_VALUE: shared dest inside the registration */
    unaligned_size_put_signal, /* INVALID_VALUE: size, with a signal */
    invalid_sig_op,            /* INVALID_VALUE: opcode */
    add_misaligned_signal,     /* INVALID_VALUE: ADD word not 16-byte aligned in the heap */
    unregistered_put,          /* NOT_SUPPORTED: no registered shared memory */
    unregistered_signal_op,    /* NOT_SUPPORTED: no registered shared memory */
    unregistered_quiet,        /* NOT_SUPPORTED: no registered shared memory */
    barriers_only_put,         /* NOT_SUPPORTED: no staging tile */
    add_unavailable,           /* NOT_SUPPORTED where fabric atomics are missing */
};

__host__ __device__ constexpr int error_registration(error_case which) {
    return which == error_case::unregistered_put || which == error_case::unregistered_signal_op ||
                   which == error_case::unregistered_quiet
               ? 0
           : which == error_case::barriers_only_put ? NVSHMEMI_SMEM_DATA_REGION_OFFSET
                                                    : kRegistered;
}

__global__ void error_kernel(error_case which, unsigned char *remote, const unsigned char *source,
                             unsigned char *out, uint64_t *signal, int peer, int npes,
                             int *status) {
    const int registered = error_registration(which);
    unsigned char *payload = register_smem(registered);
    copy_block(payload, source, kBytes);
    __syncthreads();
    /* Registered shared memory past the barrier region, NVSHMEM's staging. */
    unsigned char *staging = smem_base() + NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    __align__(16) unsigned char private_buffer[kSmallBytes];

    int result = NVSHMEMX_SUCCESS;
    switch (which) {
        case error_case::invalid_pe:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(remote, payload,
                                                                                   kBytes, npes);
            break;
        case error_case::unaligned_size_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kUnalignedBytes, peer);
            break;
        case error_case::unaligned_size_get:
            result = nvshmemx_get<uint8_t, CFT, TMA, kBlock, kRemoteSource, local_dest(kGlobal)>(
                out, source, kUnalignedBytes, peer);
            break;
        case error_case::misaligned_remote_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGlobal, kRemoteDest>(
                remote + kMisalignment, source, kSmallBytes, peer);
            break;
        case error_case::misaligned_local_get_nbi:
            result =
                nvshmemx_get_nbi<uint8_t, CFT, TMA, kBlock, kRemoteSource, local_dest(kGlobal)>(
                    out + kMisalignment, source, kSmallBytes, peer);
            break;
        case error_case::generic_local_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGeneric, kRemoteDest>(
                remote, private_buffer, kSmallBytes, peer);
            break;
        case error_case::registered_region_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(
                remote, staging, kSmallBytes, peer);
            break;
        case error_case::registered_region_get:
            result = nvshmemx_get<uint8_t, CFT, TMA, kBlock, kRemoteSource, local_dest(kGeneric)>(
                staging, source, kSmallBytes, peer);
            break;
        case error_case::unaligned_size_put_signal:
            result = nvshmemx_put_signal<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kUnalignedBytes, signal, 1, NVSHMEM_SIGNAL_SET, peer);
            break;
        case error_case::invalid_sig_op:
            result = nvshmemx_signal_op<CFT, TMA, kBlock>(signal, 1, kInvalidSigOp, peer);
            break;
        case error_case::add_misaligned_signal:
            result = nvshmemx_put_signal_nbi<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kBytes, signal + 1, 1, NVSHMEM_SIGNAL_ADD, peer);
            break;
        case error_case::unregistered_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(remote, payload,
                                                                                   kBytes, peer);
            break;
        case error_case::unregistered_signal_op:
            result = nvshmemx_signal_op<CFT, TMA, kBlock>(signal, 1, NVSHMEM_SIGNAL_SET, peer);
            break;
        case error_case::unregistered_quiet:
            result = complete<kBlock, boundary::quiet>();
            break;
        case error_case::barriers_only_put:
            result = nvshmemx_put<uint8_t, CFT, TMA, kBlock, kGlobal, kRemoteDest>(remote, source,
                                                                                   kBytes, peer);
            break;
        case error_case::add_unavailable:
            result = nvshmemx_put_signal<uint8_t, CFT, TMA, kBlock, kShared, kRemoteDest>(
                remote, payload, kBytes, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            break;
    }
    release_smem(registered);
    if (threadIdx.x == 0) {
        *status = result;
    }
}

/* ---- Host-side cases ------------------------------------------------------ */

struct route_context {
    world w;
    bool active;
    bool add_supported;
    bool pointer_mapped;
    int peer;
    int previous;
    unsigned char *remote; /* symmetric put destination */
    unsigned char *source; /* symmetric put source and peer get source */
    unsigned char *out;    /* symmetric get destination */
    uint64_t *signal;      /* symmetric, two 16-byte-aligned words */
};

template <LocalSpace Local, bool InShared, bool Nbi>
int put_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
        },
        [&] {
            launch(put_kernel<Local, InShared, Nbi>, c.remote, c.source, kBytes, c.peer,
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
            launch_shape(2, kThreads, put_signal_kernel<Local, InShared, Nbi>, c.remote,
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
            launch_shape(2, kThreads, signal_op_kernel, c.signal, value, sig_op, c.peer, expected,
                         c.previous, c.w.status);
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
            launch_shape(2, kThreads, fence_kernel, c.remote,
                         static_cast<const unsigned char *>(c.source), kBytes, c.signal,
                         kSignalBase + c.w.mype, c.peer, expected, c.previous, c.w.status);
        },
        [&] {
            return check_signal_and_wait(what, c.signal, expected, c.w.status) +
                   check_payload(what, c.remote, kBytes, c.previous);
        });
}

int typed_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            CUDA_CHECK(cudaMemset(c.out, 0, kBytes));
        },
        [&] { launch(typed_kernel, c.remote, c.source, c.out, kBytes, c.peer, c.w.status); },
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

int mixed_case(const route_context &c, const char *what) {
    return run_case(
        c.w, c.active && c.pointer_mapped, what, NVSHMEMX_SUCCESS,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, 0, kBytes));
            CUDA_CHECK(cudaMemset(c.out, 0, kBytes));
        },
        [&] {
            CUDA_CHECK(cudaMemset(c.w.status, 0, sizeof(*c.w.status)));
            launch_shape(1, kMixedThreads, mixed_kernel, c.remote,
                         static_cast<const unsigned char *>(c.source), c.out, kBytes, c.peer,
                         c.w.status);
        },
        [&] {
            return check_payload(what, c.remote, kBytes, c.previous) +
                   check_payload(what, c.out, kBytes, c.peer);
        });
}

/* A rejected call must not issue work: every PE's destination, get buffer,
 * and signal words keep their sentinels. */
int error_case_run(const route_context &c, const char *what, error_case which, int expected) {
    return run_case(
        c.w, c.active, what, expected,
        [&] {
            fill_payload<<<1, kThreads>>>(c.source, kBytes, c.w.mype);
            CUDA_CHECK(cudaMemset(c.remote, kSentinelByte, kBytes));
            CUDA_CHECK(cudaMemset(c.out, kSentinelByte, kBytes));
            set_signal(c.signal, kSentinelSignal);
            set_signal(c.signal + 1, kSentinelSignal);
        },
        [&] {
            launch(error_kernel, which, c.remote, static_cast<const unsigned char *>(c.source),
                   c.out, c.signal, c.peer, c.w.npes, c.w.status);
        },
        [&] {
            return check_fill(what, c.remote, kBytes, kSentinelByte) +
                   check_fill(what, c.out, kBytes, kSentinelByte) +
                   check_signal(what, c.signal, kSentinelSignal) +
                   check_signal(what, c.signal + 1, kSentinelSignal);
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
        {error_case::invalid_pe, "CFT put invalid PE", kInvalid},
        {error_case::unaligned_size_put, "CFT put size not a 16-byte multiple", kInvalid},
        {error_case::unaligned_size_get, "CFT get size not a 16-byte multiple", kInvalid},
        {error_case::misaligned_remote_put, "CFT put misaligned remote", kInvalid},
        {error_case::misaligned_local_get_nbi, "CFT get_nbi misaligned local", kInvalid},
        {error_case::generic_local_put, "CFT put generic(local memory) source", kInvalid},
        {error_case::registered_region_put, "CFT put from the registered region", kInvalid},
        {error_case::registered_region_get, "CFT get into the registered region", kInvalid},
        {error_case::unaligned_size_put_signal, "CFT put_signal size not a 16-byte multiple",
         kInvalid},
        {error_case::invalid_sig_op, "CFT signal_op invalid opcode", kInvalid},
        {error_case::add_misaligned_signal, "CFT put_signal_nbi ADD to an 8-byte-aligned word",
         kInvalid},
        {error_case::unregistered_put, "CFT put without registered shared memory", kNotSupported},
        {error_case::unregistered_signal_op, "CFT signal_op without registered shared memory",
         kNotSupported},
        {error_case::unregistered_quiet, "CFT quiet without registered shared memory",
         kNotSupported},
        {error_case::barriers_only_put, "CFT put with a barrier-only registration", kNotSupported},
    };
    int failures = 0;
    for (const auto &e : cases) {
        failures += error_case_run(c, e.what, e.which, e.expected);
    }
    if (!c.add_supported) {
        failures += error_case_run(c, "CFT put_signal ADD without fabric atomics",
                                   error_case::add_unavailable, kNotSupported);
    }
    return failures;
}

int cft_cases(const route_context &c) {
    int failures = 0;
    failures += put_case<kShared, true, false>(c, "CFT put Shared");
    failures += put_case<kGlobal, false, false>(c, "CFT put Global");
    failures += put_case<kGeneric, true, false>(c, "CFT put generic(shared)");
    failures += put_case<kGeneric, false, false>(c, "CFT put generic(global)");
    failures += put_case<kShared, true, true>(c, "CFT put_nbi Shared+flush reuse");
    failures += put_case<kGlobal, false, true>(c, "CFT put_nbi Global+flush reuse");
    failures += put_case<kGeneric, true, true>(c, "CFT put_nbi generic(shared)+flush reuse");
    failures += put_case<kGeneric, false, true>(c, "CFT put_nbi generic(global)+flush reuse");

    failures += get_case<kShared, true, false>(c, "CFT get Shared");
    failures += get_case<kGlobal, false, false>(c, "CFT get Global");
    failures += get_case<kGeneric, true, false>(c, "CFT get generic(shared)");
    failures += get_case<kGeneric, false, false>(c, "CFT get generic(global)");
    failures += get_case<kShared, true, true, boundary::quiet>(c, "CFT 2x get_nbi Shared+quiet");
    failures += get_case<kGlobal, false, true, boundary::flush>(c, "CFT 2x get_nbi Global+flush");
    failures +=
        get_case<kGeneric, false, true, boundary::fence>(c, "CFT 2x get_nbi generic(global)+fence");
    failures +=
        get_case<kGeneric, true, true, boundary::quiet>(c, "CFT 2x get_nbi generic(shared)+quiet");

    const int set = NVSHMEM_SIGNAL_SET;
    failures += put_signal_case<kShared, true, false>(c, "CFT put_signal Shared SET", set);
    failures += put_signal_case<kGlobal, false, false>(c, "CFT put_signal Global SET", set);
    failures +=
        put_signal_case<kGeneric, false, false>(c, "CFT put_signal generic(global) SET", set);
    failures += put_signal_case<kShared, true, true>(c, "CFT put_signal_nbi Shared SET", set);
    failures += put_signal_case<kGlobal, false, true>(c, "CFT put_signal_nbi Global SET", set);
    failures +=
        put_signal_case<kGeneric, true, true>(c, "CFT put_signal_nbi generic(shared) SET", set);
    failures += signal_op_case(c, "CFT signal_op SET", set);
    if (c.add_supported) {
        const int add = NVSHMEM_SIGNAL_ADD;
        failures += put_signal_case<kShared, true, false>(c, "CFT put_signal Shared ADD", add);
        failures += put_signal_case<kGlobal, false, true>(c, "CFT put_signal_nbi Global ADD", add);
        failures += signal_op_case(c, "CFT signal_op ADD", add);
    } else if (c.w.mype == 0) {
        printf("transport_specific_cft_tma: SKIP CFT signal ADD (no fabric atomic capability)\n");
    }
    failures += fence_case(c, "CFT put_nbi Global+fence+signal_op SET");
    failures += typed_case(c, "CFT typed uint64 put/get");
    failures += zero_length_case(c, "CFT zero-length null local");

    failures +=
        mixed_case(c, kStableWarpCft ? "Pointer/TMA deferred get_nbi pending across CFT/TMA and "
                                       "stable warp CFT puts"
                                     : "Pointer/TMA deferred get_nbi pending across a CFT/TMA put");

    failures += error_cases(c);
    return failures;
}

/* The route is unavailable on this job: an argument error still reports
 * INVALID_VALUE and a valid put NOT_SUPPORTED, never INTERNAL, and neither
 * touches any buffer. */
int unavailable_cases(const route_context &c) {
    return error_case_run(c, "CFT put invalid PE while the route is unavailable",
                          error_case::invalid_pe, NVSHMEMX_ERROR_INVALID_VALUE) +
           error_case_run(c, "CFT put size not a 16-byte multiple while the route is unavailable",
                          error_case::unaligned_size_put, NVSHMEMX_ERROR_INVALID_VALUE) +
           error_case_run(c, "CFT put while the route is unavailable", error_case::unregistered_put,
                          NVSHMEMX_ERROR_NOT_SUPPORTED) +
           error_case_run(c, "CFT quiet while the route is unavailable",
                          error_case::unregistered_quiet, NVSHMEMX_ERROR_NOT_SUPPORTED);
}

int probe(const route_context &c, bool add) {
    if (!c.active) {
        return NVSHMEMX_SUCCESS;
    }
    CUDA_CHECK(cudaMemset(c.w.status, 0xff, sizeof(*c.w.status)));
    if (add) {
        launch(add_probe_kernel, c.signal, c.peer, c.w.status);
    } else {
        launch(probe_kernel, c.remote, c.peer, c.w.status);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return read_status(c.w.status);
}

}  // namespace

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) {
            printf("transport_specific_cft_tma: SKIP (requires at least two PEs)\n");
        }
        finalize_wrapper();
        return 0;
    }

    constexpr size_t kAlignment = 16;
    int *scratch = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    route_context c{};
    c.remote = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.source = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.out = static_cast<unsigned char *>(nvshmem_align(kAlignment, kBytes));
    c.signal = static_cast<uint64_t *>(nvshmem_align(kAlignment, 2 * sizeof(uint64_t)));
    int *status = nullptr;
    const bool local_ok = scratch && c.remote && c.source && c.out && c.signal &&
                          cudaMalloc(&status, sizeof(*status)) == cudaSuccess;
    if (!scratch || world_max(scratch, !local_ok) != 0) {
        fprintf(stderr, "PE %d: transport_specific_cft_tma setup failed\n", mype);
        nvshmem_global_exit(1);
        return 1;
    }
    c.w = world{mype, npes, scratch, status};

    /* A ring over every PE: logical endpoints address peers across the
     * fabric, mapped or not. */
    c.peer = (mype + 1) % npes;
    c.previous = (mype + npes - 1) % npes;
    c.active = true;
    /* The mixed case also needs the Pointer / TMA route to the same peer. */
    c.pointer_mapped = world_max(scratch, nvshmem_ptr(c.remote, c.peer) == nullptr) == 0;

    int failures = 0;
    const int route = probe(c, false);
    const bool probe_failed = route != NVSHMEMX_SUCCESS && route != NVSHMEMX_ERROR_NOT_SUPPORTED;
    if (probe_failed) {
        fprintf(stderr, "PE %d: transport_specific_cft_tma capability probe returned %d\n", mype,
                route);
    }
    if (world_max(scratch, probe_failed) != 0) {
        failures = 1;
        report(mype, "transport_specific_cft_tma", "CFT / TMA capability probe", failures);
    } else if (world_max(scratch, route == NVSHMEMX_ERROR_NOT_SUPPORTED) != 0) {
        failures = world_max(scratch, unavailable_cases(c));
        if (mype == 0) {
            printf(
                "transport_specific_cft_tma: SKIP CFT / TMA (route unavailable: needs a "
                "CFT-enabled build on a capable GPU, NVSHMEM_ENABLE_LOGICAL_ENDPOINT=1, "
                "NVSHMEM_TMA_POLICY=ENABLE, and registered shared memory)\n");
        }
        if (failures) {
            report(mype, "transport_specific_cft_tma", "CFT / TMA unavailable status", failures);
        }
    } else {
        const int add = probe(c, true);
        const bool add_failed = add != NVSHMEMX_SUCCESS && add != NVSHMEMX_ERROR_NOT_SUPPORTED;
        if (add_failed) {
            fprintf(stderr, "PE %d: transport_specific_cft_tma ADD probe returned %d\n", mype, add);
        }
        if (world_max(scratch, add_failed) != 0) {
            failures = 1;
        } else {
            c.add_supported = world_max(scratch, add == NVSHMEMX_ERROR_NOT_SUPPORTED) == 0;
            failures = world_max(scratch, cft_cases(c));
        }
        report(mype, "transport_specific_cft_tma", "CFT / TMA", failures);
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
