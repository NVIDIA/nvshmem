/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Positive compile contract for the transport-specific API surface. The
 * kernels are never launched: compiling every legal route, scope, local memory space,
 * API, and element-type instantiation is the test. Negative combinations
 * live in transport_specific_compile_contract.cmake.
 */

#define NVSHMEMX_ENABLE_TRANSPORT_POLICY

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include "nvshmem.h"
#include "nvshmemx.h"

namespace {

namespace ts = nvshmemx;
using Source = ts::memory_space::source;
using Dest = ts::memory_space::dest;
constexpr Source kShared = Source::shared;
constexpr Source kGlobal = Source::global;
constexpr Source kGeneric = Source::generic;
constexpr auto kThread = ts::threadgroup::thread;
constexpr auto kWarp = ts::threadgroup::warp;
constexpr auto kBlock = ts::threadgroup::block;

static_assert(!std::is_convertible<Source, Dest>::value);
static_assert(!std::is_convertible<Dest, Source>::value);

/* The same local memory space spelled as a get destination. */
__host__ __device__ constexpr Dest as_dest(Source local) {
    return local == Source::shared ? Dest::shared
                                   : (local == Source::global ? Dest::global : Dest::generic);
}

/* Every data-moving API (byte-typed) for one local memory space. Put sources and get destinations
 * are local; the remote operand is always global. */
template <typename A, typename P, ts::threadgroup S, Source Local>
__device__ void byte_rma(void *remote_ptr, void *local_ptr, uint64_t *signal, int pe) {
    constexpr Dest kLocalDest = as_dest(Local);
    uint8_t *remote = static_cast<uint8_t *>(remote_ptr);
    uint8_t *local = static_cast<uint8_t *>(local_ptr);
    (void)nvshmemx_put<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, pe);
    (void)nvshmemx_put_nbi<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, pe);
    (void)nvshmemx_get<uint8_t, A, P, S, Source::global, kLocalDest>(local, remote, 16, pe);
    (void)nvshmemx_get_nbi<uint8_t, A, P, S, Source::global, kLocalDest>(local, remote, 16, pe);
    (void)nvshmemx_put_signal<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, signal, 1,
                                                                     NVSHMEM_SIGNAL_SET, pe);
    (void)nvshmemx_put_signal_nbi<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, signal,
                                                                         1, NVSHMEM_SIGNAL_ADD, pe);
}

template <typename A, typename P, ts::threadgroup S, Source... Locals>
__device__ void scope_all(void *remote, void *local, uint64_t *signal, int pe) {
    (byte_rma<A, P, S, Locals>(remote, local, signal, pe), ...);
    (void)nvshmemx_signal_op<A, P, S>(signal, 1, NVSHMEM_SIGNAL_SET, pe);
    (void)nvshmemx_quiet<A, P, S>();
    (void)nvshmemx_flush<A, P, S>();
    (void)nvshmemx_fence<A, P, S>();
}

template <typename A, typename P, ts::threadgroup S, Source Local>
__device__ void typed_all(void *remote, void *local, uint64_t *signal, int pe) {
#define TS_COMPILE_TYPED(NAME, TYPE)                                                              \
    {                                                                                             \
        TYPE *remote_typed = static_cast<TYPE *>(remote);                                         \
        TYPE *local_typed = static_cast<TYPE *>(local);                                           \
        (void)nvshmemx_put<TYPE, A, P, S, Local, Dest::global>(remote_typed, local_typed, 2, pe); \
        (void)nvshmemx_put_nbi<TYPE, A, P, S, Local, Dest::global>(remote_typed, local_typed, 2,  \
                                                                   pe);                           \
        (void)nvshmemx_get<TYPE, A, P, S, Source::global, as_dest(Local)>(local_typed,            \
                                                                          remote_typed, 2, pe);   \
        (void)nvshmemx_get_nbi<TYPE, A, P, S, Source::global, as_dest(Local)>(                    \
            local_typed, remote_typed, 2, pe);                                                    \
        (void)nvshmemx_put_signal<TYPE, A, P, S, Local, Dest::global>(                            \
            remote_typed, local_typed, 2, signal, 1, NVSHMEM_SIGNAL_SET, pe);                     \
        (void)nvshmemx_put_signal_nbi<TYPE, A, P, S, Local, Dest::global>(                        \
            remote_typed, local_typed, 2, signal, 1, NVSHMEM_SIGNAL_SET, pe);                     \
    }
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(TS_COMPILE_TYPED)
#undef TS_COMPILE_TYPED
}

template <typename A, typename P, Source... Locals>
__device__ void all_scopes(void *remote, void *local, uint64_t *signal, int pe) {
    scope_all<A, P, kThread, Locals...>(remote, local, signal, pe);
    scope_all<A, P, kWarp, Locals...>(remote, local, signal, pe);
    scope_all<A, P, kBlock, Locals...>(remote, local, signal, pe);
}

}  // namespace

__global__ void compile_pointer_sm(void *remote, void *local, uint64_t *signal, int pe) {
    using A = ts::addressing::Pointer;
    using P = ts::transport_policy::SM;
    all_scopes<A, P, kShared, kGlobal, kGeneric>(remote, local, signal, pe);
    typed_all<A, P, kBlock, kGlobal>(remote, local, signal, pe);
}

/* Pointer / TMA: every API at block scope; below block scope only the put
 * family from a shared or generic local source. */
template <typename A, typename P, ts::threadgroup S, Source Local>
__device__ void put_family(void *remote_ptr, void *local_ptr, uint64_t *signal, int pe) {
    uint8_t *remote = static_cast<uint8_t *>(remote_ptr);
    const uint8_t *local = static_cast<const uint8_t *>(local_ptr);
    (void)nvshmemx_put<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, pe);
    (void)nvshmemx_put_nbi<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, pe);
    (void)nvshmemx_put_signal<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, signal, 1,
                                                                     NVSHMEM_SIGNAL_SET, pe);
    (void)nvshmemx_put_signal_nbi<uint8_t, A, P, S, Local, Dest::global>(remote, local, 16, signal,
                                                                         1, NVSHMEM_SIGNAL_ADD, pe);
    (void)nvshmemx_signal_op<A, P, S>(signal, 1, NVSHMEM_SIGNAL_SET, pe);
    (void)nvshmemx_quiet<A, P, S>();
    (void)nvshmemx_flush<A, P, S>();
    (void)nvshmemx_fence<A, P, S>();
}

__global__ void compile_pointer_tma(void *remote, void *local, uint64_t *signal, int pe) {
    using A = ts::addressing::Pointer;
    using P = ts::transport_policy::TMA;
    scope_all<A, P, kBlock, kShared, kGlobal, kGeneric>(remote, local, signal, pe);
    typed_all<A, P, kBlock, kShared>(remote, local, signal, pe);
    put_family<A, P, kThread, kShared>(remote, local, signal, pe);
    put_family<A, P, kThread, kGeneric>(remote, local, signal, pe);
    put_family<A, P, kWarp, kShared>(remote, local, signal, pe);
    put_family<A, P, kWarp, kGeneric>(remote, local, signal, pe);
}

/* Auto / Network: every API at every scope. Explicit shared local values are
 * legal spellings that return INVALID_VALUE at run time. */
__global__ void compile_auto_network(void *remote, void *local, uint64_t *signal, int pe) {
    using A = ts::addressing::Auto;
    using P = ts::transport_policy::Network;
    all_scopes<A, P, kShared, kGlobal, kGeneric>(remote, local, signal, pe);
    typed_all<A, P, kBlock, kGlobal>(remote, local, signal, pe);
    typed_all<A, P, kThread, kGeneric>(remote, local, signal, pe);
    /* Pointer / Network is the same route. */
    all_scopes<ts::addressing::Pointer, P, kGlobal, kGeneric>(remote, local, signal, pe);
}

__global__ void compile_auto_auto(void *remote, void *local, uint64_t *signal, int pe) {
    using A = ts::addressing::Auto;
    using P = ts::transport_policy::Auto;
    all_scopes<A, P, kShared, kGlobal, kGeneric>(remote, local, signal, pe);
    typed_all<A, P, kBlock, kGeneric>(remote, local, signal, pe);
}

/* Pairs with an Auto side lower to an existing route with that route's rules:
 * Pointer / Auto, CFT / Auto, and Auto / SM to the portable dispatch, and
 * Auto / TMA to Pointer / TMA. */
__global__ void compile_auto_aliases(void *remote, void *local, uint64_t *signal, int pe) {
    all_scopes<ts::addressing::Pointer, ts::transport_policy::Auto, kGeneric>(remote, local, signal,
                                                                              pe);
    all_scopes<ts::addressing::CFT, ts::transport_policy::Auto, kGeneric>(remote, local, signal,
                                                                          pe);
    all_scopes<ts::addressing::Auto, ts::transport_policy::SM, kGeneric>(remote, local, signal, pe);
    using A = ts::addressing::Auto;
    using P = ts::transport_policy::TMA;
    scope_all<A, P, kBlock, kShared, kGlobal, kGeneric>(remote, local, signal, pe);
    put_family<A, P, kWarp, kShared>(remote, local, signal, pe);
}

int main() { return 0; }
