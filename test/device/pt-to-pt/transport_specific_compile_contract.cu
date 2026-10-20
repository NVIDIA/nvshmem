/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Compile contracts for the transport-policy headers. The driver compiles this
 * file once per case, selected with -DCONTRACT_CASE_<name>. memory_space_ptx
 * also takes -DLOCAL_SPACE=shared|global|generic and is compiled to PTX. */

#if defined(CONTRACT_CASE_package_unavailable)
/* Emulate a package built without the feature while keeping its other
 * generated options; the header uses pragma once, so nvshmemx.h keeps this. */
#include "non_abi/c/nvshmem_build_options.h"
#undef NVSHMEMX_TRANSPORT_POLICY_AVAILABLE
#define NVSHMEMX_TRANSPORT_POLICY_AVAILABLE 0
#endif

/* Every case except optin_required opts in. */
#if !defined(CONTRACT_CASE_optin_required)
#define NVSHMEMX_ENABLE_TRANSPORT_POLICY
#endif
#if defined(CONTRACT_CASE_host_header_only)
/* The host-only header never exposes device declarations, even with the opt-in. */
#include "nvshmem_host.h"
#else
#include "nvshmemx.h"
#endif

#if defined(CONTRACT_CASE_optin_required) || defined(CONTRACT_CASE_host_header_only)
/* The transport-policy declarations must not leak into these consumers. */
__global__ void names_absent(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_put<uint8_t, nvshmemx::addressing::Pointer, nvshmemx::transport_policy::SM,
                       nvshmemx::threadgroup::block>(destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_package_unavailable)
/* nvshmemx.h itself must reject the opt-in. */

#else
using Auto = nvshmemx::addressing::Auto;
using Pointer = nvshmemx::addressing::Pointer;
using AutoPolicy = nvshmemx::transport_policy::Auto;
using Network = nvshmemx::transport_policy::Network;
using SM = nvshmemx::transport_policy::SM;
using TMA = nvshmemx::transport_policy::TMA;
namespace space = nvshmemx::memory_space;
constexpr auto kBlock = nvshmemx::threadgroup::block;

#if defined(CONTRACT_CASE_legal)
/* Positive control: the harness flags are sufficient for a legal consumer. */
__global__ void legal(uint8_t *destination, const uint8_t *source, uint64_t *signal, int pe) {
    constexpr space::source kSource = space::source::generic;
    constexpr space::dest kDest = space::dest::global;
    (void)nvshmemx_put<uint8_t, Pointer, SM, kBlock, kSource, kDest>(destination, source, 16, pe);
    (void)nvshmemx_get<uint8_t, Pointer, SM, nvshmemx::threadgroup::warp>(destination, source, 16,
                                                                          pe);
    (void)nvshmemx_put_signal_nbi<uint8_t, Auto, AutoPolicy, nvshmemx::threadgroup::thread,
                                  space::source::global>(destination, source, 16, signal, 1,
                                                         NVSHMEM_SIGNAL_SET, pe);
    (void)nvshmemx_signal_op<Pointer, SM, kBlock>(signal, 1, NVSHMEM_SIGNAL_ADD, pe);
    (void)nvshmemx_quiet<Pointer, SM, kBlock>();
    (void)nvshmemx_fence<Auto, AutoPolicy, kBlock>();
}

#elif defined(CONTRACT_CASE_invalid_pair)
__global__ void invalid_pair(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_put<uint8_t, Pointer, Network, kBlock>(destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_tma_get_below_block)
/* TMA gets complete on the CTA-wide mbarrier, so they are block-scope only. */
__global__ void tma_get_below_block(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_get_nbi<uint8_t, Pointer, TMA, nvshmemx::threadgroup::warp>(destination, source,
                                                                               16, pe);
}

#elif defined(CONTRACT_CASE_tma_global_put_below_block)
/* A global source stages through the CTA-wide tile, so it is block-scope only. */
__global__ void tma_global_put_below_block(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_put<uint8_t, Pointer, TMA, nvshmemx::threadgroup::thread, space::source::global>(
        destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_remote_put_tag)
__global__ void remote_put_tag(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_put<uint8_t, Pointer, SM, kBlock, space::source::global, space::dest::shared>(
        destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_remote_get_tag)
__global__ void remote_get_tag(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_get<uint8_t, Pointer, SM, kBlock, space::source::generic, space::dest::global>(
        destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_source_category_tag)
__global__ void source_category_tag(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_put<uint8_t, Pointer, SM, kBlock, space::dest::generic, space::dest::global>(
        destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_destination_category_tag)
__global__ void destination_category_tag(uint8_t *destination, const uint8_t *source, int pe) {
    (void)nvshmemx_get<uint8_t, Pointer, SM, kBlock, space::source::global, space::source::generic>(
        destination, source, 16, pe);
}

#elif defined(CONTRACT_CASE_unsupported_type)
struct pair16 {
    uint64_t a, b;
};
__global__ void unsupported_type(pair16 *destination, const pair16 *source, int pe) {
    (void)nvshmemx_put<pair16, Pointer, SM, kBlock>(destination, source, 1, pe);
}

#elif defined(CONTRACT_CASE_cv_qualified_type)
__global__ void cv_qualified_type(const float *destination, const float *source, int pe) {
    (void)nvshmemx_get<const float, Pointer, SM, kBlock>(destination, source, 1, pe);
}

#elif defined(CONTRACT_CASE_memory_space_ptx)
/* Compiled to PTX once per local memory space (LOCAL_SPACE is shared, global,
 * or generic). The contract requires the shared-window test (isspacep.shared)
 * that __isShared lowers to in the generic build and forbids it otherwise.
 * Pointer / SM must never classify its local operand, so every RMA family and
 * scope of that route is instantiated. Pointer / TMA is instantiated for every
 * RMA family at block scope and, except from a global source, for the put
 * family below block scope, its only legal forms. The internal source_local / dest_local
 * classifiers, which later routes use to classify a generic operand, are instantiated as well: they
 * keep the generic build observable and must fold away for an explicit spelling. Auto / Auto
 * intentionally retains portable classification and is excluded. */
#ifndef LOCAL_SPACE
#error "LOCAL_SPACE must be shared, global, or generic"
#endif

constexpr auto kSource = space::source::LOCAL_SPACE;
constexpr auto kDest = space::dest::LOCAL_SPACE;
constexpr auto kRemoteDest = space::dest::global;
constexpr auto kRemoteSource = space::source::global;

template <typename A, typename P, nvshmemx::threadgroup Scope>
__device__ int route_rma(uint8_t *remote, uint8_t *local, uint64_t *signal, size_t bytes, int pe) {
    int status = nvshmemx_put<uint8_t, A, P, Scope, kSource, kRemoteDest>(remote, local, bytes, pe);
    status |=
        nvshmemx_put_nbi<uint8_t, A, P, Scope, kSource, kRemoteDest>(remote, local, bytes, pe);
    status |= nvshmemx_get<uint8_t, A, P, Scope, kRemoteSource, kDest>(local, remote, bytes, pe);
    status |=
        nvshmemx_get_nbi<uint8_t, A, P, Scope, kRemoteSource, kDest>(local, remote, bytes, pe);
    status |= nvshmemx_put_signal<uint8_t, A, P, Scope, kSource, kRemoteDest>(
        remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_SET, pe);
    status |= nvshmemx_put_signal_nbi<uint8_t, A, P, Scope, kSource, kRemoteDest>(
        remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_SET, pe);
    return status;
}

template <typename A, typename P, nvshmemx::threadgroup Scope>
__device__ int route_put(uint8_t *remote, uint8_t *local, uint64_t *signal, size_t bytes, int pe) {
    int status = nvshmemx_put<uint8_t, A, P, Scope, kSource, kRemoteDest>(remote, local, bytes, pe);
    status |=
        nvshmemx_put_nbi<uint8_t, A, P, Scope, kSource, kRemoteDest>(remote, local, bytes, pe);
    status |= nvshmemx_put_signal<uint8_t, A, P, Scope, kSource, kRemoteDest>(
        remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_SET, pe);
    status |= nvshmemx_put_signal_nbi<uint8_t, A, P, Scope, kSource, kRemoteDest>(
        remote, local, bytes, signal, 1, NVSHMEM_SIGNAL_SET, pe);
    return status;
}

template <nvshmemx::threadgroup Scope>
__global__ void memory_space_ptx(uint8_t *remote, uint8_t *param, uint64_t *signal, size_t bytes,
                                 int pe, int *status) {
    /* Hide the kernel-parameter origin, from which the compiler would infer a
     * global address and fold __isShared away. */
    uint8_t *local;
    asm("mov.b64 %0, %1;" : "=l"(local) : "l"(param));
    int result = route_rma<Pointer, SM, Scope>(remote, local, signal, bytes, pe);
    if constexpr (Scope == nvshmemx::threadgroup::block) {
        result += route_rma<Pointer, TMA, Scope>(remote, local, signal, bytes, pe);
    } else if constexpr (kSource != space::source::global) {
        result += route_put<Pointer, TMA, Scope>(remote, local, signal, bytes, pe);
    }
#ifdef __CUDA_ARCH__
    result += nvshmemi::transport_policy::source_local<kSource>::is_shared(local);
    result += nvshmemi::transport_policy::dest_local<kDest>::is_shared(local);
#endif
    *status = result;
}

template __global__ void memory_space_ptx<nvshmemx::threadgroup::thread>(uint8_t *, uint8_t *,
                                                                         uint64_t *, size_t, int,
                                                                         int *);
template __global__ void memory_space_ptx<nvshmemx::threadgroup::warp>(uint8_t *, uint8_t *,
                                                                       uint64_t *, size_t, int,
                                                                       int *);
template __global__ void memory_space_ptx<nvshmemx::threadgroup::block>(uint8_t *, uint8_t *,
                                                                        uint64_t *, size_t, int,
                                                                        int *);

#else
#error "select a case with -DCONTRACT_CASE_<name>"
#endif
#endif
