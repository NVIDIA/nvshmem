/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Compile contracts for the transport-policy headers. The driver compiles this
 * file once per case, selected with -DCONTRACT_CASE_<name>. */

#if defined(CONTRACT_CASE_legal)
#define NVSHMEMX_ENABLE_TRANSPORT_POLICY
#elif defined(CONTRACT_CASE_host_header_only)
#define NVSHMEMX_ENABLE_TRANSPORT_POLICY
#elif defined(CONTRACT_CASE_package_unavailable)
/* Emulate a package built without the feature while keeping its other
 * generated options; the header uses pragma once, so nvshmemx.h keeps this. */
#include "non_abi/c/nvshmem_build_options.h"
#undef NVSHMEMX_TRANSPORT_POLICY_AVAILABLE
#define NVSHMEMX_TRANSPORT_POLICY_AVAILABLE 0
#define NVSHMEMX_ENABLE_TRANSPORT_POLICY
#endif
/* CONTRACT_CASE_optin_required defines nothing. */

#if defined(CONTRACT_CASE_host_header_only)
/* The host-only header never exposes device declarations. */
#include "nvshmem_host.h"
#else
#include "nvshmemx.h"
#endif

__global__ void contract() {
    (void)nvshmemx::addressing::Pointer{};
    (void)nvshmemx::transport_policy::SM{};
    constexpr nvshmemx::threadgroup scope = nvshmemx::threadgroup::block;
    constexpr nvshmemx::memory_space::source source = nvshmemx::memory_space::source::generic;
    constexpr nvshmemx::memory_space::dest dest = nvshmemx::memory_space::dest::global;
    (void)scope;
    (void)source;
    (void)dest;
}
