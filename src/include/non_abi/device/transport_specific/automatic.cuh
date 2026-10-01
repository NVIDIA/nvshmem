/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_AUTOMATIC_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_AUTOMATIC_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

/*
 * Auto / Auto: the compatibility route over the portable dispatcher, keeping
 * its route selection, fallback, MemorySpace classification, and ordering.
 * The CFT data-movement branch of that dispatcher is block-only, so thread and
 * warp scopes instantiate it with ALLOW_CFT = false: a reachable peer uses TMA
 * when this CTA registered shared memory and the copy qualifies, otherwise
 * LD/ST, and an unreachable peer uses the device transport.
 */
template <>
struct route_backend<automatic_route> {
    struct rma_plan {
        void *remote;
        int pe;
    };
    struct signal_plan {
        uint64_t *sig_addr;
        int pe;
    };

    __device__ static const char *name() { return "Auto/Auto"; }

    /* ---- Preflight: per participant, before any work is issued ---- */

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        plan = rma_plan{const_cast<void *>(remote), pe};
        return check_rma_arguments(remote, local, bytes, pe);
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        plan = signal_plan{sig_addr, pe};
        return check_signal_arguments(sig_addr, sig_op, pe);
    }

    /* ---- Issue: called only after the whole group agreed on preflight ---- */

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan,
                                                            const void *source, size_t bytes) {
        char *dest = static_cast<char *>(plan.remote);
        const char *src = static_cast<const char *>(source);
        if constexpr (IsNbi) {
            nvshmemi_put_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE, kAllowCft<Scope>>(
                dest, src, bytes, plan.pe);
        } else {
            nvshmemi_put<char, kScope<Scope>, kAllowCft<Scope>>(dest, src, bytes, plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            size_t bytes) {
        char *dst = static_cast<char *>(dest);
        const char *src = static_cast<const char *>(plan.remote);
        if constexpr (IsNbi) {
            nvshmemi_get_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE, kAllowCft<Scope>>(
                dst, src, bytes, plan.pe);
        } else {
            nvshmemi_get<char, kScope<Scope>, kAllowCft<Scope>>(dst, src, bytes, plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   const void *source, size_t bytes,
                                                                   uint64_t signal, int sig_op) {
        nvshmemi_put_signal<char, kScope<Scope>, kAllowCft<Scope>>(
            static_cast<char *>(plan.remote), static_cast<const char *>(source), bytes,
            sig.sig_addr, signal, sig_op, plan.pe, IsNbi);
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(const signal_plan &plan,
                                                                  uint64_t signal, int sig_op) {
        if (scope_leader<Scope>()) {
            nvshmemi_signal_op(plan.sig_addr, signal, sig_op, plan.pe);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* ---- Completion: the portable ordering semantics ---- */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        nvshmemi_quiet<kScope<Scope>>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        nvshmemi_flush<kScope<Scope>>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        nvshmemi_fence<kScope<Scope>>();
        return NVSHMEMX_SUCCESS;
    }

   private:
    template <threadgroup Scope>
    static constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
    template <threadgroup Scope>
    static constexpr bool kAllowCft = Scope == threadgroup::block;
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_AUTOMATIC_CUH_
