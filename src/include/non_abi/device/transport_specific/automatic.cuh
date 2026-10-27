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
 * its route selection, scope rules, fallback, and ordering unchanged. A
 * generic local operand keeps the portable run-time classification; an
 * explicit shared or global one is passed to the dispatcher, which then
 * selects its TMA and handle paths without classifying.
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

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        plan = signal_plan{sig_addr, pe};
        return check_signal_arguments(sig_addr, sig_op, pe);
    }

    /* The portable ordering helpers serve every job. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
        return NVSHMEMX_SUCCESS;
    }

    /* ---- Issue: called only after the whole group agreed on preflight ---- */

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan,
                                                            const void *source, size_t bytes) {
        char *dest = static_cast<char *>(plan.remote);
        const char *src = static_cast<const char *>(source);
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Source>;
        if constexpr (IsNbi) {
            nvshmemi_put_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE, true, kLocal>(
                dest, src, bytes, plan.pe);
        } else {
            nvshmemi_put<char, kScope<Scope>, kLocal>(dest, src, bytes, plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            size_t bytes) {
        char *dst = static_cast<char *>(dest);
        const char *src = static_cast<const char *>(plan.remote);
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Dest>;
        if constexpr (IsNbi) {
            nvshmemi_get_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE, kLocal>(
                dst, src, bytes, plan.pe);
        } else {
            nvshmemi_get<char, kScope<Scope>, true, kLocal>(dst, src, bytes, plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   const void *source, size_t bytes,
                                                                   uint64_t signal, int sig_op) {
        nvshmemi_put_signal<char, kScope<Scope>, stable_local_space<Source>>(
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
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_AUTOMATIC_CUH_
