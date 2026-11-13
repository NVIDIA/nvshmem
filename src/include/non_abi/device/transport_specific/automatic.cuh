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

template <>
struct route_backend<automatic_route> {
    struct rma_plan {
        int pe;
    };
    struct signal_plan {
        uint64_t *sig_addr;
        int pe;
    };

    __device__ static const char *name() { return "Auto/Auto"; }

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        plan = rma_plan{pe};
        return check_rma_arguments(remote, local, bytes, pe);
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        plan = signal_plan{sig_addr, pe};
        return check_signal_arguments(sig_addr, sig_op, pe);
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan, void *remote,
                                                            const void *source, size_t bytes) {
        char *dest = static_cast<char *>(remote);
        const char *src = static_cast<const char *>(source);
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Source>;
        if constexpr (Issue == issue_mode::nbi) {
            nvshmemi_put_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE,
                             nvshmemi_tma_use::allowed, kLocal>(dest, src, bytes, plan.pe);
        } else {
            nvshmemi_put<char, kScope<Scope>, kLocal>(dest, src, bytes, plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            const void *remote, size_t bytes) {
        char *dst = static_cast<char *>(dest);
        const char *src = static_cast<const char *>(remote);
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Dest>;
        if constexpr (Issue == issue_mode::nbi) {
            nvshmemi_get_nbi<char, kScope<Scope>, NVSHMEMI_REGION_OPERATION_NONE, kLocal>(
                dst, src, bytes, plan.pe);
        } else {
            nvshmemi_get<char, kScope<Scope>, nvshmemi_tma_use::allowed, kLocal>(dst, src, bytes,
                                                                                 plan.pe);
        }
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   void *remote, const void *source,
                                                                   size_t bytes, uint64_t signal,
                                                                   int sig_op) {
        nvshmemi_put_signal<char, kScope<Scope>, stable_local_space<Source>>(
            static_cast<char *>(remote), static_cast<const char *>(source), bytes, sig.sig_addr,
            signal, sig_op, plan.pe, Issue == issue_mode::nbi);
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
