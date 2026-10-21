/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_SM_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_SM_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

/*
 * Pointer / SM: the peer's symmetric address is mapped with nvshmemi_ptr and
 * the participating threads copy through it with ordinary loads and stores.
 * A peer without a direct mapping is a missing capability for this explicit
 * route; it never falls back to another path. The data path is synchronous,
 * so the NBI spellings complete before they return.
 */
template <>
struct route_backend<pointer_sm_route> {
    struct rma_plan {
        void *peer;
    };
    struct signal_plan {
        void *peer;
    };

    __device__ static const char *name() { return "Pointer/SM"; }

    /* ---- Preflight: per participant, before any work is issued ---- */

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        plan.peer = nullptr;
        const int status = check_rma_arguments(remote, local, bytes, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        plan.peer = peer_pointer(remote, pe);
        return plan.peer != nullptr ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        plan.peer = nullptr;
        const int status = check_signal_arguments(sig_addr, sig_op, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        plan.peer = peer_pointer(sig_addr, pe);
        /* ADD is a system-scope atomic on the mapped peer. */
        const bool supported =
            plan.peer != nullptr &&
            (sig_op == NVSHMEM_SIGNAL_SET || nvshmemi_device_state_d.mapped_peer_atomics);
        return supported ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    /* ---- Issue: called only after the whole group agreed on preflight,
     * which is also the pre-issue group barrier ---- */

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan,
                                                            const void *source, size_t bytes) {
        nvshmemi_memcpy_threadgroup<scope_traits<Scope>::internal>(plan.peer, source, bytes);
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            size_t bytes) {
        nvshmemi_memcpy_threadgroup<scope_traits<Scope>::internal>(dest, plan.peer, bytes);
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   const void *source, size_t bytes,
                                                                   uint64_t signal, int sig_op) {
        put<Scope, IsNbi, Source>(plan, source, bytes);
        if (scope_leader<Scope>()) {
            __threadfence_system();
            issue_signal(sig.peer, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(const signal_plan &plan,
                                                                  uint64_t signal, int sig_op) {
        if (scope_leader<Scope>()) {
            issue_signal(plan.peer, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* ---- Completion: the copies are synchronous, so quiet and fence only
     * need system-scope ordering and flush only a group barrier ---- */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        return system_fence<Scope>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        return system_fence<Scope>();
    }

   private:
    /* The mapped peer address, or null. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void *peer_pointer(const void *remote, int pe) {
        return nvshmemi_ptr(remote, pe);
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void issue_signal(void *peer, uint64_t signal,
                                                                      int sig_op) {
        if (sig_op == NVSHMEM_SIGNAL_SET) {
            *reinterpret_cast<volatile uint64_t *>(peer) = signal;
        } else {
            atomicAdd_system(reinterpret_cast<unsigned long long *>(peer),
                             static_cast<unsigned long long>(signal));
        }
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int system_fence() {
        scope_sync<Scope>();
        if (scope_leader<Scope>()) {
            __threadfence_system();
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_SM_CUH_
