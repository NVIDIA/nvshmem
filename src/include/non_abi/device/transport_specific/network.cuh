/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_NETWORK_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_NETWORK_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

/* P2P-only builds stub out nvshmemi_transfer_*; LD/ST-only jobs have no network route. */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool network_available() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return false;
#else
    return !nvshmemi_use_ldst_path();
#endif
}

/* The NIC may need an lkey for the local buffer; that cannot be preflighted, so
 * registered memory is a caller precondition. */
template <>
struct route_backend<network_route> {
    struct rma_plan {
        int pe;
    };
    struct signal_plan {
        uint64_t *sig_addr;
        int pe;
    };

    __device__ static const char *name() { return "Auto/Network"; }

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        plan = rma_plan{pe};
        if constexpr (Local::kind == local_space::shared) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        } else {
            const int status = check_rma_arguments(remote, local, bytes, pe);
            if (status != NVSHMEMX_SUCCESS) {
                return status;
            }
            if constexpr (Local::kind == local_space::generic) {
                if (Local::is_shared(local)) {
                    return NVSHMEMX_ERROR_INVALID_VALUE;
                }
            }
            return network_available() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        plan = signal_plan{sig_addr, pe};
        const int status = check_signal_arguments(sig_addr, sig_op, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        return network_available() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
        return network_available() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan, void *remote,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        void *local = const_cast<void *>(source);
        if constexpr (Issue == issue_mode::nbi) {
            nvshmemi_transfer_rma_nbi<kScope, NVSHMEMI_OP_PUT>(remote, local, bytes, plan.pe,
                                                               NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_rma<kScope, NVSHMEMI_OP_PUT>(remote, local, bytes, plan.pe,
                                                           NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            const void *remote, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        void *rptr = const_cast<void *>(remote);
        if constexpr (Issue == issue_mode::nbi) {
            nvshmemi_transfer_rma_nbi<kScope, NVSHMEMI_OP_GET>(rptr, dest, bytes, plan.pe,
                                                               NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_rma<kScope, NVSHMEMI_OP_GET>(rptr, dest, bytes, plan.pe,
                                                           NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   void *remote, const void *source,
                                                                   size_t bytes, uint64_t signal,
                                                                   int sig_op) {
        if (bytes == 0) {
            return signal_op<Scope>(sig, signal, sig_op);
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        void *local = const_cast<void *>(source);
        const nvshmemi_amo_t op = static_cast<nvshmemi_amo_t>(sig_op);
        if constexpr (Issue == issue_mode::nbi) {
            nvshmemi_transfer_put_signal_nbi<kScope>(remote, local, bytes, sig.sig_addr, signal, op,
                                                     plan.pe, NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_put_signal<kScope>(remote, local, bytes, sig.sig_addr, signal, op,
                                                 plan.pe, NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(const signal_plan &plan,
                                                                  uint64_t signal, int sig_op) {
        if (scope_leader<Scope>()) {
            nvshmemi_transfer_amo_nonfetch<uint64_t>(plan.sig_addr, signal, plan.pe,
                                                     static_cast<nvshmemi_amo_t>(sig_op),
                                                     NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* Not nvshmemi_quiet/flush/fence, which would also drain TMA and CFT work. */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        drain<scope_traits<Scope>::internal>(drain_fence::system);
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        scope_sync<Scope>();
        if (scope_leader<Scope>()) {
            drain<NVSHMEMI_THREADGROUP_THREAD>(drain_fence::none);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        scope_sync<Scope>();
        if (scope_leader<Scope>()) {
            nvshmemx_qp_handle_t qp = NVSHMEMX_QP_DEFAULT;
            nvshmemi_transfer_fence<NVSHMEMI_THREADGROUP_THREAD>(NVSHMEMX_PE_ALL, &qp, 1);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

   private:
    enum class drain_fence { none, system };

    template <threadgroup_t Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void drain(drain_fence fence) {
        nvshmemx_qp_handle_t qp = NVSHMEMX_QP_DEFAULT;
        nvshmemi_transfer_quiet<Scope>(fence == drain_fence::system, NVSHMEMX_PE_ALL, &qp, 1);
    }
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_NETWORK_CUH_
