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

/*
 * Whether the selected device network transport serves this job. A P2P-only
 * build compiles the nvshmemi_transfer_* calls against empty stubs, so the
 * route folds to unavailable there and never reaches them. Otherwise the job
 * uses a remote transport when its connectivity is weaker than direct LD/ST;
 * that transport then reaches every PE, P2P-mapped peers included. A job that
 * NVSHMEM runs over LD/ST alone leaves the network route unavailable rather
 * than falling back to P2P.
 */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool network_available() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return false;
#else
    return !nvshmemi_use_ldst_path();
#endif
}

/*
 * Auto / Network: the payload, its signal, and the standalone signal move
 * through the selected device network transport (IBGDA, GPUNetIO, or the
 * proxy) with the stable nvshmemi_transfer_* primitives on the default QP. The
 * route never probes or falls back to a P2P pointer, even for a mapped peer.
 * A missing network transport is NOT_SUPPORTED.
 *
 * The transfer path consumes only GPU global memory. An explicit shared local
 * operand is INVALID_VALUE without classification, a generic one is classified
 * with __isShared and is INVALID_VALUE when shared, and an explicit global one
 * is a trusted precondition. The selected NIC path may also need an lkey for
 * the local buffer: the caller must pass symmetric-heap or otherwise
 * registered memory where the transport requires it. That registration cannot
 * be preflighted, so it is a caller precondition rather than a status.
 */
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

    /* ---- Preflight: per participant, before any work is issued. Memory-space
     * and argument problems are reported in every build and job; the
     * capability check comes last. ---- */

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

    /* Ordering needs the transport too. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
        return network_available() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    /* ---- Issue: called only after the whole group agreed on preflight,
     * which is also the pre-issue group barrier. Every participant enters the
     * transfer primitive, which elects its own issuer; the closing barrier
     * publishes a blocking completion to the whole group. A zero-length
     * transfer moves nothing and is not posted. ---- */

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan, void *remote,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        void *local = const_cast<void *>(source);
        if constexpr (IsNbi) {
            nvshmemi_transfer_rma_nbi<kScope, NVSHMEMI_OP_PUT>(remote, local, bytes, plan.pe,
                                                               NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_rma<kScope, NVSHMEMI_OP_PUT>(remote, local, bytes, plan.pe,
                                                           NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            const void *remote, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        void *rptr = const_cast<void *>(remote);
        if constexpr (IsNbi) {
            nvshmemi_transfer_rma_nbi<kScope, NVSHMEMI_OP_GET>(rptr, dest, bytes, plan.pe,
                                                               NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_rma<kScope, NVSHMEMI_OP_GET>(rptr, dest, bytes, plan.pe,
                                                           NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* The transport's put-signal primitive orders the signal after the payload.
     * A zero-length put_signal is the signal alone. */
    template <threadgroup Scope, bool IsNbi, typename Source>
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
        if constexpr (IsNbi) {
            nvshmemi_transfer_put_signal_nbi<kScope>(remote, local, bytes, sig.sig_addr, signal, op,
                                                     plan.pe, NVSHMEMX_QP_DEFAULT);
        } else {
            nvshmemi_transfer_put_signal<kScope>(remote, local, bytes, sig.sig_addr, signal, op,
                                                 plan.pe, NVSHMEMX_QP_DEFAULT);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* The transport's remote atomic; never the P2P signal store or atomic. */
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

    /* ---- Completion: the transfer backend's own quiet and fence on the
     * default QP, never nvshmemi_quiet/flush/fence, which also drain TMA and
     * CFT work. The guarantee is a minimum: direct backends can share DCIs and
     * the proxy completes PE_ALL globally, so broader Network work may complete
     * too. quiet completes on every participant, with the transport's own
     * memory barrier; flush is the leader's quiet without it, which makes put
     * sources reusable; fence is the transport fence alone, since the route
     * issues no direct loads or stores that a system fence would order. ---- */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        nvshmemx_qp_handle_t qp = NVSHMEMX_QP_DEFAULT;
        nvshmemi_transfer_quiet<scope_traits<Scope>::internal>(true, NVSHMEMX_PE_ALL, &qp, 1);
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        scope_sync<Scope>();
        if (scope_leader<Scope>()) {
            nvshmemx_qp_handle_t qp = NVSHMEMX_QP_DEFAULT;
            nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(false, NVSHMEMX_PE_ALL, &qp, 1);
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
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_NETWORK_CUH_
