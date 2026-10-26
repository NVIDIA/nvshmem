/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

/* The handle data path exists only in a CFT-enabled build for a capable
 * architecture and CUDA version; elsewhere the route compiles to its argument
 * checks and NOT_SUPPORTED. */
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
#define NVSHMEMI_TS_CFT_HANDLES 1
#else
#define NVSHMEMI_TS_CFT_HANDLES 0
#endif

namespace nvshmemi {
namespace transport_policy {

/* The one scope rule of the route, checked by every preflight. */
template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void validate_cft_tma_scope() {
    static_assert(Scope == threadgroup::block,
                  "CFT/TMA is block-scope only: all CTA threads participate convergently in the "
                  "logical-endpoint handle path");
}

/*
 * CFT / TMA: the peer is addressed through its logical endpoint (a fabric
 * handle built from the peer's LE id and the symmetric-heap offset), and the
 * payload moves with the stable handle helpers nvshmemi_handle_put/get, which
 * stage local global memory through the CTA's registered shared memory with
 * TMA bulk copies and issue fabric transfers. The route needs a CFT-enabled
 * build on a capable GPU, NVSHMEM_ENABLE_LOGICAL_ENDPOINT=1 (which needs
 * NVSHMEM_TMA_POLICY=ENABLE), a CTA that registered its shared memory with
 * nvshmemx_give_smem, and a valid LE id for the peer; a missing prerequisite
 * is NOT_SUPPORTED and never falls back to another path. The capability is
 * the stable nvshmemi_is_le_implemented predicate the portable API uses.
 *
 * Every transfer moves a multiple of CFT_HANDLE_TX_SIZE bytes between a
 * remote address whose heap offset is CFT_HANDLE_TX_SIZE-aligned and a
 * 16-byte-aligned local operand. A shared local operand must lie outside the
 * registered region, which NVSHMEM uses for barriers and staging.
 *
 * Scope. Every operation is block-scope only, as in the portable API's CFT
 * use: the handle path stages through the CTA's registered shared memory and
 * its barriers, so all CTA threads participate convergently.
 *
 * Signals. SET uses the handle scalar put, which needs only an 8-byte-aligned
 * word. ADD uses the handle fabric atomic, which updates a 16-byte block: the
 * word's heap offset must be CFT_HANDLE_TX_SIZE-aligned, and the route needs
 * the stable atomic capability (nvshmemi_is_le_atomic_implemented), which a
 * build or CUDA version without fabric atomics lacks. Both forms complete
 * before they return.
 */
template <>
struct route_backend<cft_tma_route> {
    struct rma_plan {
        void *remote;
        int pe;
    };
    struct signal_plan {
        uint64_t *sig_addr;
        int pe;
    };

    __device__ static const char *name() { return "CFT/TMA"; }

    /* ---- Preflight: per participant, before any work is issued ---- */

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        validate_cft_tma_scope<Scope>();
        plan = rma_plan{const_cast<void *>(remote), pe};
        int status = check_rma_arguments(remote, local, bytes, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        bool shared_local = false;
        status = check_handle_operands<Local>(remote, local, bytes, shared_local);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        /* A zero-length transfer moves nothing, so it needs only the route:
         * registered shared memory and a valid logical endpoint for the peer. */
        if (bytes == 0) {
            return nvshmemi_tma_smem_registered() && nvshmemi_ld_and_check_valid_le_id(pe)
                       ? NVSHMEMX_SUCCESS
                       : NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        /* The size and alignment parts of the stable predicate passed above,
         * so a false result is a missing capability. */
        if (!nvshmemi_is_le_implemented<false>(pe, bytes, scope_traits<Scope>::internal, remote,
                                               local)) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        if (shared_local && overlaps_registration(local, bytes)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        return NVSHMEMX_SUCCESS;
#else
        (void)shared_local;
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    /* The leader issues the signal, and the stable signal predicates describe
     * the calling thread's handle slot and staging, so only the leader
     * evaluates them; agree() carries its result to the group. */
    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        validate_cft_tma_scope<Scope>();
        plan = signal_plan{sig_addr, pe};
        const int status = check_signal_arguments(sig_addr, sig_op, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        if (sig_op == NVSHMEM_SIGNAL_ADD &&
            !nvshmemi_is_addr_offset_aligned(sig_addr, CFT_HANDLE_TX_SIZE)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        if (!scope_leader<Scope>()) {
            return NVSHMEMX_SUCCESS;
        }
        const bool supported = sig_op == NVSHMEM_SIGNAL_SET
                                   ? nvshmemi_is_le_implemented(pe)
                                   : nvshmemi_is_le_atomic_implemented(pe, sig_addr);
        return supported ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    /* The handle completion family lives in the CTA's registered shared
     * memory, which is uniform across the CTA. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
#if NVSHMEMI_TS_CFT_HANDLES
        return nvshmemi_tma_smem_registered() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
#else
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    /* ---- Issue: called only after the whole group agreed on preflight,
     * which is also the pre-issue group barrier. ---- */

    /* Blocking: the payload is complete on return. NBI: the handle barriers
     * keep the payload's completion pending until the next drain (quiet,
     * flush, fence, or a blocking handle operation). Either
     * way the stable helper has finished reading the source, so it is
     * reusable on return. */
    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        nvshmemi_handle_put<scope_traits<Scope>::internal>(source, plan.remote, bytes, plan.pe,
                                                           !IsNbi);
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
#else
        (void)plan;
        (void)source;
        /* Preflight reports NOT_SUPPORTED in this build. */
        return NVSHMEMX_ERROR_INTERNAL;
#endif
    }

    /* The stable handle get drains its fabric transfers before it returns, so
     * get_nbi completes eagerly, like get. */
    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        nvshmemi_handle_get<scope_traits<Scope>::internal>(plan.remote, dest, bytes, plan.pe,
                                                           !IsNbi);
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
#else
        (void)plan;
        (void)dest;
        return NVSHMEMX_ERROR_INTERNAL;
#endif
    }

    /* A fabric operation is ordered after an earlier one only once that one
     * has completed, so both spellings complete the payload before the leader
     * publishes the signal; put_signal_nbi therefore completes eagerly. */
    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   const void *source, size_t bytes,
                                                                   uint64_t signal, int sig_op) {
        const int status = put<Scope, false, Source>(plan, source, bytes);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        if (scope_leader<Scope>()) {
            order_before_signal();
            issue_signal(sig, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(const signal_plan &plan,
                                                                  uint64_t signal, int sig_op) {
        if (scope_leader<Scope>()) {
            issue_signal(plan, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* ---- Completion: the CFT handle completion family. The CTA drains its
     * pending handle barriers, and nothing here touches TMA bulk groups, deferred
     * Pointer/TMA gets, or network work. Put sources are reusable as soon as
     * the put returns and the only handle primitive is a drain, so flush is
     * as strong as quiet. fence and quiet also order later fabric and generic
     * operations after the drained ones. ---- */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        return complete<Scope>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        return complete<Scope>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        return complete<Scope>();
    }

   private:
    /* The size and alignment rules of the handle path, and the local memory
     * space. A zero-length transfer may name a null local operand and is never
     * classified. A generic operand must be shared or global memory. */
    template <typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int check_handle_operands(const void *remote,
                                                                              const void *local,
                                                                              size_t bytes,
                                                                              bool &shared_local) {
        shared_local = false;
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        if (bytes % CFT_HANDLE_TX_SIZE != 0 ||
            !nvshmemi_is_addr_offset_aligned(remote, CFT_HANDLE_TX_SIZE) ||
            !nvshmemi_tma_is_16b_aligned(reinterpret_cast<uintptr_t>(local))) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        shared_local = Local::is_shared(local);
        if constexpr (Local::kind == local_space::generic) {
            if (!shared_local && !__isGlobal(local)) {
                return NVSHMEMX_ERROR_INVALID_VALUE;
            }
        }
        return NVSHMEMX_SUCCESS;
    }

#if NVSHMEMI_TS_CFT_HANDLES
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE bool overlaps_registration(const void *local,
                                                                               size_t bytes) {
        const nvshmemi_tma_smem_registration_t registration = nvshmemi_tma_get_smem_registration();
        const uintptr_t start = reinterpret_cast<uintptr_t>(local);
        return start < registration.base + registration.size && start + bytes > registration.base;
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void order_before_signal() {
        __threadfence_system();
        fence_proxy_fabric2fabric_alias();
        fence_proxy_generic2fabric_alias();
    }

    /* Preflight admitted SET with the scalar handle capability and ADD with
     * the atomic one, so the leader issues exactly what the hardware can do. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void issue_signal(const signal_plan &plan,
                                                                      uint64_t signal, int sig_op) {
        if (sig_op == NVSHMEM_SIGNAL_SET) {
            nvshmemi_handle_p<uint64_t>(plan.sig_addr, signal, plan.pe);
        } else {
            nvshmemi_handle_atomic_nonfetch<uint64_t, NVSHMEMI_AMO_SIGNAL_ADD>(plan.sig_addr,
                                                                               signal, plan.pe);
        }
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int complete() {
        validate_cft_tma_scope<Scope>();
        scope_sync<Scope>();
        nvshmemi_handle_quiet<scope_traits<Scope>::internal>();
        __threadfence_system();
        fence_proxy_fabric2fabric_alias();
        fence_proxy_generic2fabric_alias();
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }
#else
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void order_before_signal() {}

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void issue_signal(const signal_plan &, uint64_t,
                                                                      int) {}

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int complete() {
        validate_cft_tma_scope<Scope>();
        /* Preflight reports NOT_SUPPORTED in this build. */
        return NVSHMEMX_ERROR_INTERNAL;
    }
#endif
};

}  // namespace transport_policy
}  // namespace nvshmemi

#undef NVSHMEMI_TS_CFT_HANDLES

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_
