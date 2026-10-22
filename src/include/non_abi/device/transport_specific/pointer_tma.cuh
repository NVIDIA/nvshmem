/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_TMA_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_TMA_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

/*
 * Pointer / TMA: the peer's symmetric address is mapped with nvshmemi_ptr and
 * the payload moves with TMA bulk copies through it. The route needs an sm_90+
 * build, a CTA that registered its shared memory with nvshmemx_give_smem
 * (which also requires NVSHMEM_TMA_POLICY to enable TMA), and a mapped peer;
 * a missing prerequisite is NOT_SUPPORTED and never falls back to another path.
 *
 * A put from a shared local source is a per-thread bulk copy that completes
 * through the issuing thread's own bulk group, so it is legal at every scope
 * and safe with independent concurrent issuers. A get, and a put from a global
 * source, stage through or complete on the CTA's single mbarrier and staging
 * tile, so they are block-scope only: an explicit global source or any get
 * below block scope is a compile-time error, and a generic source that is not
 * in shared memory is INVALID_VALUE below block scope.
 */
template <>
struct route_backend<pointer_tma_route> {
    struct rma_plan {
        void *peer;
        nvshmemi_tma_smem_registration_t registration;
        bool shared_local;
    };
    struct signal_plan {
        void *peer;
    };

    __device__ static const char *name() { return "Pointer/TMA"; }

    /* ---- Preflight: per participant, before any work is issued ---- */

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        static_assert(Scope == threadgroup::block || Local::is_source,
                      "Pointer/TMA get and get_nbi are block-scope only");
        static_assert(
            Scope == threadgroup::block || !Local::is_source || Local::kind != local_space::global,
            "Pointer/TMA put from a global source is block-scope only; below block "
            "scope the source must be shared");
        plan = rma_plan{nullptr, {0, 0, -1}, false};
        const int status = check_rma_arguments(remote, local, bytes, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
#if __CUDA_ARCH__ >= 900
        /* The registration lookup also reports NVSHMEM_TMA_POLICY=DISABLE. */
        plan.registration = nvshmemi_tma_get_smem_registration();
        plan.peer = pointer_peer(remote, pe);
        if (!plan.registration.is_valid() || plan.peer == nullptr) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }

        plan.shared_local = Local::is_shared(local);
        if constexpr (Scope != threadgroup::block) {
            if (!plan.shared_local) {
                return NVSHMEMX_ERROR_INVALID_VALUE;
            }
        }
        if (!nvshmemi_tma_is_16b_aligned(reinterpret_cast<uintptr_t>(local)) ||
            !nvshmemi_tma_is_16b_aligned(reinterpret_cast<uintptr_t>(plan.peer)) ||
            !nvshmemi_tma_is_16b_aligned(bytes)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        if (!plan.shared_local) {
            return staged_copy_ready(plan.registration);
        }

        const int shared_status =
            shared_copy_ready<Local::is_source>(plan.registration, local, bytes);
        if (shared_status == NVSHMEMX_SUCCESS) {
            /* Every participant publishes its earlier shared-memory writes to
             * the async proxy before the agreement barrier, as the TMA
             * shared-memory access pattern requires. */
            nvshmemi_tma_fence_proxy_async_shared_cta();
        }
        return shared_status;
#else
        (void)local;
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        return pointer_preflight_signal(sig_addr, sig_op, pe, plan.peer);
    }

    /* ---- Issue: called only after the whole group agreed on preflight,
     * which is also the pre-issue group barrier. The stable TMA primitives
     * elect the issuing thread and end with the scope's own barrier. ---- */

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        int status;
        if (is_shared<Source>(plan)) {
            status = nvshmemi_memcpy_tma_shared_global<kScope, !IsNbi>(plan.peer, source, bytes);
        } else if constexpr (Scope == threadgroup::block) {
            if constexpr (IsNbi) {
                status = nvshmemi_memcpy_tma_global_global_nbi<kScope>(plan.registration, plan.peer,
                                                                       source, bytes);
            } else {
                status = nvshmemi_memcpy_tma_global_global<kScope>(plan.registration, plan.peer,
                                                                   source, bytes);
            }
        } else {
            /* Preflight admits only shared sources below block scope. */
            status = -1;
        }
        return status == 0 ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
    }

    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            size_t bytes) {
        if constexpr (Scope != threadgroup::block) {
            /* Rejected at compile time by preflight_rma. */
            return NVSHMEMX_ERROR_INTERNAL;
        } else {
            if (bytes == 0) {
                return NVSHMEMX_SUCCESS;
            }
            constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
            int status;
            if (is_shared<Dest>(plan)) {
                if constexpr (IsNbi) {
                    deferred_get_issue(plan, dest, bytes);
                    return NVSHMEMX_SUCCESS;
                }
                status = nvshmemi_memcpy_tma_global_shared<kScope>(plan.registration, dest,
                                                                   plan.peer, bytes);
            } else if constexpr (IsNbi) {
                status = nvshmemi_memcpy_tma_global_global_nbi<kScope>(plan.registration, dest,
                                                                       plan.peer, bytes);
            } else {
                status = nvshmemi_memcpy_tma_global_global<kScope>(plan.registration, dest,
                                                                   plan.peer, bytes);
            }
            return status == 0 ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
        }
    }

    /* Blocking: the payload completes before the leader publishes the signal.
     * NBI: the payload stays in flight; every participant, the elected TMA
     * issuer among them, executes an async-proxy fence before the group
     * barrier that precedes the signal. Neither form drains the bulk group. */
    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   const void *source, size_t bytes,
                                                                   uint64_t signal, int sig_op) {
        const int status = put<Scope, IsNbi, Source>(plan, source, bytes);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        if constexpr (IsNbi) {
            fence_proxy_async_global();
            scope_sync<Scope>();
        }
        if (scope_leader<Scope>()) {
            if constexpr (!IsNbi) {
                __threadfence_system();
            }
            pointer_issue_signal(sig.peer, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(const signal_plan &plan,
                                                                  uint64_t signal, int sig_op) {
        if (scope_leader<Scope>()) {
            pointer_issue_signal(plan.peer, signal, sig_op);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

    /* ---- Completion: the hardware completion domain is the issuing thread's
     * bulk group, and the issuer was elected inside the copy primitive, so
     * every participant commits and waits on its own group (an empty group is
     * harmless). Block scope also waits for deferred shared-destination gets.
     * flush waits only for the bulk reads, which makes put sources reusable;
     * quiet and fence wait for the writes and add a system fence. The signal
     * path needs no TMA state, so ordering has no capability to preflight. ---- */

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        return complete<Scope, false>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        return complete<Scope, true>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        return complete<Scope, false>();
    }

   private:
    /* Whether the local operand takes the shared-memory path. An explicit
     * spelling folds at compile time; generic uses the preflight result. */
    template <typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE bool is_shared(const rma_plan &plan) {
        if constexpr (Local::kind == local_space::shared) {
            return true;
        } else if constexpr (Local::kind == local_space::global) {
            return false;
        } else {
            return plan.shared_local;
        }
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void fence_proxy_async_global() {
#if __CUDA_ARCH__ >= 900
        asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
    }

    template <threadgroup Scope, bool Flush>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int complete() {
        scope_sync<Scope>();
#if __CUDA_ARCH__ >= 900
        if constexpr (Scope == threadgroup::block) {
            if (scope_leader<Scope>()) {
                const nvshmemi_tma_smem_registration_t registration =
                    nvshmemi_tma_get_smem_registration();
                if (registration.is_valid()) {
                    deferred_get_wait(registration.base);
                }
            }
        }
#endif
        nvshmemi_tma_bulk_commit_group();
        if constexpr (Flush) {
            nvshmemi_tma_bulk_wait_group_read_0();
            nvshmemi_tma_fence_proxy_async_shared_cta();
        } else {
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
    }

#if __CUDA_ARCH__ >= 900
    /* Local shared operand. A put reads it with cp.async.bulk shared-to-global;
     * a get writes it with the global-to-shared copy, which also needs a
     * registration larger than the reserved barrier region. Neither may
     * overlap that region. */
    template <bool IsSource>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int shared_copy_ready(
        const nvshmemi_tma_smem_registration_t &registration, const void *local, size_t bytes) {
        constexpr size_t kReserve = static_cast<size_t>(NVSHMEMI_SMEM_DATA_REGION_OFFSET);
        if (!IsSource && registration.size <= kReserve) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        const uintptr_t start = reinterpret_cast<uintptr_t>(local);
        if (bytes > static_cast<size_t>(UINT32_MAX) ||
            (start < registration.base + kReserve && start + bytes > registration.base)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        return NVSHMEMX_SUCCESS;
    }

    /* Local global operand, block scope: the double-buffered staging copy
     * needs two non-empty tiles and two full warps. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int staged_copy_ready(
        const nvshmemi_tma_smem_registration_t &registration) {
        const unsigned int block_threads = blockDim.x * blockDim.y * blockDim.z;
        return nvshmemi_smem_data_buf_size(registration, 2) != 0 && block_threads >= 2 * warpSize
                   ? NVSHMEMX_SUCCESS
                   : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    /*
     * Deferred block-scope gets into shared memory. A global-to-shared bulk
     * copy completes only on an mbarrier, so these gets share one CTA-wide
     * barrier epoch in the registered barrier region: every get adds its bytes
     * to the pending transaction count, and the next block-scope TMA quiet,
     * flush, or fence performs the single arrival and waits for the phase. A
     * block-scope operation is issued convergently by the whole CTA, so the
     * state has one user at a time. nvshmemx_give_smem clears the state slot
     * when the CTA registers its shared memory, and the slot records which
     * grid opened the epoch.
     *
     * The epoch outlives the call that opened it, so its two slots must be
     * ones no other operation reinitializes meanwhile. The stable TMA copies
     * use slots 0-4. CFT handle copies also borrow TMA barrier slots: a
     * single-issuer copy uses its issuing warp's index, and a multi-warp copy
     * uses four slots per threadgroup. A block-scope copy stays in slots 0-3,
     * but the stable warp-scope handle path (the stable warp APIs, or
     * Auto/Auto at warp scope) uses the calling warp's index, up to
     * NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS / TMA_COPY_NUM_STAGES - 1 because the
     * handle resource check admits no further warp. The epoch therefore lives
     * in the last two slots, which only a stable warpgroup-scope handle copy
     * issued by the eighth warpgroup of a 1024-thread CTA can reach.
     */
    static constexpr int kDeferredGetBarrierSlot = NVSHMEMI_TMA_DEFERRED_GET_BARRIER_SLOT;
    static constexpr int kDeferredGetStateSlot = NVSHMEMI_TMA_DEFERRED_GET_STATE_SLOT;
    static_assert(kDeferredGetBarrierSlot >=
                      NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS / TMA_COPY_NUM_STAGES,
                  "deferred Pointer/TMA gets must not share a TMA barrier slot with a block- or "
                  "warp-scope stable CFT handle copy");
    static constexpr uint32_t kMaxPendingTx = (1u << 20) - 1;

    struct deferred_get_state {
        uint64_t owner;
        uint32_t pending_bytes;
    };
    static_assert(sizeof(deferred_get_state) <= 16,
                  "the deferred-get state must fit in the TMA barrier slot nvshmemx_give_smem "
                  "clears");

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE deferred_get_state *deferred_state(
        uintptr_t smem_base) {
        return reinterpret_cast<deferred_get_state *>(
            nvshmemi_tma_barrier_slot(smem_base, kDeferredGetStateSlot));
    }

    /* Called by one thread that is synchronized with the issuers. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void deferred_get_wait(uintptr_t smem_base) {
        deferred_get_state *state = deferred_state(smem_base);
        if (state->owner != nvshmemi_tma_registration_key()) {
            return;
        }
        uint64_t *mbar = nvshmemi_tma_barrier_slot(smem_base, kDeferredGetBarrierSlot);
        nvshmemi_tma_mbarrier_arrive_expect_tx(mbar, 0);
        nvshmemi_tma_mbarrier_try_wait(mbar, 0);
        __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(mbar));
        state->owner = 0;
        state->pending_bytes = 0;
    }

    /* The agreement barrier orders this issue after the previous one, which
     * may have been elected on another lane. */
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void deferred_get_issue(const rma_plan &plan,
                                                                            void *dest,
                                                                            size_t bytes) {
        if (!nvshmemi_tma_block_is_elected()) {
            return;
        }
        const uintptr_t smem_base = plan.registration.base;
        deferred_get_state *state = deferred_state(smem_base);
        uint64_t *mbar = nvshmemi_tma_barrier_slot(smem_base, kDeferredGetBarrierSlot);
        const uint64_t key = nvshmemi_tma_registration_key();
        const uint32_t tx = static_cast<uint32_t>(bytes);

        if (state->owner == key && tx > kMaxPendingTx - state->pending_bytes) {
            deferred_get_wait(smem_base);
        }
        if (state->owner != key) {
            nvshmemi_tma_mbarrier_init(mbar);
            state->owner = key;
            state->pending_bytes = 0;
        }
        /* Orders the barrier initialization and earlier generic writes to the
         * destination before the async-proxy copy. */
        nvshmemi_tma_fence_proxy_async_shared_cta();
        state->pending_bytes += tx;
        barrier_expect_tx(reinterpret_cast<__mbarrier_t *>(mbar), tx);
        nvshmemi_tma_bulk_global_to_shared(dest, plan.peer, tx, mbar);
    }
#else
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE void deferred_get_issue(const rma_plan &,
                                                                            void *, size_t) {}
#endif
};

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_POINTER_TMA_CUH_
