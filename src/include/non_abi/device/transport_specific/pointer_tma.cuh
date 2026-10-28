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

/* Gets and global-source puts use the CTA's single mbarrier and staging tile,
 * so they are block-scope only. */
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
        /* Also reports NVSHMEM_TMA_POLICY=DISABLE. */
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

        const int shared_status = shared_copy_ready<Local>(plan.registration, local, bytes);
        if (shared_status == NVSHMEMX_SUCCESS) {
            /* Publish earlier smem writes to the async proxy before the agreement barrier. */
            nvshmemi_tma_fence_proxy_async_shared_cta();
        }
        return shared_status;
#else
        (void)local;
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_signal(uint64_t *sig_addr,
                                                                         int sig_op, int pe,
                                                                         signal_plan &plan) {
        return pointer_preflight_signal(sig_addr, sig_op, pe, plan.peer);
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
        return NVSHMEMX_SUCCESS;
    }

    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan, void *,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        int status;
        if (is_shared<Source>(plan)) {
            status = nvshmemi_memcpy_tma_shared_global<kScope, Issue == issue_mode::blocking>(
                plan.peer, source, bytes);
        } else if constexpr (Scope == threadgroup::block) {
            if constexpr (Issue == issue_mode::nbi) {
                status = nvshmemi_memcpy_tma_global_global_nbi<kScope>(plan.registration, plan.peer,
                                                                       source, bytes);
            } else {
                status = nvshmemi_memcpy_tma_global_global<kScope>(plan.registration, plan.peer,
                                                                   source, bytes);
            }
        } else {
            /* Unreachable: preflight admits only shared sources here. */
            status = -1;
        }
        return status == 0 ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
    }

    template <threadgroup Scope, issue_mode Issue, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            const void *, size_t bytes) {
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
                if constexpr (Issue == issue_mode::nbi) {
                    deferred_get_issue(plan, dest, bytes);
                    return NVSHMEMX_SUCCESS;
                }
                status = nvshmemi_memcpy_tma_global_shared<kScope>(plan.registration, dest,
                                                                   plan.peer, bytes);
            } else if constexpr (Issue == issue_mode::nbi) {
                status = nvshmemi_memcpy_tma_global_global_nbi<kScope>(plan.registration, dest,
                                                                       plan.peer, bytes);
            } else {
                status = nvshmemi_memcpy_tma_global_global<kScope>(plan.registration, dest,
                                                                   plan.peer, bytes);
            }
            return status == 0 ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
        }
    }

    /* NBI leaves the payload in flight; the async-proxy fence orders it before the signal. */
    template <threadgroup Scope, issue_mode Issue, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   void *dest, const void *source,
                                                                   size_t bytes, uint64_t signal,
                                                                   int sig_op) {
        const int status = put<Scope, Issue, Source>(plan, dest, source, bytes);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        if constexpr (Issue == issue_mode::nbi) {
            fence_proxy_async_global();
            scope_sync<Scope>();
        }
        if (scope_leader<Scope>()) {
            if constexpr (Issue == issue_mode::blocking) {
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

    /* The copy primitive elected the issuer, so each participant waits on its own bulk group. */
    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int quiet() {
        return complete<Scope, bulk_wait::writes>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int flush() {
        return complete<Scope, bulk_wait::reads>();
    }

    template <threadgroup Scope>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int fence() {
        return complete<Scope, bulk_wait::writes>();
    }

   private:
    enum class bulk_wait { writes, reads };

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

    template <threadgroup Scope, bulk_wait Wait>
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
        if constexpr (Wait == bulk_wait::reads) {
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
    template <typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int shared_copy_ready(
        const nvshmemi_tma_smem_registration_t &registration, const void *local, size_t bytes) {
        constexpr size_t kReserve = static_cast<size_t>(NVSHMEMI_SMEM_DATA_REGION_OFFSET);
        if (!Local::is_source && registration.size <= kReserve) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        const uintptr_t start = reinterpret_cast<uintptr_t>(local);
        if (bytes > static_cast<size_t>(UINT32_MAX) ||
            (start < registration.base + kReserve && start + bytes > registration.base)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        return NVSHMEMX_SUCCESS;
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int staged_copy_ready(
        const nvshmemi_tma_smem_registration_t &registration) {
        const unsigned int block_threads = blockDim.x * blockDim.y * blockDim.z;
        return nvshmemi_smem_data_buf_size(registration, 2) != 0 && block_threads >= 2 * warpSize
                   ? NVSHMEMX_SUCCESS
                   : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    /* Deferred shared-destination gets share one CTA-wide mbarrier epoch, drained by
     * the next block-scope quiet/flush/fence. Its slots are beyond stable handle copies. */
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

    /* The previous issue may have been elected on another lane; agreement orders it. */
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
        /* Orders the mbarrier init and earlier destination writes before the async copy. */
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
