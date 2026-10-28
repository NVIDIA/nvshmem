/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_

#include "non_abi/device/transport_specific/traits.cuh"

#ifdef __CUDA_ARCH__

#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
#define NVSHMEMI_TS_CFT_HANDLES 1
#else
#define NVSHMEMI_TS_CFT_HANDLES 0
#endif

namespace nvshmemi {
namespace transport_policy {

template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void validate_cft_tma_scope() {
    static_assert(Scope == threadgroup::block,
                  "CFT/TMA is block-scope only: all CTA threads participate convergently in the "
                  "logical-endpoint handle path");
}

template <>
struct route_backend<cft_tma_route> {
    struct rma_plan {
        int pe;
        bool shared_local;
    };
    struct signal_plan {
        uint64_t *sig_addr;
        int pe;
    };

    __device__ static const char *name() { return "CFT/TMA"; }

    template <threadgroup Scope, typename Local>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_rma(const void *remote,
                                                                      const void *local,
                                                                      size_t bytes, int pe,
                                                                      rma_plan &plan) {
        validate_cft_tma_scope<Scope>();
        plan = rma_plan{pe, false};
        int status = check_rma_arguments(remote, local, bytes, pe);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        status = check_handle_operands<Local>(remote, local, bytes, plan.shared_local);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        /* The handle path would serve a self-target with SM copies. */
        if (bytes != 0 && pe == nvshmemi_device_state_d.mype) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        if (bytes == 0) {
            return nvshmemi_tma_smem_registered() && nvshmemi_ld_and_check_valid_le_id(pe)
                       ? NVSHMEMX_SUCCESS
                       : NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        if (!nvshmemi_is_le_implemented<false>(pe, bytes, scope_traits<Scope>::internal, remote,
                                               local)) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        if (plan.shared_local && overlaps_registration(local, bytes)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        return NVSHMEMX_SUCCESS;
#else
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    /* The signal predicates describe the leader's handle slot. */
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
        if (pe == nvshmemi_device_state_d.mype) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        if (!scope_leader<Scope>()) {
            return NVSHMEMX_SUCCESS;
        }
        const bool supported = sig_op == NVSHMEM_SIGNAL_SET
                                   ? nvshmemi_is_le_implemented(pe)
                                   : nvshmemi_is_le_atomic_implemented(pe, sig_addr);
        return supported ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int preflight_ordering() {
#if NVSHMEMI_TS_CFT_HANDLES
        return nvshmemi_tma_smem_registered() ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
#else
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
#endif
    }

    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put(const rma_plan &plan, void *remote,
                                                            const void *source, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Source>;
        if constexpr (kLocal != NVSHMEMI_LOCAL_SPACE_UNKNOWN) {
            nvshmemi_handle_put<kScope, kLocal>(source, remote, bytes, plan.pe, !IsNbi);
        } else if (plan.shared_local) {
            nvshmemi_handle_put<kScope, NVSHMEMI_LOCAL_SPACE_SHARED>(source, remote, bytes, plan.pe,
                                                                     !IsNbi);
        } else {
            nvshmemi_handle_put<kScope, NVSHMEMI_LOCAL_SPACE_GLOBAL>(source, remote, bytes, plan.pe,
                                                                     !IsNbi);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
#else
        (void)plan;
        (void)source;
        return NVSHMEMX_ERROR_INTERNAL;
#endif
    }

    /* The handle get drains before it returns, so get_nbi is eager. */
    template <threadgroup Scope, bool IsNbi, typename Dest>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int get(const rma_plan &plan, void *dest,
                                                            const void *remote, size_t bytes) {
        if (bytes == 0) {
            return NVSHMEMX_SUCCESS;
        }
#if NVSHMEMI_TS_CFT_HANDLES
        constexpr threadgroup_t kScope = scope_traits<Scope>::internal;
        constexpr nvshmemi_local_space_t kLocal = stable_local_space<Dest>;
        if constexpr (kLocal != NVSHMEMI_LOCAL_SPACE_UNKNOWN) {
            nvshmemi_handle_get<kScope, kLocal>(remote, dest, bytes, plan.pe, !IsNbi);
        } else if (plan.shared_local) {
            nvshmemi_handle_get<kScope, NVSHMEMI_LOCAL_SPACE_SHARED>(remote, dest, bytes, plan.pe,
                                                                     !IsNbi);
        } else {
            nvshmemi_handle_get<kScope, NVSHMEMI_LOCAL_SPACE_GLOBAL>(remote, dest, bytes, plan.pe,
                                                                     !IsNbi);
        }
        scope_sync<Scope>();
        return NVSHMEMX_SUCCESS;
#else
        (void)plan;
        (void)dest;
        return NVSHMEMX_ERROR_INTERNAL;
#endif
    }

    /* Fabric operations are ordered only after completion, so the payload
     * completes before the signal and put_signal_nbi is eager. */
    template <threadgroup Scope, bool IsNbi, typename Source>
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE int put_signal(const rma_plan &plan,
                                                                   const signal_plan &sig,
                                                                   void *remote, const void *source,
                                                                   size_t bytes, uint64_t signal,
                                                                   int sig_op) {
        const int status = put<Scope, false, Source>(plan, remote, source, bytes);
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

    /* The only handle primitive is a drain, so flush is as strong as quiet. */

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
        return NVSHMEMX_ERROR_INTERNAL;
    }
#endif
};

}  // namespace transport_policy
}  // namespace nvshmemi

#undef NVSHMEMI_TS_CFT_HANDLES

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_CFT_TMA_CUH_
