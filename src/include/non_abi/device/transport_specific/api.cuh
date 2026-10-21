/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_API_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_API_CUH_

#include "non_abi/device/transport_specific/automatic.cuh"
#include "non_abi/device/transport_specific/cft_tma.cuh"
#include "non_abi/device/transport_specific/network.cuh"
#include "non_abi/device/transport_specific/pointer_sm.cuh"
#include "non_abi/device/transport_specific/pointer_tma.cuh"
#include "non_abi/device/transport_specific/traits.cuh"
#include "non_abi/device/transport_specific/types.cuh"

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__clang_llvm_bitcode_lib__) || \
    defined(NVSHMEM_BUILD_LTOIR_LIBRARY)

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

/*
 * The public entry points below are thin: each maps its addressing/policy
 * pair to a route, applies the operation, scope, and memory-space checks,
 * runs the route's preflight on every participant, agrees on one status across
 * the group, and only then lets the route backend issue the work.
 */

enum class trace_op {
    put,
    put_nbi,
    get,
    get_nbi,
    put_signal,
    put_signal_nbi,
    signal_op,
    quiet,
    flush,
    fence,
};

#ifdef NVSHMEM_DEBUG
__device__ inline const char *trace_op_name(trace_op op) {
    switch (op) {
        case trace_op::put:
            return "put";
        case trace_op::put_nbi:
            return "put_nbi";
        case trace_op::get:
            return "get";
        case trace_op::get_nbi:
            return "get_nbi";
        case trace_op::put_signal:
            return "put_signal";
        case trace_op::put_signal_nbi:
            return "put_signal_nbi";
        case trace_op::signal_op:
            return "signal_op";
        case trace_op::quiet:
            return "quiet";
        case trace_op::flush:
            return "flush";
        default:
            return "fence";
    }
}

__device__ inline const char *trace_scope_name(threadgroup scope) {
    return scope == threadgroup::thread ? "thread" : scope == threadgroup::warp ? "warp" : "block";
}
#endif

/* The trace's target PE for operations that have none, such as ordering. */
inline constexpr int kNoTargetPe = -1;

/* Returns a status that was decided before any work was issued. Debug builds
 * report it once per threadgroup; release builds compile this to a plain
 * return. */
template <typename Route, threadgroup Scope, trace_op Op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int not_issued(int status, int pe) {
#ifdef NVSHMEM_DEBUG
    if (scope_leader<Scope>()) {
        printf(
            "NVSHMEM transport-policy: PE %d %s on %s at %s scope (target PE %d) failed before "
            "issuing work with status %d\n",
            nvshmemi_device_state_d.mype, trace_op_name(Op), route_backend<Route>::name(),
            trace_scope_name(Scope), pe, status);
    }
#else
    (void)pe;
#endif
    return status;
}

/* Per-participant RMA preparation: the debug annotation check and the route's
 * preflight. It contains no group synchronization; callers agree on its status
 * (with any other operand's) before issuing. */
template <typename Backend, threadgroup Scope, typename Local>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int prepare_rma(const void *remote, const void *local,
                                                         size_t bytes, int pe,
                                                         typename Backend::rma_plan &plan) {
    assert_local_annotation<Local>(local);
    return Backend::template preflight_rma<Scope, Local>(remote, local, bytes, pe, plan);
}

template <typename Addressing, typename TransportPolicy, threadgroup Scope, bool IsNbi,
          nvshmemx::memory_space::source Src, nvshmemx::memory_space::dest Dst>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int putmem(void *dest, const void *source, size_t bytes,
                                                    int pe) {
    using Route = route_for<Addressing, TransportPolicy>;
    using Backend = route_backend<Route>;
    using Local = source_local<Src>;
    constexpr trace_op kOp = IsNbi ? trace_op::put_nbi : trace_op::put;
    validate_scope<Scope>();
    validate_put_memory_spaces<Src, Dst>();

    typename Backend::rma_plan plan;
    const int status =
        agree<Scope>(prepare_rma<Backend, Scope, Local>(dest, source, bytes, pe, plan));
    if (status != NVSHMEMX_SUCCESS) {
        return not_issued<Route, Scope, kOp>(status, pe);
    }
    return Backend::template put<Scope, IsNbi, Local>(plan, source, bytes);
}

template <typename Addressing, typename TransportPolicy, threadgroup Scope, bool IsNbi,
          nvshmemx::memory_space::source Src, nvshmemx::memory_space::dest Dst>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int getmem(void *dest, const void *source, size_t bytes,
                                                    int pe) {
    using Route = route_for<Addressing, TransportPolicy>;
    using Backend = route_backend<Route>;
    using Local = dest_local<Dst>;
    constexpr trace_op kOp = IsNbi ? trace_op::get_nbi : trace_op::get;
    validate_scope<Scope>();
    validate_get_memory_spaces<Src, Dst>();

    typename Backend::rma_plan plan;
    const int status =
        agree<Scope>(prepare_rma<Backend, Scope, Local>(source, dest, bytes, pe, plan));
    if (status != NVSHMEMX_SUCCESS) {
        return not_issued<Route, Scope, kOp>(status, pe);
    }
    return Backend::template get<Scope, IsNbi, Local>(plan, dest, bytes);
}

template <typename Addressing, typename TransportPolicy, threadgroup Scope, bool IsNbi,
          nvshmemx::memory_space::source Src, nvshmemx::memory_space::dest Dst>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int putmem_signal(void *dest, const void *source,
                                                           size_t bytes, uint64_t *sig_addr,
                                                           uint64_t signal, int sig_op, int pe) {
    using Route = route_for<Addressing, TransportPolicy>;
    using Backend = route_backend<Route>;
    using Local = source_local<Src>;
    constexpr trace_op kOp = IsNbi ? trace_op::put_signal_nbi : trace_op::put_signal;
    validate_scope<Scope>();
    validate_put_memory_spaces<Src, Dst>();

    /* One agreement covers the payload and the signal before either is issued. */
    typename Backend::rma_plan plan;
    typename Backend::signal_plan sig;
    const int status =
        agree<Scope>(worse(prepare_rma<Backend, Scope, Local>(dest, source, bytes, pe, plan),
                           Backend::template preflight_signal<Scope>(sig_addr, sig_op, pe, sig)));
    if (status != NVSHMEMX_SUCCESS) {
        return not_issued<Route, Scope, kOp>(status, pe);
    }
    return Backend::template put_signal<Scope, IsNbi, Local>(plan, sig, source, bytes, signal,
                                                             sig_op);
}

template <typename Addressing, typename TransportPolicy, threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int signal_op(uint64_t *sig_addr, uint64_t signal,
                                                       int sig_op, int pe) {
    using Route = route_for<Addressing, TransportPolicy>;
    using Backend = route_backend<Route>;
    validate_scope<Scope>();

    typename Backend::signal_plan sig;
    const int status =
        agree<Scope>(Backend::template preflight_signal<Scope>(sig_addr, sig_op, pe, sig));
    if (status != NVSHMEMX_SUCCESS) {
        return not_issued<Route, Scope, trace_op::signal_op>(status, pe);
    }
    return Backend::template signal_op<Scope>(sig, signal, sig_op);
}

/* Ordering has no arguments, only the route's capability to preflight. That
 * capability is job-wide state, so it is group-uniform without an agreement,
 * and a route that always has it folds the check away. */
template <typename Addressing, typename TransportPolicy, threadgroup Scope, trace_op Op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int ordering() {
    using Route = route_for<Addressing, TransportPolicy>;
    using Backend = route_backend<Route>;
    validate_scope<Scope>();
    const int status = Backend::preflight_ordering();
    if (status != NVSHMEMX_SUCCESS) {
        return not_issued<Route, Scope, Op>(status, kNoTargetPe);
    }
    if constexpr (Op == trace_op::quiet) {
        return Backend::template quiet<Scope>();
    } else if constexpr (Op == trace_op::flush) {
        return Backend::template flush<Scope>();
    } else {
        return Backend::template fence<Scope>();
    }
}

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

/* nvcc's host pass parses kernel bodies that call these APIs, but the
 * nvshmemi::transport_policy internals only exist in the device pass. */
#ifdef __CUDA_ARCH__
#define NVSHMEMXI_TS_RETURN(...) return __VA_ARGS__
#else
#define NVSHMEMXI_TS_RETURN(...) return NVSHMEMX_ERROR_INTERNAL
#endif

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::generic,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::global>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_put(Type *dest, const Type *source,
                                                          size_t nelems, int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(nvshmemi::transport_policy::putmem<Addressing, TransportPolicy, Scope,
                                                           false, SrcMemorySpace, DestMemorySpace>(
        dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), pe));
}

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::generic,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::global>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_put_nbi(Type *dest, const Type *source,
                                                              size_t nelems, int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(nvshmemi::transport_policy::putmem<Addressing, TransportPolicy, Scope, true,
                                                           SrcMemorySpace, DestMemorySpace>(
        dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), pe));
}

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::global,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::generic>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_get(Type *dest, const Type *source,
                                                          size_t nelems, int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(nvshmemi::transport_policy::getmem<Addressing, TransportPolicy, Scope,
                                                           false, SrcMemorySpace, DestMemorySpace>(
        dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), pe));
}

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::global,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::generic>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_get_nbi(Type *dest, const Type *source,
                                                              size_t nelems, int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(nvshmemi::transport_policy::getmem<Addressing, TransportPolicy, Scope, true,
                                                           SrcMemorySpace, DestMemorySpace>(
        dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), pe));
}

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::generic,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::global>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_put_signal(Type *dest, const Type *source,
                                                                 size_t nelems, uint64_t *sig_addr,
                                                                 uint64_t signal, int sig_op,
                                                                 int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(
        nvshmemi::transport_policy::putmem_signal<Addressing, TransportPolicy, Scope, false,
                                                  SrcMemorySpace, DestMemorySpace>(
            dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), sig_addr,
            signal, sig_op, pe));
}

template <typename Type, typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope,
          nvshmemx::memory_space::source SrcMemorySpace = nvshmemx::memory_space::source::generic,
          nvshmemx::memory_space::dest DestMemorySpace = nvshmemx::memory_space::dest::global>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_put_signal_nbi(Type *dest, const Type *source,
                                                                     size_t nelems,
                                                                     uint64_t *sig_addr,
                                                                     uint64_t signal, int sig_op,
                                                                     int pe) {
    static_assert(nvshmemi::transport_policy::is_supported_rma_type<Type>::value,
                  "transport-specific RMA element type must be a standard NVSHMEM RMA type");
    NVSHMEMXI_TS_RETURN(
        nvshmemi::transport_policy::putmem_signal<Addressing, TransportPolicy, Scope, true,
                                                  SrcMemorySpace, DestMemorySpace>(
            dest, source, nvshmemi::transport_policy::bytes_for_elements<Type>(nelems), sig_addr,
            signal, sig_op, pe));
}

template <typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal,
                                                                int sig_op, int pe) {
    NVSHMEMXI_TS_RETURN(nvshmemi::transport_policy::signal_op<Addressing, TransportPolicy, Scope>(
        sig_addr, signal, sig_op, pe));
}

template <typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_quiet() {
    NVSHMEMXI_TS_RETURN(
        nvshmemi::transport_policy::ordering<Addressing, TransportPolicy, Scope,
                                             nvshmemi::transport_policy::trace_op::quiet>());
}

template <typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_flush() {
    NVSHMEMXI_TS_RETURN(
        nvshmemi::transport_policy::ordering<Addressing, TransportPolicy, Scope,
                                             nvshmemi::transport_policy::trace_op::flush>());
}

template <typename Addressing, typename TransportPolicy, nvshmemx::threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_fence() {
    NVSHMEMXI_TS_RETURN(
        nvshmemi::transport_policy::ordering<Addressing, TransportPolicy, Scope,
                                             nvshmemi::transport_policy::trace_op::fence>());
}

#undef NVSHMEMXI_TS_RETURN

#endif  // CUDA front-end visibility

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_API_CUH_
