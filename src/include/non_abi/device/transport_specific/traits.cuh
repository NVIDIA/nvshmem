/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_TRAITS_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_TRAITS_CUH_

#if !defined __CUDACC_RTC__
#include <assert.h>
#endif

#include "c_api/nvshmem_api_macros.h"
#include "c_api/nvshmemx_status.h"
#include "device/nvshmem_device_macros.h"
#include "non_abi/device/common/nvshmemi_cccl_compat.h"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/transport_specific/types.cuh"

namespace nvshmemi {
namespace transport_policy {

/* Exactly the element types of the stable typed RMA API, cv-unqualified. The
 * public wrappers check it in both compilation passes. */
template <typename Type>
struct is_supported_rma_type {
#define NVSHMEMI_TS_IS_RMA_TYPE(NAME, TYPE) cuda::std::is_same<Type, TYPE>::value ||
    static constexpr bool value =
        NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TS_IS_RMA_TYPE) false;
#undef NVSHMEMI_TS_IS_RMA_TYPE
};

}  // namespace transport_policy
}  // namespace nvshmemi

#ifdef __CUDA_ARCH__

namespace nvshmemi {
namespace transport_policy {

using nvshmemx::threadgroup;

/* ---- Routes ------------------------------------------------------------ */

/* Canonical internal routes. Later stages add pointer_tma_route,
 * cft_ce_route, and network_route. */
struct pointer_sm_route {};
struct automatic_route {};

/* Maps a public addressing/transport-policy pair to its route. There is no
 * primary definition: an unsupported pair has no mapping. */
template <typename Addressing, typename TransportPolicy>
struct route_compatibility;

template <>
struct route_compatibility<nvshmemx::addressing::Pointer, nvshmemx::transport_policy::SM> {
    using route = pointer_sm_route;
};

template <>
struct route_compatibility<nvshmemx::addressing::Auto, nvshmemx::transport_policy::Auto> {
    using route = automatic_route;
};

/* Everything about one route: preflight, issue, and completion. Each route
 * specializes it in its own header. */
template <typename Route>
struct route_backend;

template <typename Addressing, typename TransportPolicy, typename = void>
struct is_supported_pair {
    static constexpr bool value = false;
};

template <typename Addressing, typename TransportPolicy>
struct is_supported_pair<
    Addressing, TransportPolicy,
    cuda::std::void_t<typename route_compatibility<Addressing, TransportPolicy>::route>> {
    static constexpr bool value = true;
};

template <typename Addressing, typename TransportPolicy,
          bool Supported = is_supported_pair<Addressing, TransportPolicy>::value>
struct route_selection {
    using route = typename route_compatibility<Addressing, TransportPolicy>::route;
};

template <typename Addressing, typename TransportPolicy>
struct route_selection<Addressing, TransportPolicy, false> {
    static_assert(is_supported_pair<Addressing, TransportPolicy>::value,
                  "unsupported addressing/transport-policy pair");
    /* Any complete route, so that the assertion above is the only diagnostic. */
    using route = automatic_route;
};

template <typename Addressing, typename TransportPolicy>
using route_for = typename route_selection<Addressing, TransportPolicy>::route;

/* ---- Scopes ------------------------------------------------------------ */

template <threadgroup Scope>
struct scope_traits {
    static constexpr bool supported = false;
    static constexpr threadgroup_t internal = NVSHMEMI_THREADGROUP_THREAD;
};

template <>
struct scope_traits<threadgroup::thread> {
    static constexpr bool supported = true;
    static constexpr threadgroup_t internal = NVSHMEMI_THREADGROUP_THREAD;
};

template <>
struct scope_traits<threadgroup::warp> {
    static constexpr bool supported = true;
    static constexpr threadgroup_t internal = NVSHMEMI_THREADGROUP_WARP;
};

template <>
struct scope_traits<threadgroup::block> {
    static constexpr bool supported = true;
    static constexpr threadgroup_t internal = NVSHMEMI_THREADGROUP_BLOCK;
};

template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void validate_scope() {
    static_assert(scope_traits<Scope>::supported, "unsupported transport-specific threadgroup");
}

template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void scope_sync() {
    nvshmemi_threadgroup_sync<scope_traits<Scope>::internal>();
}

template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool scope_leader() {
    return nvshmemi_thread_id_in_threadgroup<scope_traits<Scope>::internal>() == 0;
}

/*
 * Group-uniform preflight. Every participant passes its own preflight status;
 * all of them return the numerically largest one, so the most severe category
 * (INTERNAL > NOT_SUPPORTED > INVALID_VALUE > SUCCESS) wins deterministically.
 * The agreement is also a group barrier with the ordering of
 * nvshmemi_threadgroup_sync, so a route may issue right after it succeeds.
 * Warp scope uses the whole warp, as the stable warp APIs do.
 */
static_assert(NVSHMEMX_SUCCESS < NVSHMEMX_ERROR_INVALID_VALUE &&
                  NVSHMEMX_ERROR_INVALID_VALUE < NVSHMEMX_ERROR_NOT_SUPPORTED &&
                  NVSHMEMX_ERROR_NOT_SUPPORTED < NVSHMEMX_ERROR_INTERNAL,
              "agree() and worse() rank preflight statuses by numeric value");

template <threadgroup Scope>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int agree(int status) {
    if constexpr (Scope == threadgroup::thread) {
        return status;
    } else if constexpr (Scope == threadgroup::warp) {
        __syncwarp();
#if __CUDA_ARCH__ >= 800
        return __reduce_max_sync(0xffffffffu, status);
#else
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            status = max(status, __shfl_xor_sync(0xffffffffu, status, offset));
        }
        return status;
#endif
    } else {
        /* One barrier when everyone succeeded; the reduction runs only on failure. */
        if (!__syncthreads_or(status != NVSHMEMX_SUCCESS)) {
            return NVSHMEMX_SUCCESS;
        }
        __shared__ int group_status;
        if (scope_leader<Scope>()) {
            group_status = NVSHMEMX_SUCCESS;
        }
        __syncthreads();
        atomicMax(&group_status, status);
        __syncthreads();
        status = group_status;
        __syncthreads();
        return status;
    }
}

/* ---- Memory spaces ----------------------------------------------------- */

enum class local_space {
    invalid,
    shared,
    global,
    generic,
};

__host__ __device__ constexpr local_space to_local_space(nvshmemx::memory_space::source space) {
    return space == nvshmemx::memory_space::source::shared    ? local_space::shared
           : space == nvshmemx::memory_space::source::global  ? local_space::global
           : space == nvshmemx::memory_space::source::generic ? local_space::generic
                                                              : local_space::invalid;
}

__host__ __device__ constexpr local_space to_local_space(nvshmemx::memory_space::dest space) {
    return space == nvshmemx::memory_space::dest::shared    ? local_space::shared
           : space == nvshmemx::memory_space::dest::global  ? local_space::global
           : space == nvshmemx::memory_space::dest::generic ? local_space::generic
                                                            : local_space::invalid;
}

/* Only a generic local operand is classified at run time. Explicit shared and
 * global values compile directly to their path and never call __isShared. */
template <local_space Space>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool is_shared_ptr(const void *ptr) {
    if constexpr (Space == local_space::shared) {
        return true;
    } else if constexpr (Space == local_space::global) {
        return false;
    } else {
        static_assert(Space == local_space::generic, "unsupported local memory space");
        return __isShared(ptr);
    }
}

/* A route's view of the local operand, whichever direction spelled it: `kind`
 * is its compile-time category and is_shared() classifies it. */
template <nvshmemx::memory_space::source Space>
struct source_local {
    static constexpr local_space kind = to_local_space(Space);
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE bool is_shared(const void *ptr) {
        return is_shared_ptr<kind>(ptr);
    }
};

template <nvshmemx::memory_space::dest Space>
struct dest_local {
    static constexpr local_space kind = to_local_space(Space);
    __device__ static NVSHMEMI_DEVICE_ALWAYS_INLINE bool is_shared(const void *ptr) {
        return is_shared_ptr<kind>(ptr);
    }
};

template <nvshmemx::memory_space::source Src, nvshmemx::memory_space::dest Dst>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void validate_put_memory_spaces() {
    static_assert(Dst == nvshmemx::memory_space::dest::global,
                  "transport-specific put destination must be remote symmetric global memory");
    static_assert(to_local_space(Src) != local_space::invalid,
                  "unsupported transport-specific put source memory space");
}

template <nvshmemx::memory_space::source Src, nvshmemx::memory_space::dest Dst>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void validate_get_memory_spaces() {
    static_assert(Src == nvshmemx::memory_space::source::global,
                  "transport-specific get source must be remote symmetric global memory");
    static_assert(to_local_space(Dst) != local_space::invalid,
                  "unsupported transport-specific get destination memory space");
}

/* Explicit shared and global annotations are trusted preconditions. Debug
 * builds check them; release builds never classify the pointer. */
template <typename Local>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void assert_local_annotation(const void *ptr) {
#ifdef NVSHMEM_DEBUG
    if constexpr (Local::kind == local_space::shared) {
        assert(ptr == nullptr || __isShared(ptr));
    } else if constexpr (Local::kind == local_space::global) {
        assert(!__isShared(ptr));
    }
#else
    (void)ptr;
#endif
}

/* ---- Argument checks shared by route preflights ------------------------ */

/* Saturate instead of wrapping so an overflowing element count reaches the byte
 * preflight, which rejects the heap range. */
template <typename Type>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE size_t bytes_for_elements(size_t nelems) {
    return nelems > static_cast<size_t>(-1) / sizeof(Type) ? static_cast<size_t>(-1)
                                                           : nelems * sizeof(Type);
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool heap_range_is_valid(const void *ptr, size_t bytes) {
    if (ptr == nullptr || nvshmemi_device_state_d.heap_base == nullptr) {
        return false;
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t heap_base = reinterpret_cast<uintptr_t>(nvshmemi_device_state_d.heap_base);
    if (address < heap_base) {
        return false;
    }

    const size_t offset = static_cast<size_t>(address - heap_base);
    const size_t heap_size = nvshmemi_device_state_d.heap_size;
    return offset <= heap_size && bytes <= heap_size - offset;
}

/* The arguments every RMA route needs: a valid PE, a remote range inside the
 * symmetric heap, and a local operand that may be null only when no bytes
 * move (the portable zero-length behavior). */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int check_rma_arguments(const void *remote,
                                                                 const void *local, size_t bytes,
                                                                 int pe) {
    const bool valid = pe >= 0 && pe < nvshmemi_device_state_d.npes &&
                       heap_range_is_valid(remote, bytes) && (local != nullptr || bytes == 0);
    return valid ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INVALID_VALUE;
}

/* The arguments every signalling route needs: a valid PE, a SET or ADD
 * opcode, and an 8-byte-aligned signal word inside the symmetric heap. */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int check_signal_arguments(const uint64_t *sig_addr,
                                                                    int sig_op, int pe) {
    const bool valid = pe >= 0 && pe < nvshmemi_device_state_d.npes &&
                       (sig_op == NVSHMEM_SIGNAL_SET || sig_op == NVSHMEM_SIGNAL_ADD) &&
                       reinterpret_cast<uintptr_t>(sig_addr) % sizeof(*sig_addr) == 0 &&
                       heap_range_is_valid(sig_addr, sizeof(*sig_addr));
    return valid ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INVALID_VALUE;
}

/* The more severe of two preflight statuses, with the ordering agree() uses. */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int worse(int a, int b) { return a > b ? a : b; }

}  // namespace transport_policy
}  // namespace nvshmemi

#endif  // __CUDA_ARCH__

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_TRAITS_CUH_
