/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMX_DEFINES_H_
#define _NVSHMEMX_DEFINES_H_

#include <cuda_runtime.h>
#include "device/nvshmem_device_macros.h"
#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/pt-to-pt/counted_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device/nvshmemx_collective_launch_apis.h"

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
extern "C" {
#endif

/*
 * nvshmemx_ask_smem - Returns the amount of shared memory (in bytes) that NVSHMEM
 * needs for TMA-based transfers.
 *
 * This function is available on both host and device. On the host side, the
 * returned value can be used to configure the dynamic shared memory size for
 * kernel launches. Each CTA must provide at least this much shared memory via
 * nvshmemx_give_smem() for TMA to be used.
 *
 * All returned values include NVSHMEMI_SMEM_DATA_REGION_OFFSET at the base
 * of the user's buffer; TMA data tiles live in the remainder.
 *
 * flag:
 *   NVSHMEMX_SMEM_RECOMMENDED  - Recommended amount for best performance.
 *                                 64 KiB: the maximum single cp.async.bulk
 *                                 transfer size on Hopper/Blackwell, sized to
 *                                 allow a full smem buffer for the gmem→gmem
 *                                 double-buffering path.
 *   NVSHMEMX_SMEM_MINIMUM      - Minimum for single-buffered TMA transfers.
 *                                 32 KiB: half of RECOMMENDED; sufficient for
 *                                 smem→gmem single-buffer puts.
 *   NVSHMEMX_SMEM_BARRIERS_ONLY - Only space for barriers and TMA descriptors
 *                                  (NVSHMEMI_SMEM_DATA_REGION_OFFSET); no
 *                                  data tile.  The gmem→gmem staging path
 *                                  won't run, but smem→gmem puts still work
 *                                  if the user manages their own data in the
 *                                  remainder of their own smem buffer.
 */
__host__ __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_ask_smem(
    nvshmemx_smem_amount_t flag) {
    switch (flag) {
        case NVSHMEMX_SMEM_RECOMMENDED:
            return 65536; /* 64 KiB */
        case NVSHMEMX_SMEM_MINIMUM:
            return 32768; /* 32 KiB */
        case NVSHMEMX_SMEM_BARRIERS_ONLY:
            return NVSHMEMI_SMEM_DATA_REGION_OFFSET;
        default:
            return 65536;
    }
}

/*
 * nvshmemx_give_smem - Give a block of shared memory to the NVSHMEM runtime for
 * TMA-based transfers.
 *
 * Must be called by all threads in each participating CTA before issuing
 * TMA-backed operations. All threads in a CTA must provide the same shared
 * memory pointer and size; different CTAs may provide different sizes.
 * Synchronize the CTA after registration before any thread uses the TMA path.
 *
 * User contract: every CTA that calls give_smem MUST call nvshmemx_release_smem()
 * before the kernel returns.
 *
 * TMA-backed put routing also requires 16-byte aligned source/destination
 * pointers and a 16-byte multiple transfer size.  Operations that do not meet
 * those routing constraints preserve the regular put contract by falling back
 * to P2P stores.
 * Block-scoped TMA puts from global-memory sources use a double-buffered
 * staging path that additionally requires at least two full warps in the CTA;
 * smaller CTAs fall back to P2P stores.
 *
 * smem: Pointer to shared memory (must be 16-byte aligned)
 * size: Size in bytes (must be >= nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM))
 */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_give_smem(void *smem, size_t size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) {
        return;
    }
    if (smem == NULL || size == 0) {
        return;
    }

    /* Size must be at least NVSHMEMI_SMEM_DATA_REGION_OFFSET — we reserve the
     * initial bytes for mbarriers/TMA descriptors.  A smaller allocation
     * can't hold our barriers, so don't register this CTA; it falls back to
     * P2P stores for all puts. */
    if (size < (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET) {
        return;
    }

    /* Use nvshmemi_tma_block_is_elected() — elect.sync with a shfl_sync
     * warp_id broadcast — so the compiler sees a warp-uniform predicate and
     * avoids inserting a peeling loop (which if(tid==0) would cause). */
    if (nvshmemi_tma_block_is_elected()) {
        int registration_slot = nvshmemi_tma_claim_smem_registration();
        if (registration_slot >= 0) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
            uintptr_t smem_base = reinterpret_cast<uintptr_t>(smem);
            for (int slot = 0; slot < NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS; slot++) {
                nvshmemi_handle_barrier_slot(smem_base, slot)->reset_pending_handle_state();
            }
            /* Publish the registration only after its deferred-completion
             * metadata has been initialized. */
            __threadfence_block();
#endif
            nvshmemi_tma_publish_smem_registration(registration_slot,
                                                   reinterpret_cast<uintptr_t>(smem), size);
        }
    }
    __syncthreads();
#endif /* __CUDA_ARCH__ >= 900 */
}

/*
 * nvshmemx_release_smem - Deregister this CTA's shared memory from the NVSHMEM
 * TMA runtime.
 *
 * Must be called by every CTA that previously called nvshmemx_give_smem(),
 * before the kernel returns.
 *
 * Call from all threads.  Any deferred handle PUTs are completed before the
 * elected warp-0 leader clears the registration.
 */
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_release_smem() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) {
        return;
    }
    int registration_slot = nvshmemi_tma_find_smem_registration();
    if (registration_slot >= 0) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
        nvshmemi_handle_quiet_owned();
        __syncthreads();
#endif
        if (nvshmemi_tma_block_is_elected()) {
            nvshmemi_tma_release_smem_registration(registration_slot);
        }
    }
#endif /* __CUDA_ARCH__ >= 900 */
}

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
}
#endif

#ifdef __CUDA_ARCH__
#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
extern "C" {
#endif

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_vendor_get_version_info(
    int *major, int *minor, int *patch) {
    *major = NVSHMEM_VENDOR_MAJOR_VERSION;
    *minor = NVSHMEM_VENDOR_MINOR_VERSION;
    *patch = NVSHMEM_VENDOR_PATCH_VERSION;
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_signal_op(uint64_t *sig_addr,
                                                                             uint64_t signal,
                                                                             int sig_op, int pe) {
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_signal_counted_reset(
    uint64_t *signal_addr) {
    *reinterpret_cast<volatile uint64_t *>(signal_addr) = 0;
}

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
nvshmemx_signal_counted_load(const uint64_t *signal_addr) {
    return *reinterpret_cast<const volatile uint64_t *>(signal_addr);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_signal_counted_wait_until(
    const uint64_t *signal_addr, uint64_t expected) {
    while ((uint64_t)(nvshmemx_signal_counted_load(signal_addr) - expected) >=
           (static_cast<uint64_t>(1) << 63)) {
    }
#if LE_HW_SW_REQUIREMENTS_MET && !defined(__CUDACC_RTC__) && \
    !defined(__clang_llvm_bitcode_lib__) && !defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
    fence_proxy_fabric2generic_acquire_system();
#endif
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemx_putmem_signal_counted_nbi_block(
    void *dest, const void *source, size_t bytes, uint64_t *signal_addr, int pe) {
    return nvshmemi_putmem_signal_counted_nbi_block(dest, source, bytes, signal_addr, pe);
}
#endif

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void *nvshmemx_mc_ptr(nvshmem_team_t team,
                                                                           const void *ptr) {
    return nvshmemi_mc_ptr(nvshmemi_device_state_d.team_pool[team], ptr);
}

#define NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_put_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe) {                             \
        nvshmemi_put<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe);          \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP
#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
}
#endif

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_signal(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr)) {
        volatile T *dest_actual =
            (volatile T *)((char *)(peer_base_addr) +
                           ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
    } else {
        nvshmemi_transfer_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
    }
}

#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)        \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_##TYPENAME##_put_signal##SC_SUFFIX( \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,    \
        int sig_op, int pe, bool is_nbi) {                                                     \
        NVSHMEMI_DECL_THREAD_IDX##SC_SUFFIX();                                                 \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (nvshmemi_peer_reachable(peer_base_addr)) {                                         \
            nvshmemx_##TYPENAME##_put##SC_SUFFIX(dest, source, nelems, pe);                    \
            if (myIdx == 0) {                                                                  \
                __threadfence_system();                                                        \
                nvshmemx_signal_op(sig_addr, signal, sig_op, pe);                              \
            }                                                                                  \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
        } else {                                                                               \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
            nvshmemi_transfer_put_signal<nvshmemi_threadgroup_##SCOPE>(                        \
                (void *)dest, (void *)source, nelems * sizeof(TYPE), (void *)sig_addr, signal, \
                (nvshmemi_amo_t)sig_op, pe, is_nbi);                                           \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
        }                                                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, block, _block,
                                                 x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
extern "C" {
#endif

/* __device__ nvshmem_<typename>_put_signal_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)     \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                    \
    nvshmemx_##TYPENAME##_put_signal##SC_SUFFIX(TYPE *dest, const TYPE *source, size_t nelems,   \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                int pe) {                                        \
        nvshmemi_put_signal<TYPE, nvshmemi_threadgroup_##SCOPE>(dest, source, nelems, sig_addr,  \
                                                                signal, sig_op, pe, 0);          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                           \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_putmem_signal##SC_SUFFIX( \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,      \
        int sig_op, int pe) {                                                                    \
        nvshmemi_put_signal<char, nvshmemi_threadgroup_##SCOPE>(                                 \
            (char *)dest, (const char *)source, nelems, sig_addr, signal, sig_op, pe, 0);        \
    }

NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putsize_signal_scope */
#define NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                  \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                  \
    nvshmemx_put##BITS##_signal##SC_SUFFIX(void *dest, const void *source, size_t nelems,      \
                                           uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                           int pe) {                                           \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems * (BITS / 8), sig_addr, signal, \
                                          sig_op, pe);                                         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_FOR_SCOPE

/* __device__ nvshmem_<typename>_put_signal_nbi_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)   \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                      \
    nvshmemx_##TYPENAME##_put_signal_nbi##SC_SUFFIX(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal,           \
                                                    int sig_op, int pe) {                          \
        nvshmemi_put_signal<TYPE, nvshmemi_threadgroup_##SCOPE>(dest, source, nelems, sig_addr,    \
                                                                signal, sig_op, pe, 1);            \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_nbi_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                 \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                              \
    nvshmemx_putmem_signal_nbi##SC_SUFFIX(void *dest, const void *source, size_t nelems,   \
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                          int pe) {                                        \
        nvshmemi_put_signal<char, nvshmemi_threadgroup_##SCOPE>(                           \
            (char *)dest, (const char *)source, nelems, sig_addr, signal, sig_op, pe, 1);  \
    }

NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL

#define NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)               \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                   \
    nvshmemx_put##BITS##_signal_nbi##SC_SUFFIX(void *dest, const void *source, size_t nelems,   \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                               int pe) {                                        \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems * (BITS / 8), sig_addr, signal,  \
                                          sig_op, pe);                                          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_NBI_FOR_SCOPE

#define NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_get_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe) {                             \
        nvshmemi_get<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe);          \
    }

#define DEFINE_NVSHMEM_TYPE_GET(Name, Type)        \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET)
#undef DEFINE_NVSHMEM_TYPE_GET

#define NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, Group)                                       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_put##Name##_##Group(  \
        void *dest, const void *source, size_t nelems, int pe) {                             \
        nvshmemi_put<Type, nvshmemi_threadgroup_##Group>((Type *)dest, (const Type *)source, \
                                                         nelems, pe);                        \
    }

#define DEFINE_NVSHMEM_PUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_THREADGROUP

#define NVSHMEM_GETSIZE_THREADGROUP(Name, Type, Group)                                       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_get##Name##_##Group(  \
        void *dest, const void *source, size_t nelems, int pe) {                             \
        nvshmemi_get<Type, nvshmemi_threadgroup_##Group>((Type *)dest, (const Type *)source, \
                                                         nelems, pe);                        \
    }

#define DEFINE_NVSHMEM_GETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_THREADGROUP(Group)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_putmem_##Group(       \
        void *dest, const void *source, size_t bytes, int pe) {                              \
        nvshmemi_put<char, nvshmemi_threadgroup_##Group>((char *)dest, (const char *)source, \
                                                         bytes, pe);                         \
    }

DEFINE_NVSHMEM_PUTMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_THREADGROUP(Group)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_getmem_##Group(       \
        void *dest, const void *source, size_t bytes, int pe) {                              \
        nvshmemi_get<char, nvshmemi_threadgroup_##Group>((char *)dest, (const char *)source, \
                                                         bytes, pe);                         \
    }

DEFINE_NVSHMEM_GETMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_THREADGROUP(block)

#define NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_put_nbi_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe) {                                 \
        nvshmemi_put_nbi<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe);          \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_get_nbi_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe) {                                 \
        nvshmemi_get_nbi<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe);          \
    }

#define DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP

#define NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, Group)                                       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_put##Name##_nbi_##Group(  \
        void *dest, const void *source, size_t nelems, int pe) {                                 \
        nvshmemi_put_nbi<Type, nvshmemi_threadgroup_##Group>((Type *)dest, (const Type *)source, \
                                                             nelems, pe);                        \
    }

#define DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP

#define NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, Group)                                       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_get##Name##_nbi_##Group(  \
        void *dest, const void *source, size_t nelems, int pe) {                                 \
        nvshmemi_get_nbi<Type, nvshmemi_threadgroup_##Group>((Type *)dest, (const Type *)source, \
                                                             nelems, pe);                        \
    }

#define DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(Group)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_putmem_nbi_##Group(       \
        void *dest, const void *source, size_t bytes, int pe) {                                  \
        nvshmemi_put_nbi<char, nvshmemi_threadgroup_##Group>((char *)dest, (const char *)source, \
                                                             bytes, pe);                         \
    }

DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(Group)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_getmem_nbi_##Group(       \
        void *dest, const void *source, size_t bytes, int pe) {                                  \
        nvshmemi_get_nbi<char, nvshmemi_threadgroup_##Group>((char *)dest, (const char *)source, \
                                                             bytes, pe);                         \
    }

DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(block)

#define NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, Group)                                          \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_iput_##Group(     \
        Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {    \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);         \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            Type *dest_actual;                                                                    \
            dest_actual = (Type *)((char *)(peer_base_addr) +                                     \
                                   ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *(dest_actual + i * dst) = *((volatile Type *)source + i * sst);                  \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_" #Name "_iput_" #Group                                              \
                   " not implemented over remote network transports\n");                          \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP

#define NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, Group)                                           \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_iput##Name##_##Group(      \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {    \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);         \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            Type *dest_actual;                                                                    \
            dest_actual = (Type *)((char *)(peer_base_addr) +                                     \
                                   ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *((Type *)dest_actual + i * dst) = *((Type *)source + i * sst);                   \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_iput" #Name "_" #Group                                               \
                   " not implemented over remote network transports\n");                          \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP

#define NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, Group)                                       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_iget_##Group(  \
        Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (peer_base_addr) {                                                                  \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                          \
            Type *source_actual;                                                               \
            source_actual =                                                                    \
                (Type *)((char *)(peer_base_addr) +                                            \
                         ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));      \
            int i;                                                                             \
            for (i = myIdx; i < nelems; i += groupSize) {                                      \
                *(dest + i * dst) = *(source_actual + i * sst);                                \
            }                                                                                  \
            NVSHMEMI_SYNC_##Group();                                                           \
        } else {                                                                               \
            printf("nvshmemx_" #Name "_iget_" #Group                                           \
                   " not implemented over remote network transports\n");                       \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP

#define NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, Group)                                        \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_iget##Name##_##Group(   \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (peer_base_addr) {                                                                  \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                          \
            char *source_actual;                                                               \
            source_actual = ((char *)(peer_base_addr) +                                        \
                             ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));  \
            int i;                                                                             \
            for (i = myIdx; i < nelems; i += groupSize) {                                      \
                *((Type *)dest + i * dst) = *((Type *)source_actual + i * sst);                \
            }                                                                                  \
            NVSHMEMI_SYNC_##Group();                                                           \
        } else {                                                                               \
            printf("nvshmemx_iget" #Name "_" #Group                                            \
                   " not implemented over remote network transports\n");                       \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define DEFINE_NVSHMEM_IGETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IGETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IGETSIZE_THREADGROUP

/* qpair specific APIs*/
#define NVSHMEM_TYPE_GET_QP_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_get_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) { \
        nvshmemi_get<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe, qp_index);   \
    }

#define NVSHMEM_TYPE_GET_QP(Name, Type)                                                         \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_get(         \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) { \
        nvshmemi_get<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, pe, qp_index);    \
    }

#define DEFINE_NVSHMEM_TYPE_GET_QP(Name, Type)        \
    NVSHMEM_TYPE_GET_QP(Name, Type)                   \
    NVSHMEM_TYPE_GET_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_QP)
#undef DEFINE_NVSHMEM_TYPE_GET_QP
#undef NVSHMEM_TYPE_GET_QP_THREADGROUP
#undef NVSHMEM_TYPE_GET_QP

#define NVSHMEM_TYPE_GET_NBI_QP_THREADGROUP(Name, Type, Group)                                    \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                     \
    nvshmemx_qp_##Name##_get_nbi_##Group(Type *dest, const Type *source, size_t nelems, int pe,   \
                                         nvshmemx_qp_handle_t qp_index) {                         \
        nvshmemi_get_nbi<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe, qp_index); \
    }

#define NVSHMEM_TYPE_GET_NBI_QP(Name, Type)                                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_get_nbi(      \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) {  \
        nvshmemi_get_nbi<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, pe, qp_index); \
    }

#define DEFINE_NVSHMEM_TYPE_GET_NBI_QP(Name, Type)        \
    NVSHMEM_TYPE_GET_NBI_QP(Name, Type)                   \
    NVSHMEM_TYPE_GET_NBI_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_NBI_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_NBI_QP)
#undef DEFINE_NVSHMEM_TYPE_GET_NBI_QP
#undef NVSHMEM_TYPE_GET_NBI_QP_THREADGROUP
#undef NVSHMEM_TYPE_GET_NBI_QP

#define NVSHMEM_TYPE_PUT_QP_THREADGROUP(Name, Type, Group)                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_put_##Group( \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) { \
        nvshmemi_put<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe, qp_index);   \
    }

#define NVSHMEM_TYPE_PUT_QP(Name, Type)                                                         \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_put(         \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) { \
        nvshmemi_put<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, pe, qp_index);    \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_QP(Name, Type)        \
    NVSHMEM_TYPE_PUT_QP(Name, Type)                   \
    NVSHMEM_TYPE_PUT_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_PUT_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_QP)
#undef DEFINE_NVSHMEM_TYPE_PUT_QP
#undef NVSHMEM_TYPE_PUT_QP_THREADGROUP
#undef NVSHMEM_TYPE_PUT_QP

#define NVSHMEM_TYPE_PUT_NBI_QP_THREADGROUP(Name, Type, Group)                                    \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                     \
    nvshmemx_qp_##Name##_put_nbi_##Group(Type *dest, const Type *source, size_t nelems, int pe,   \
                                         nvshmemx_qp_handle_t qp_index) {                         \
        nvshmemi_put_nbi<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe, qp_index); \
    }

#define NVSHMEM_TYPE_PUT_NBI_QP(Name, Type)                                                      \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_put_nbi(      \
        Type *dest, const Type *source, size_t nelems, int pe, nvshmemx_qp_handle_t qp_index) {  \
        nvshmemi_put_nbi<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, pe, qp_index); \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_NBI_QP(Name, Type)        \
    NVSHMEM_TYPE_PUT_NBI_QP(Name, Type)                   \
    NVSHMEM_TYPE_PUT_NBI_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_PUT_NBI_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_NBI_QP)
#undef DEFINE_NVSHMEM_TYPE_PUT_NBI_QP
#undef NVSHMEM_TYPE_PUT_NBI_QP_THREADGROUP
#undef NVSHMEM_TYPE_PUT_NBI_QP

/*__device__ nvshmem_p*/
#define NVSHMEM_TYPENAME_P_QP(TYPENAME, TYPE)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##TYPENAME##_p( \
        TYPE *dest, const TYPE value, int pe, nvshmemx_qp_handle_t qp_index) {            \
        nvshmemi_p<TYPE>(dest, value, pe, qp_index);                                      \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPENAME_P_QP)
#undef NVSHMEMI_TYPENAME_P_QP

/*__device__ nvshmem_g*/
#define NVSHMEM_TYPENAME_G_QP(TYPENAME, TYPE)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE TYPE nvshmemx_qp_##TYPENAME##_g( \
        const TYPE *source, int pe, nvshmemx_qp_handle_t qp_index) {                      \
        return nvshmemi_g<TYPE>(source, pe, qp_index);                                    \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPENAME_G_QP)
#undef NVSHMEMI_TYPENAME_G_QP

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_signal_op(
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, nvshmemx_qp_handle_t qp_index) {
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe, qp_index);
}

#define NVSHMEM_TYPE_PUT_SIGNAL_QP_THREADGROUP(Name, Type, Group)                                 \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                     \
    nvshmemx_qp_##Name##_put_signal_##Group(Type *dest, const Type *source, size_t nelems,        \
                                            uint64_t *sig_addr, uint64_t signal, int sig_op,      \
                                            int pe, nvshmemx_qp_handle_t qp_index) {              \
        nvshmemi_put_signal<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, sig_addr,   \
                                                                signal, sig_op, pe, 0, qp_index); \
    }

#define NVSHMEM_TYPE_PUT_SIGNAL_QP(Name, Type)                                                   \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_put_signal(   \
        Type *dest, const Type *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,      \
        int sig_op, int pe, nvshmemx_qp_handle_t qp_index) {                                     \
        nvshmemi_put_signal<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, sig_addr,   \
                                                               signal, sig_op, pe, 0, qp_index); \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_QP(Name, Type)        \
    NVSHMEM_TYPE_PUT_SIGNAL_QP(Name, Type)                   \
    NVSHMEM_TYPE_PUT_SIGNAL_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_PUT_SIGNAL_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_QP)
#undef DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_QP
#undef NVSHMEM_TYPE_PUT_SIGNAL_QP_THREADGROUP
#undef NVSHMEM_TYPE_PUT_SIGNAL_QP

#define NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP_THREADGROUP(Name, Type, Group)                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                     \
    nvshmemx_qp_##Name##_put_signal_nbi_##Group(Type *dest, const Type *source, size_t nelems,    \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op,  \
                                                int pe, nvshmemx_qp_handle_t qp_index) {          \
        nvshmemi_put_signal<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, sig_addr,   \
                                                                signal, sig_op, pe, 1, qp_index); \
    }

#define NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP(Name, Type)                                                 \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_##Name##_put_signal_nbi( \
        Type *dest, const Type *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,        \
        int sig_op, int pe, nvshmemx_qp_handle_t qp_index) {                                       \
        nvshmemi_put_signal<Type, nvshmemi_threadgroup_thread>(dest, source, nelems, sig_addr,     \
                                                               signal, sig_op, pe, 1, qp_index);   \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP(Name, Type)        \
    NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP(Name, Type)                   \
    NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP)
#undef DEFINE_NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP
#undef NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP_THREADGROUP
#undef NVSHMEM_TYPE_PUT_SIGNAL_NBI_QP

/*
 * nvshmemx_flush - Wait until all source buffers used by
 * preceding non-blocking puts issued from this thread are safe to reuse.
 *
 * Guarantees reusability only: the source buffer may be overwritten or
 * freed after this call returns.  Does NOT guarantee that the data is visible
 * at the remote PE; callers must still use nvshmem_quiet() / nvshmem_fence()
 * before the remote consumer reads the destination.
 *
 * For NVLink (P2P) puts: st.global stores are blocking at the instruction
 * level, so the source is already consumed when put_nbi returns.  This call
 * is a no-op on pure-P2P deployments.
 *
 * For network (IB/RoCE, EFA, proxy) puts: waits for the transport to confirm
 * that the source buffer has been DMA'd.  Does not issue __threadfence_system.
 */
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_flush(void) {
    nvshmemi_flush<NVSHMEMI_THREADGROUP_THREAD>();
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_flush_warp(void) {
    nvshmemi_flush<NVSHMEMI_THREADGROUP_WARP>();
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_flush_block(void) {
    nvshmemi_flush<NVSHMEMI_THREADGROUP_BLOCK>();
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_quiet(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>(pe, qp_handle, num_qps);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_quiet_warp(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_quiet<NVSHMEMI_THREADGROUP_WARP>(pe, qp_handle, num_qps);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_quiet_block(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>(pe, qp_handle, num_qps);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_fence(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_fence<NVSHMEMI_THREADGROUP_THREAD>(pe, qp_handle, num_qps);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_fence_warp(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_fence<NVSHMEMI_THREADGROUP_WARP>(pe, qp_handle, num_qps);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_qp_fence_block(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    nvshmemi_fence<NVSHMEMI_THREADGROUP_BLOCK>(pe, qp_handle, num_qps);
}

/* end qpair specific APIs*/

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
}
#endif

#endif /* __CUDA_ARCH__ */

#include "non_abi/device/coll/defines.cuh"

#endif
