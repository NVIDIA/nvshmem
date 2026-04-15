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
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device/nvshmemx_collective_launch_apis.h"

/*
 * nvshmemx_ask_smem - Returns the amount of shared memory (in bytes) that NVSHMEM
 * needs for TMA-based transfers.
 *
 * This function is available on both host and device. On the host side, the
 * returned value can be used to configure the dynamic shared memory size for
 * kernel launches. Each CTA must provide at least this much shared memory via
 * nvshmemx_give_smem() for TMA to be used.
 *
 * flag:
 *   NVSHMEMX_SMEM_RECOMMENDED  - Recommended amount for best performance.
 *                                 64 KiB: the maximum single cp.async.bulk
 *                                 transfer size on Hopper/Blackwell, sized to
 *                                 allow a full smem buffer for the gmem→gmem
 *                                 double-buffering path (future MR).
 *   NVSHMEMX_SMEM_MINIMUM      - Minimum for single-buffered TMA transfers.
 *                                 32 KiB: half of RECOMMENDED; sufficient for
 *                                 smem→gmem single-buffer puts.
 *   NVSHMEMX_SMEM_BARRIERS_ONLY - Only space for barriers and TMA descriptors
 *                                  (256 B); no data buffer. Useful when the
 *                                  user manages buffering externally.
 */
__host__ __device__ inline int nvshmemx_ask_smem(nvshmemx_smem_amount_t flag) {
    switch (flag) {
        case NVSHMEMX_SMEM_RECOMMENDED:
            return 65536; /* 64 KiB */
        case NVSHMEMX_SMEM_MINIMUM:
            return 32768; /* 32 KiB */
        case NVSHMEMX_SMEM_BARRIERS_ONLY:
            return 256;
        default:
            return 65536;
    }
}

/*
 * nvshmemx_give_smem - Give a block of shared memory to the NVSHMEM runtime for
 * TMA-based transfers.
 *
 * Must be called by EVERY CTA in the grid, once per kernel launch, before
 * issuing any TMA-backed puts.  CTAs that skip this call will fall back to
 * P2P stores for all puts in that kernel.
 *
 * User contract: every CTA that calls give_smem MUST call nvshmemx_release_smem()
 * before the kernel returns.  Without this, the registration persists across
 * kernel launches: a later kernel whose CTAs share block_ids with a prior kernel
 * and omit give_smem will unexpectedly take the TMA path using a stale pointer.
 *
 * TODO: investigate automating the clear (e.g. via a pre-launch host helper or
 * a CUDA graph epilogue) so users are not required to call release_smem manually.
 *
 * The registered smem base pointer is stored persistently because it will serve
 * as the staging buffer in the upcoming gmem→gmem TMA path.  Until that path
 * is implemented, it acts only as a presence gate (non-zero = TMA enabled for
 * this CTA).
 *
 * Note: grids larger than NVSHMEMI_TMA_MAX_BLOCKS CTAs are supported, but
 * CTAs with block_id >= NVSHMEMI_TMA_MAX_BLOCKS cannot register and will
 * fall back to P2P stores (a warning is printed by block 0 thread 0).
 *
 * smem: Pointer to shared memory (must be 16-byte aligned)
 * size: Size in bytes (must be >= nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM))
 */
__device__ inline void nvshmemx_give_smem(char *smem, size_t size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) return;
    if (smem == NULL || size == 0) return;

    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    uintptr_t *bases = nvshmemi_device_state_d.tma_smem_bases;
    if (bases == NULL || (size_t)block_id >= nvshmemi_device_state_d.tma_smem_bases_len) {
        /* Grid is larger than NVSHMEMI_TMA_MAX_BLOCKS; this CTA cannot use TMA.
         * Print a one-shot warning from block 0 thread 0 to avoid log spam. */
        if (block_id == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            printf("NVSHMEM WARNING: grid has %d CTAs but TMA smem table only holds %zu entries; "
                   "CTAs with block_id >= %zu will use P2P stores instead of TMA.\n",
                   gridDim.x * gridDim.y * gridDim.z,
                   nvshmemi_device_state_d.tma_smem_bases_len,
                   nvshmemi_device_state_d.tma_smem_bases_len);
        return;
    }
    /* Use nvshmemi_tma_block_is_elected() — elect.sync with a shfl_sync
     * warp_id broadcast — so the compiler sees a warp-uniform predicate and
     * avoids inserting a peeling loop (which if(tid==0) would cause). */
    if (nvshmemi_tma_block_is_elected()) {
        bases[block_id] = (uintptr_t)smem;
    }
#endif /* __CUDA_ARCH__ >= 900 */
}

/*
 * nvshmemx_release_smem - Deregister this CTA's shared memory from the NVSHMEM
 * TMA runtime.
 *
 * Must be called by every CTA that previously called nvshmemx_give_smem(),
 * before the kernel returns.  Zeroes the tma_smem_bases entry so that a later
 * kernel whose CTAs share the same block_id does not inherit a stale pointer
 * and unexpectedly take the TMA path.
 *
 * Call from all threads; only the elected warp-0 leader performs the write.
 * Must be followed by __syncthreads() if any threads still need to observe
 * the cleared state before proceeding.
 */
__device__ inline void nvshmemx_release_smem() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) return;
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    uintptr_t *bases = nvshmemi_device_state_d.tma_smem_bases;
    if (bases != NULL && (size_t)block_id < nvshmemi_device_state_d.tma_smem_bases_len) {
        if (nvshmemi_tma_block_is_elected()) {
            bases[block_id] = 0;
        }
    }
#endif /* __CUDA_ARCH__ >= 900 */
}

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
    if (peer_base_addr != NULL) {
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
        if (peer_base_addr) {                                                                  \
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
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)       \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                      \
        nvshmemx_##TYPENAME##_put_signal##SC_SUFFIX(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal,           \
                                                    int sig_op, int pe) {                          \
        nvshmemi_put_signal<TYPE, nvshmemi_threadgroup_##SCOPE>(dest, source, nelems, sig_addr,    \
                                                                signal, sig_op, pe, 0);            \
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
#define NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                   \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                   \
        nvshmemx_put##BITS##_signal##SC_SUFFIX(void *dest, const void *source, size_t nelems,   \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                               int pe) {                                        \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal,   \
                                          sig_op, pe);                                          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_FOR_SCOPE

/* __device__ nvshmem_<typename>_put_signal_nbi_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                    \
        nvshmemx_##TYPENAME##_put_signal_nbi##SC_SUFFIX(TYPE *dest, const TYPE *source,          \
                                                        size_t nelems, uint64_t *sig_addr,       \
                                                        uint64_t signal, int sig_op, int pe) {   \
        nvshmemi_put_signal<TYPE, nvshmemi_threadgroup_##SCOPE>(dest, source, nelems, sig_addr,  \
                                                                signal, sig_op, pe, 1);          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_nbi_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                     \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                  \
        nvshmemx_putmem_signal_nbi##SC_SUFFIX(void *dest, const void *source, size_t nelems,   \
                                              uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                              int pe) {                                        \
        nvshmemi_put_signal<char, nvshmemi_threadgroup_##SCOPE>(                               \
            (char *)dest, (const char *)source, nelems, sig_addr, signal, sig_op, pe, 1);      \
    }

NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL

#define NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                 \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                                     \
        nvshmemx_put##BITS##_signal_nbi##SC_SUFFIX(void *dest, const void *source, size_t nelems, \
                                                   uint64_t *sig_addr, uint64_t signal,           \
                                                   int sig_op, int pe) {                          \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal,     \
                                          sig_op, pe);                                            \
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
        nvshmemx_qp_##Name##_get_nbi_##Group(Type *dest, const Type *source, size_t nelems,       \
                                             int pe, nvshmemx_qp_handle_t qp_index) {             \
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
        nvshmemx_qp_##Name##_put_nbi_##Group(Type *dest, const Type *source, size_t nelems,       \
                                             int pe, nvshmemx_qp_handle_t qp_index) {             \
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
        nvshmemx_qp_##Name##_put_signal_##Group(Type *dest, const Type *source, size_t nelems,    \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op,  \
                                                int pe, nvshmemx_qp_handle_t qp_index) {          \
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
        nvshmemx_qp_##Name##_put_signal_nbi_##Group(                                              \
            Type *dest, const Type *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,   \
            int sig_op, int pe, nvshmemx_qp_handle_t qp_index) {                                  \
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
