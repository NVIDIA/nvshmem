/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NVSHMEM_COMMON_DEVICE_CUH_
#define _NVSHMEM_COMMON_DEVICE_CUH_

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/limits>
#if !defined __CUDACC_RTC__
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#else
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#endif
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
// This is added so the entrypoint (init_device.cu) can receive the implementations of NVSHMEM
// transfer APIs.
#if defined(NVSHMEM_ENABLE_ALL_DEVICE_INLINING) || defined(__NVSHMEM_NUMBA_SUPPORT__) || \
    defined(NVSHMEM_BUILD_LTOIR_LIBRARY) || defined(NVSHMEM_BUILD_P2P_ONLY)
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
// proxy_device.cuh has no reachable call sites in P2P-only builds.
#if !defined(NVSHMEM_BUILD_P2P_ONLY)
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#endif
#include "non_abi/device/common/nvshmemi_path_predicates.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "device/logical_endpoint_device.cuh"
#include "non_abi/device/pt-to-pt/tma_device.cuh"

#define _LL_MAX_UNROLL 4

#define _LL_128_FLAG_THREAD 8
#define _LL_128_FLAG_SIZE 8
#define _LL_128_STORE_SIZE 16
#define _LL_128_PACKETS_PER_WARP 4
#define _LL_128_PACKET_DATA_SIZE 120
#define _LL_128_PACKET_PSYNC_SIZE 128
#define _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) (_LL_128_PACKET_DATA_SIZE / sizeof(T))
#define _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) (_LL_128_PACKET_PSYNC_SIZE / sizeof(T))
#define _LL_128_NUM_ELEMS_PER_STORE(T) (_LL_128_STORE_SIZE / sizeof(T))
#define _LL_128_NUM_ELEMS_PER_FLAG(T) (_LL_128_FLAG_SIZE / sizeof(T))
#define _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) \
    (UNROLL * _LL_128_PACKETS_PER_WARP * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T))
#define _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T) \
    (UNROLL * _LL_128_PACKETS_PER_WARP * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T))

#define _LL_8_DATA_BYTES_PER_WARP 256
#define _LL_8_PSYNC_BYTES_PER_WARP 512
#define _LL_128_DATA_BYTES_PER_WARP 480
#define _LL_128_PSYNC_BYTES_PER_WARP 512

typedef enum { _LL_PSYNC_NON_VOLATILE = 0, _LL_PSYNC_VOLATILE } nvshmemi_psync_volatile_t;

#ifdef __CUDA_ARCH__

__device__ __forceinline__ void *nvshmemi_ptr_add(void *ptr, size_t byte_offset) {
    return static_cast<void *>(static_cast<char *>(ptr) + byte_offset);
}

__device__ __forceinline__ const void *nvshmemi_ptr_add(const void *ptr, size_t byte_offset) {
    return static_cast<const void *>(static_cast<const char *>(ptr) + byte_offset);
}

/* This is used to create custom floating type traits such fp16, fp8, and native float types etc
 */
#if !defined __CUDACC_RTC__
template <typename T>
struct is_float : std::false_type {};
template <>
struct is_float<float> : std::true_type {};

template <typename T>
struct is_half : std::false_type {};
template <>
struct is_half<half> : std::true_type {};

template <typename T>
struct is_bfloat : std::false_type {};
template <>
struct is_bfloat<__nv_bfloat16> : std::true_type {};

template <typename T>
struct is_double : std::false_type {};
template <>
struct is_double<double> : std::true_type {};

template <typename T>
struct is_uint16 : std::false_type {};
template <>
struct is_uint16<uint16_t> : std::true_type {};

template <typename T>
struct is_int16 : std::false_type {};
template <>
struct is_int16<int16_t> : std::true_type {};

#else

template <typename T>
struct is_float : cuda::std::false_type {};
template <>
struct is_float<float> : cuda::std::true_type {};

template <typename T>
struct is_double : cuda::std::false_type {};
template <>
struct is_double<double> : cuda::std::true_type {};

template <typename T>
struct is_half : cuda::std::false_type {};
template <>
struct is_half<half> : cuda::std::true_type {};

template <typename T>
struct is_bfloat : cuda::std::false_type {};
template <>
struct is_bfloat<__nv_bfloat16> : cuda::std::true_type {};

template <typename T>
struct is_uint16 : cuda::std::false_type {};
template <>
struct is_uint16<uint16_t> : cuda::std::true_type {};

template <typename T>
struct is_int16 : cuda::std::false_type {};
template <>
struct is_int16<int16_t> : cuda::std::true_type {};

#endif

// Fence instructions
inline __device__ void fence_async_proxy() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_can_use_handle_atomic(T *target, int pe);

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_handle_atomic_fetch(T *target, T value,
                                                                        T compare, int pe);

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_atomic_nonfetch(T *target, T value,
                                                                              int pe);

#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
// forward declaration

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put(const void *src, void *dst,
                                                                  size_t len, int pe,
                                                                  bool is_blocking);

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_get(const void *src, void *dst,
                                                                  size_t len, int pe,
                                                                  bool is_blocking);

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_p(void *__restrict__ dst, const T src,
                                                                int pe);

#endif  // LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)

__device__ int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe,
                                          nvshmemi_team_t *dest_team);
__device__ long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op);
__device__ long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_test(volatile T *ivar, int cmp, T cmp_value) {
    int return_value = 0;
    if (NVSHMEM_CMP_GE == cmp) {
        if (*ivar >= cmp_value) {
            return_value = 1;
        }
    } else if (NVSHMEM_CMP_EQ == cmp) {
        if (*ivar == cmp_value) {
            return_value = 1;
        }
    } else if (NVSHMEM_CMP_NE == cmp) {
        if (*ivar != cmp_value) {
            return_value = 1;
        }
    } else if (NVSHMEM_CMP_GT == cmp) {
        if (*ivar > cmp_value) {
            return_value = 1;
        }
    } else if (NVSHMEM_CMP_LT == cmp) {
        if (*ivar < cmp_value) {
            return_value = 1;
        }
    } else if (NVSHMEM_CMP_LE == cmp) {
        if (*ivar <= cmp_value) {
            return_value = 1;
        }
    }
    return return_value;
}

#if !defined __CUDACC_RTC__
#define TYPE_IS_FLOAT(T) std::is_floating_point<T>::value
#else
#define TYPE_IS_FLOAT(T) cuda::std::is_floating_point<T>::value
#endif

// mcast store of 16B
template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_mcast16_store_threadgroup(int4 *dest,
                                                                                 const int4 *source,
                                                                                 size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx * 4; j < (len / sizeof(uint32_t)); j += groupSize * 4) {
        uint32_t u4[4];
        asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4]; "
            : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
            : "l"(source + j / 4));
        asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dest + j / 4), "r"(u4[0]),
            "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
            : "memory");
    }
}

// mcast store of 8B
template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_mcast8_store_threadgroup(
    uint64_t *dest, const uint64_t *source, size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx; j < (len / sizeof(uint64_t)); j += groupSize) {
        uint64_t val1;
        asm("ld.global.b64 %0, [%1];" : "=l"(val1) : "l"(source + j));
        asm("multimem.st.global.u64 [%0], %1;" ::"l"(dest + j), "l"(val1) : "memory");
    }
}

// mcast store of 4B
template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_mcast4_store_threadgroup(
    uint32_t *dest, const uint32_t *source, size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx; j < (len / sizeof(uint32_t)); j += groupSize) {
        uint32_t val1;
        asm("ld.global.b32 %0, [%1];" : "=r"(val1) : "l"(source + j));
        asm("multimem.st.global.u32 [%0], %1;" ::"l"(dest + j), "r"(val1) : "memory");
    }
}

/**
 * This function returns non-zero unaligned bytes that require a unicast store
 */
template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE size_t
nvshmemi_mcast_memcpy_threadgroup(T *__restrict__ dst, const T *__restrict__ src, size_t len) {
    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0 && len >= 16) {
        int4 *__restrict__ dst_p = (int4 *)dst;
        const int4 *__restrict__ src_p = (const int4 *)src;
        const size_t nelems = len / 16;
        nvshmemi_mcast16_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 16;

        if (0 == len) {
            return 0;
        }
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0 && len >= 8) {
        const size_t nelems = len / 8;
        uint64_t *__restrict__ dst_p = (uint64_t *)dst;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        nvshmemi_mcast8_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 8;

        if (0 == len) {
            return 0;
        }
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0 && len >= 4) {
        const size_t nelems = len / 4;
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        nvshmemi_mcast4_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 4;

        if (0 == len) {
            return 0;
        }
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /* if len is non-zero, caller will retry with unicast stores */
    return (len);
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_memcpy_threadgroup(
    void *__restrict__ dst, const void *__restrict__ src, size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0) {
        const size_t nelems = len / 16;

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        for (size_t i = myIdx * 4; i < nelems * 4; i += groupSize * 4) {
            asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"((dst_p + i)),
                         "r"(*(src_p + i)), "r"(*(src_p + i + 1)), "r"(*(src_p + i + 2)),
                         "r"(*(src_p + i + 3)));
        }
#else
        int4 *__restrict__ dst_p = (int4 *)dst;
        const int4 *__restrict__ src_p = (const int4 *)src;
        for (size_t i = myIdx; i < nelems; i += groupSize) {
            dst_p[i] = src_p[i];
        }
#endif
        len -= nelems * 16;

        if (0 == len) {
            return;
        }

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
        dst = (void *)(dst_p + nelems * 4);
        src = (void *)(src_p + nelems * 4);
#else
        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
#endif
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0) {
        uint64_t *__restrict__ dst_p = (uint64_t *)dst;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        const size_t nelems = len / 8;

        for (size_t i = myIdx; i < nelems; i += groupSize) {
            dst_p[i] = src_p[i];
        }

        len -= nelems * 8;

        if (0 == len) {
            return;
        }

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0) {
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        const size_t nelems = len / 4;

        for (size_t i = myIdx; i < nelems; i += groupSize) {
            dst_p[i] = src_p[i];
        }

        len -= nelems * 4;

        if (0 == len) {
            return;
        }

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 2B aligned copy as much as possible using 2B chunks
     */
    if ((uintptr_t)dst % 2 == 0 && (uintptr_t)src % 2 == 0) {
        uint16_t *__restrict__ dst_p = (uint16_t *)dst;
        const uint16_t *__restrict__ src_p = (const uint16_t *)src;
        const size_t nelems = len / 2;

        for (size_t i = myIdx; i < nelems; i += groupSize) {
            dst_p[i] = src_p[i];
        }

        len -= nelems * 2;

        if (0 == len) {
            return;
        }

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    unsigned char *__restrict__ dst_c = (unsigned char *)dst;
    const unsigned char *__restrict__ src_c = (const unsigned char *)src;

    for (size_t i = myIdx; i < len; i += groupSize) {
        dst_c[i] = src_c[i];
    }
}

/* Qualify each CTA's block index with its grid ID so concurrent grids use
 * distinct registration slots. Owner keys follow the base-pointer table. */

__device__ __forceinline__ uint64_t nvshmemi_tma_registration_key() {
    uint64_t grid_id = nvshmemi_get_grid_id();
    return grid_id == UINT64_MAX ? 0 : grid_id + 1;
}

__device__ __forceinline__ unsigned long long *nvshmemi_tma_smem_owner_keys() {
    uintptr_t *bases = nvshmemi_device_state_d.tma_smem_bases;
    if (bases == NULL) {
        return NULL;
    }
    return reinterpret_cast<unsigned long long *>(bases +
                                                  nvshmemi_device_state_d.tma_smem_bases_len);
}

__device__ __forceinline__ int nvshmemi_tma_smem_registration_slot() {
    size_t len = nvshmemi_device_state_d.tma_smem_bases_len;
    constexpr size_t grid_namespaces = NVSHMEMI_TMA_MAX_CONCURRENT_GRIDS;
    if (len < grid_namespaces || len % grid_namespaces != 0) {
        return -1;
    }

    size_t blocks_per_grid = len / grid_namespaces;
    uint32_t block_id = nvshmemi_get_flat_blk_idx();
    if (block_id >= blocks_per_grid) {
        return -1;
    }

    size_t grid_slot = nvshmemi_get_grid_id() % grid_namespaces;
    return static_cast<int>(grid_slot * blocks_per_grid + block_id);
}

__device__ __forceinline__ int nvshmemi_tma_find_smem_registration() {
    unsigned long long *owners = nvshmemi_tma_smem_owner_keys();
    uint64_t key = nvshmemi_tma_registration_key();
    int slot = nvshmemi_tma_smem_registration_slot();
    if (key == 0 || owners == NULL || slot < 0) {
        return -1;
    }

    return reinterpret_cast<volatile unsigned long long *>(owners)[slot] == key ? slot : -1;
}

struct nvshmemi_tma_smem_registration_t {
    uintptr_t base;
    size_t size;
    int slot;

    __device__ __forceinline__ bool is_valid() const { return slot >= 0 && base != 0; }
};

__device__ __forceinline__ nvshmemi_tma_smem_registration_t nvshmemi_tma_get_smem_registration() {
    nvshmemi_tma_smem_registration_t registration = {0, 0, -1};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) {
        return registration;
    }

    int slot = nvshmemi_tma_find_smem_registration();
    if (slot < 0) {
        return registration;
    }

    registration.base =
        reinterpret_cast<volatile uintptr_t *>(nvshmemi_device_state_d.tma_smem_bases)[slot];
    if (nvshmemi_device_state_d.tma_smem_size != NULL) {
        registration.size =
            reinterpret_cast<volatile size_t *>(nvshmemi_device_state_d.tma_smem_size)[slot];
    }
    registration.slot = slot;
#endif
    return registration;
}

__device__ __forceinline__ int nvshmemi_tma_claim_smem_registration() {
    unsigned long long *owners = nvshmemi_tma_smem_owner_keys();
    uint64_t key = nvshmemi_tma_registration_key();
    int slot = nvshmemi_tma_smem_registration_slot();
    if (key == 0 || owners == NULL || slot < 0) {
        return -1;
    }

    unsigned long long old = atomicCAS(owners + slot, 0ULL, key);
    return old == 0 || old == key ? slot : -1;
}

__device__ __forceinline__ uintptr_t nvshmemi_tma_smem_base() {
    int slot = nvshmemi_tma_find_smem_registration();
    if (slot < 0) {
        return 0;
    }
    return reinterpret_cast<volatile uintptr_t *>(nvshmemi_device_state_d.tma_smem_bases)[slot];
}

__device__ __forceinline__ size_t nvshmemi_tma_smem_size() {
    int slot = nvshmemi_tma_find_smem_registration();
    if (slot < 0 || nvshmemi_device_state_d.tma_smem_size == NULL) {
        return 0;
    }
    return reinterpret_cast<volatile size_t *>(nvshmemi_device_state_d.tma_smem_size)[slot];
}

__device__ __forceinline__ void nvshmemi_tma_publish_smem_registration(int slot, uintptr_t base,
                                                                       size_t size) {
    nvshmemi_device_state_d.tma_smem_size[slot] = size;
    nvshmemi_device_state_d.tma_smem_bases[slot] = base;
    __threadfence();
}

__device__ __forceinline__ void nvshmemi_tma_release_smem_registration(int slot) {
    unsigned long long *owners = nvshmemi_tma_smem_owner_keys();
    nvshmemi_device_state_d.tma_smem_bases[slot] = 0;
    nvshmemi_device_state_d.tma_smem_size[slot] = 0;
    __threadfence();
    atomicExch(owners + slot, 0ULL);
}

/*
 * Returns true if this CTA has registered shared memory for TMA (via
 * nvshmemx_give_smem) and TMA policy is not DISABLE.  Used to gate the
 * TMA dispatch path in put, get, and quiet.
 */
__device__ __forceinline__ bool nvshmemi_tma_smem_registered() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (nvshmemi_device_state_d.tma_policy == NVSHMEMX_TMA_DISABLE) {
        return false;
    }
    return nvshmemi_tma_smem_base() != 0;
#else
    return false;
#endif
}

/*
 * Drain all pending TMA bulk ops issued by this thread.  Equivalent to
 * commit_group + wait_group 0, gated on this CTA having registered smem.
 * No-op if TMA is not in use.  Used by fence and quiet to enforce ordering
 * of TMA-initiated puts and completion of TMA-initiated gets.
 */
__device__ __forceinline__ void nvshmemi_tma_drain_if_registered() {
    if (nvshmemi_tma_smem_registered()) {
        nvshmemi_tma_bulk_commit_group();
        nvshmemi_tma_bulk_wait_group_0();
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
/*
 * Access a barrier slot in this CTA's registered smem.  Layout is a static
 * carve at the base of give_smem's buffer: NVSHMEMI_TMA_NUM_BARRIER_SLOTS
 * slots of 16 bytes each.  Each 16B slot holds one 8B mbarrier with 8B
 * padding to keep slot stride aligned with cp.async.bulk requirements.
 *
 * Slot assignments (must not collide across concurrently-running TMA paths):
 *   0             : single-thread impl mbarrier
 *   0, 1          : block impl ready_bar[0], ready_bar[1]
 *   2, 3          : block impl done_bar[0], done_bar[1]
 *   4             : direct global-to-shared get mbarrier
 *   5..31         : reserved for future TMA paths (warpgroup, deeper pipes,
 *                   reductions, etc.)
 *
 * Precondition: nvshmemi_tma_smem_registered() returns true (i.e. give_smem
 * succeeded with at least NVSHMEMI_TMA_BARRIER_REGION_BYTES).
 */
__device__ __forceinline__ char *nvshmemi_tma_barrier_region(uintptr_t smem_base) {
    return reinterpret_cast<char *>(smem_base);
}

__device__ __forceinline__ char *nvshmemi_tma_data_buffer(uintptr_t smem_base) {
    return reinterpret_cast<char *>(smem_base + (uintptr_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET);
}

__device__ __forceinline__ size_t nvshmemi_smem_data_buf_size(
    const nvshmemi_tma_smem_registration_t &registration, size_t num_buffers) {
    constexpr size_t kReserve = (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    if (num_buffers == 0 || registration.size <= kReserve) {
        return 0;
    }
    return nvshmemi_tma_align_down_16((registration.size - kReserve) / num_buffers);
}

__device__ __forceinline__ size_t nvshmemi_smem_data_buf_size(size_t num_buffers) {
    size_t smem_size = nvshmemi_tma_smem_size();
    constexpr size_t kReserve = (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    if (num_buffers == 0 || smem_size <= kReserve) {
        return 0;
    }
    return nvshmemi_tma_align_down_16((smem_size - kReserve) / num_buffers);
}

__device__ __forceinline__ uint64_t *nvshmemi_tma_barrier_slot(uintptr_t smem_base, int slot) {
    return reinterpret_cast<uint64_t *>(nvshmemi_tma_barrier_region(smem_base) +
                                        (uintptr_t)slot * 16);
}

__device__ __forceinline__ uint64_t *nvshmemi_tma_barrier_slot(int slot) {
    uintptr_t base = nvshmemi_tma_smem_base();
    return nvshmemi_tma_barrier_slot(base, slot);
}

#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
__device__ __forceinline__ uint8_t *nvshmemi_handle_thread_smem_slot(uintptr_t smem_base,
                                                                     uint32_t thread_idx) {
    return reinterpret_cast<uint8_t *>(nvshmemi_tma_data_buffer(smem_base) +
                                       static_cast<size_t>(thread_idx) *
                                           NVSHMEMI_HANDLE_THREAD_SMEM_STRIDE);
}

__device__ __forceinline__ handle_barrier_t *nvshmemi_handle_barrier_slot(uintptr_t smem_base,
                                                                          int slot) {
    assert(slot >= 0);
    assert(slot < NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS);
    return reinterpret_cast<handle_barrier_t *>(nvshmemi_tma_barrier_region(smem_base) +
                                                (uintptr_t)NVSHMEMI_TMA_BARRIER_REGION_BYTES +
                                                (uintptr_t)slot * 16);
}

__device__ __forceinline__ handle_barrier_t *nvshmemi_handle_barrier_slot(int slot) {
    uintptr_t base = nvshmemi_tma_smem_base();
    return nvshmemi_handle_barrier_slot(base, slot);
}

__device__ __forceinline__ char *nvshmemi_counted_state_region(uintptr_t smem_base) {
    return nvshmemi_tma_barrier_region(smem_base) + (uintptr_t)NVSHMEMI_TMA_BARRIER_REGION_BYTES +
           (uintptr_t)NVSHMEMI_HANDLE_BARRIER_BYTES;
}

__device__ __forceinline__ char *nvshmemi_counted_state_slot(uintptr_t smem_base, int slot) {
    assert(slot >= 0);
    assert(slot < NVSHMEMI_COUNTED_NUM_STATE_SLOTS);
    return nvshmemi_counted_state_region(smem_base) +
           (uintptr_t)slot * NVSHMEMI_COUNTED_STATE_SLOT_BYTES;
}

__device__ __forceinline__ void nvshmemi_drain_handle_slot_if_owned(uintptr_t smem_base,
                                                                    uint32_t slot, uint32_t owner) {
    if (slot >= NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS) {
        return;
    }

    handle_barrier_t *barrier = nvshmemi_handle_barrier_slot(smem_base, (int)slot);
    if (barrier->pending_handle_is_owned_by(owner)) {
        barrier->drain_pending_handle(true);
    }
}

/* Complete every pending handle operation in this CTA.  A thread-scoped
 * nvshmem_quiet() may be called by one elected thread after the issuing
 * threads have synchronized, so that caller must not restrict the drain to
 * its own physical warp. */
__device__ __forceinline__ void nvshmemi_handle_quiet_all() {
    const auto registration = nvshmemi_tma_get_smem_registration();
    if (!registration.is_valid()) {
        return;
    }

    uintptr_t smem_base = registration.base;
    for (uint32_t slot = 0; slot < NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS; slot++) {
        handle_barrier_t *barrier = nvshmemi_handle_barrier_slot(smem_base, (int)slot);
        if (barrier->has_pending_handle()) {
            barrier->drain_pending_handle(true);
        }
    }
}

/* Complete deferred handle operations issued by the calling logical threadgroup.
 * Owners are physical warp leaders: warp scope uses its physical warp,
 * warpgroup scope uses its first physical warp, and block scope uses warp 0. */
__device__ __forceinline__ void nvshmemi_handle_quiet_owned() {
    const auto registration = nvshmemi_tma_get_smem_registration();
    if (!registration.is_valid()) {
        return;
    }

    uint32_t tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    if ((tid % warpSize) != 0) {
        return;
    }
    uint32_t warp_idx_in_block = tid / warpSize;

    uintptr_t smem_base = registration.base;

    /* Warp and block threadgroups have compact threadgroup indices that match
     * their physical owner-warp mapping.  Block scope is owned by physical
     * warp 0, and its two handle barriers occupy slots 0 and 1. */
    uint32_t warp_slot_base = warp_idx_in_block * TMA_COPY_NUM_STAGES;
    for (uint32_t i = 0; i < TMA_COPY_NUM_STAGES; i++) {
        nvshmemi_drain_handle_slot_if_owned(smem_base, warp_slot_base + i, warp_idx_in_block);
    }

    /* Warpgroup paths allocate two compact handle slots per warpgroup instead
     * of using the physical-warp slot range.  Only the first physical warp in
     * a warpgroup owns these slots. */
    uint32_t warps_per_warpgroup =
        nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARPGROUP>() / warpSize;
    if ((warp_idx_in_block % warps_per_warpgroup) == 0) {
        uint32_t warpgroup_idx_in_block = warp_idx_in_block / warps_per_warpgroup;
        uint32_t warpgroup_slot_base = warpgroup_idx_in_block * TMA_COPY_NUM_STAGES;

        // Warpgroup 0 shares slots 0 and 1 with the physical-warp mapping above.
        if (warpgroup_slot_base != warp_slot_base) {
            for (uint32_t i = 0; i < TMA_COPY_NUM_STAGES; i++) {
                nvshmemi_drain_handle_slot_if_owned(smem_base, warpgroup_slot_base + i,
                                                    warp_idx_in_block);
            }
        }
    }
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ void nvshmemi_handle_quiet() {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        nvshmemi_handle_quiet_all();
    } else {
        nvshmemi_handle_quiet_owned();
    }
}
#endif

enum class Blocking { No, Yes };
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

/*
 * Direct global-to-shared TMA copy for get operations whose local destination
 * is shared memory.  The source pointer is the peer-mapped remote global
 * address.  The destination is local shared memory.
 *
 * Returns 0 on success; -1 if alignment, size, smem registration, or reserved
 * mbarrier-region constraints are not met.  Callers fall back to regular P2P
 * loads when this helper returns -1.
 */
template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_shared(
    const nvshmemi_tma_smem_registration_t &registration, void *smem_dst, const void *gmem_src,
    size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)smem_dst)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_src)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned(bytes)) {
        return -1;
    }
    if (bytes > (size_t)UINT32_MAX) {
        return -1;
    }

    uintptr_t base = registration.base;
    size_t smem_size = registration.size;
    constexpr size_t kReserve = (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    if (base == 0 || smem_size <= kReserve) {
        return -1;
    }

    uintptr_t dst_start = (uintptr_t)smem_dst;
    uintptr_t dst_end = dst_start + bytes;
    uintptr_t reserve_end = base + kReserve;
    if (dst_start < reserve_end && dst_end > base) {
        return -1;
    }

    uint64_t *mbar = nvshmemi_tma_barrier_slot(base, 4);
    bool is_leader = (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
                         ? true
                         : ((SCOPE == NVSHMEMI_THREADGROUP_WARP) ? nvshmemi_tma_elect_warp()
                                                                 : nvshmemi_tma_block_is_elected());

    if (is_leader) {
        nvshmemi_tma_mbarrier_init(mbar);
        nvshmemi_tma_fence_proxy_async_shared_cta();
        nvshmemi_tma_mbarrier_arrive_expect_tx(mbar, (uint32_t)bytes);
        nvshmemi_tma_bulk_global_to_shared(smem_dst, gmem_src, (uint32_t)bytes, mbar);
        nvshmemi_tma_mbarrier_try_wait(mbar, 0);
    }

    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        __syncthreads();
    } else if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        nvshmemi_threadgroup_sync<SCOPE>();
    }
    return 0;
}

/*
 * Single-issuer global-to-global TMA copy for THREAD and WARP scope.
 *
 * The issuer stages each chunk through this CTA's registered shared-memory
 * tile.  The inbound global-to-shared copy completes through the mbarrier;
 * after the barrier wait, the same tile is issued as an outbound
 * shared-to-global TMA copy.  Before reusing the tile for the next chunk, the
 * issuer waits until the outbound operation has finished reading shared
 * memory.  Blocking calls also wait for the outbound write to complete and
 * publish that completion with a system fence.
 *
 * WARP scope still uses only one issuer; the remaining lanes wait at the end
 * so callers observe normal warp-scoped completion semantics.
 *
 * Returns 0 on success; -1 if alignment, size, or smem registration fails.
 */
template <threadgroup_t SCOPE, Blocking BLOCKING>
__device__ inline int nvshmemi_memcpy_tma_global_global_single(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *gmem_src,
    size_t bytes) {
    static_assert(SCOPE == NVSHMEMI_THREADGROUP_THREAD || SCOPE == NVSHMEMI_THREADGROUP_WARP,
                  "single impl is only for THREAD or WARP scope");
    if (bytes == 0) {
        return 0;
    }
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_dst)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_src)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned(bytes)) {
        return -1;
    }

    uintptr_t base = registration.base;
    size_t smem_size = registration.size;
    if (base == 0 || smem_size == 0) {
        return -1;
    }

    /* Barrier regions are reserved at the base by give_smem; data tile is the
     * remainder.  This impl uses TMA slot 0. */
    constexpr size_t kReserve = (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    if (smem_size <= kReserve) {
        return -1;
    }
    uint64_t *mbar = nvshmemi_tma_barrier_slot(base, 0);
    char *data_buf = nvshmemi_tma_data_buffer(base);
    size_t tile_size = nvshmemi_tma_align_down_16(smem_size - kReserve);
    if (tile_size == 0) {
        return -1;
    }
    /* CTA shared memory is far below 4 GiB; TMA byte counts are 32-bit. */
    const uint32_t tile = (uint32_t)tile_size;

    bool is_leader = (SCOPE == NVSHMEMI_THREADGROUP_THREAD) ? true : nvshmemi_tma_elect_warp();

    if (is_leader) {
        const char *src = (const char *)gmem_src;
        char *dst = (char *)gmem_dst;
        size_t remaining = bytes;

        nvshmemi_tma_mbarrier_init(mbar);
        /* Make mbarrier init visible to the async proxy before cp.async.bulk. */
        nvshmemi_tma_fence_proxy_async_shared_cta();
        int phase = 0;

        while (remaining > 0) {
            uint32_t this_chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;

            /* Inbound TMA: arrive + expect_tx, issue load, wait for completion. */
            nvshmemi_tma_mbarrier_arrive_expect_tx(mbar, this_chunk);
            nvshmemi_tma_bulk_global_to_shared(data_buf, src, this_chunk, mbar);
            nvshmemi_tma_mbarrier_try_wait(mbar, phase);
            phase ^= 1;

            /* Outbound TMA: smem -> remote gmem.  No fence needed between
             * inbound and outbound TMA — both use the async proxy and the
             * mbarrier release orders them. */
            unsigned int data_addr = nvshmemi_tma_cvta_to_shared(data_buf);
            nvshmemi_tma_bulk_shared_to_global(dst, data_addr, this_chunk);
            nvshmemi_tma_bulk_commit_group();

            remaining -= this_chunk;
            src += this_chunk;
            dst += this_chunk;

            /* Wait for the outbound smem read before reusing or releasing the staging tile. */
            nvshmemi_tma_bulk_wait_group_read_0();
        }
        if constexpr (BLOCKING == Blocking::Yes) {
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
    }

    if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        nvshmemi_threadgroup_sync<SCOPE>();
    }
    return 0;
}

/*
 * nvshmemi_memcpy_tma_global_global_block - Block-scoped, warp-specialized
 * DOUBLE-BUFFERED local-gmem to remote-gmem TMA put.
 *
 * Requires at least two full warps (2 * warpSize threads).  Thread- and
 * warp-scoped primitives must route to nvshmemi_memcpy_tma_global_global_single;
 * only block-scoped primitives may use this helper.
 *
 * Combines warp specialization (load warp vs store warp) with smem
 * double-buffering to overlap inbound TMA of buffer N+1 with outbound TMA of
 * buffer N.
 *
 * Smem layout: [NVSHMEMI_SMEM_DATA_REGION_OFFSET reserved][buf0: tile][buf1: tile]
 * where tile = (smem_size - NVSHMEMI_SMEM_DATA_REGION_OFFSET) / 2, 16B-aligned.
 *
 *   ready_bar[i]: load warp signals "buf[i] ready" via cp.async.bulk
 *                 complete_tx + arrive_expect_tx.  Store warp try_waits.
 *   done_bar[i]:  store warp signals "outbound read of buf[i] done, safe
 *                 to reuse" via arrive_expect_tx(1) + complete_tx(1).
 *                 Load warp try_waits before overwriting buf[i].
 *
 * Phase flips every full pipe cycle (2 iters): `phase = (i / 2) & 1`.
 * Buffer index: `slot = i & 1`.  Load waits done_bar[slot] at phase^1,
 * store waits ready_bar[slot] at phase.  No __syncthreads in hot loop.
 *
 * Returns 0 on success; -1 on alignment/size/registration failure.
 */
template <Blocking BLOCKING>
__device__ int nvshmemi_memcpy_tma_global_global_block(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *gmem_src,
    size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    /* These are TMA routing constraints, not put API constraints.  The caller
     * falls back to regular P2P stores when this helper returns -1. */
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_dst)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_src)) {
        return -1;
    }
    if (!nvshmemi_tma_is_16b_aligned(bytes)) {
        return -1;
    }

    uintptr_t base = registration.base;
    size_t smem_size = registration.size;
    if (base == 0 || smem_size == 0) {
        return -1;
    }

    /* Barrier region reserved at the base by give_smem.  This impl uses slots
     * 0,1 for ready_bar[0,1] and slots 2,3 for done_bar[0,1].  Data tiles
     * occupy the remainder, split in half. */
    constexpr size_t kReserve = (size_t)NVSHMEMI_SMEM_DATA_REGION_OFFSET;
    if (smem_size <= kReserve) {
        return -1;
    }
    size_t tile_size = nvshmemi_tma_align_down_16((smem_size - kReserve) / 2);
    if (tile_size == 0) {
        return -1;
    }
    /* CTA shared memory is far below 4 GiB; TMA byte counts are 32-bit. */
    const uint32_t tile = (uint32_t)tile_size;

    unsigned int block_threads = blockDim.x * blockDim.y * blockDim.z;
    if (block_threads < 2 * warpSize) {
        return -1;
    }

    unsigned int tid =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    /* CUDA forms warps from the linear CTA rank, with threadIdx.x varying
     * fastest. These fixed ranks are warp 0 lane 0 and warp 1 lane 0 for any
     * CTA shape with at least two full warps. */
    bool is_load = (tid == 0);
    bool is_store = (tid == warpSize);

    /* Init all 4 barriers once.  Fence so async proxy sees init before any
     * cp.async.bulk arrives. */
    if (is_load) {
        nvshmemi_tma_mbarrier_init(nvshmemi_tma_barrier_slot(base, 0));
        nvshmemi_tma_mbarrier_init(nvshmemi_tma_barrier_slot(base, 1));
        nvshmemi_tma_mbarrier_init(nvshmemi_tma_barrier_slot(base, 2));
        nvshmemi_tma_mbarrier_init(nvshmemi_tma_barrier_slot(base, 3));
        nvshmemi_tma_fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (is_load) {
        uint64_t *ready0 = nvshmemi_tma_barrier_slot(base, 0);
        uint64_t *ready1 = nvshmemi_tma_barrier_slot(base, 1);
        uint64_t *done0 = nvshmemi_tma_barrier_slot(base, 2);
        uint64_t *done1 = nvshmemi_tma_barrier_slot(base, 3);
        char *buf0 = nvshmemi_tma_data_buffer(base);
        char *buf1 = buf0 + tile;
        const char *src = (const char *)gmem_src;
        size_t remaining = bytes;

        for (size_t i = 0; remaining > 0; i++) {
            int slot = (int)(i & 1);
            int phase = (int)((i >> 1) & 1);
            uint32_t chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;
            uint64_t *ready = slot ? ready1 : ready0;
            uint64_t *done = slot ? done1 : done0;
            char *buf = slot ? buf1 : buf0;

            /* First 2 iters (i=0,1): each slot's done_bar is fresh (parity 0),
             * try_wait with phase^1=1 returns immediately.  After that the
             * store warp has flipped done_bar[slot] and we wait for the next
             * flip. */
            if (i >= 2) {
                nvshmemi_tma_mbarrier_try_wait(done, phase ^ 1);
            }
            nvshmemi_tma_bulk_global_to_shared(buf, src, chunk, ready);
            nvshmemi_tma_mbarrier_arrive_expect_tx(ready, chunk);

            src += chunk;
            remaining -= chunk;
        }
    } else if (is_store) {
        uint64_t *ready0 = nvshmemi_tma_barrier_slot(base, 0);
        uint64_t *ready1 = nvshmemi_tma_barrier_slot(base, 1);
        uint64_t *done0 = nvshmemi_tma_barrier_slot(base, 2);
        uint64_t *done1 = nvshmemi_tma_barrier_slot(base, 3);
        char *buf0 = nvshmemi_tma_data_buffer(base);
        char *buf1 = buf0 + tile;
        unsigned int data_addr0 = nvshmemi_tma_cvta_to_shared(buf0);
        unsigned int data_addr1 = nvshmemi_tma_cvta_to_shared(buf1);
        char *dst = (char *)gmem_dst;
        size_t remaining = bytes;

        for (size_t i = 0; remaining > 0; i++) {
            int slot = (int)(i & 1);
            int phase = (int)((i >> 1) & 1);
            uint32_t chunk = remaining < (size_t)tile ? (uint32_t)remaining : tile;
            uint64_t *ready = slot ? ready1 : ready0;
            uint64_t *done = slot ? done1 : done0;
            unsigned int data_addr = slot ? data_addr1 : data_addr0;

            nvshmemi_tma_mbarrier_try_wait(ready, phase);
            nvshmemi_tma_bulk_shared_to_global(dst, data_addr, chunk);
            nvshmemi_tma_bulk_commit_group();
            nvshmemi_tma_bulk_wait_group_read_0();
            nvshmemi_tma_mbarrier_arrive_expect_tx(done, 1);
            nvshmemi_tma_mbarrier_complete_tx(done, 1);

            dst += chunk;
            remaining -= chunk;
        }
    }

    if constexpr (BLOCKING == Blocking::Yes) {
        if (is_store) {
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
    }
    __syncthreads();
    return 0;
}

/*
 * Dispatcher: routes THREAD and WARP scope to the single-thread impl, BLOCK
 * scope to the double-buffered impl.
 */
template <threadgroup_t SCOPE, Blocking BLOCKING>
__device__ inline int nvshmemi_memcpy_tma_global_global(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *gmem_src,
    size_t bytes) {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        return nvshmemi_memcpy_tma_global_global_block<BLOCKING>(registration, gmem_dst, gmem_src,
                                                                 bytes);
    } else {
        return nvshmemi_memcpy_tma_global_global_single<SCOPE, BLOCKING>(registration, gmem_dst,
                                                                         gmem_src, bytes);
    }
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_global_nbi(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *gmem_src,
    size_t bytes) {
    return nvshmemi_memcpy_tma_global_global<SCOPE, Blocking::No>(registration, gmem_dst, gmem_src,
                                                                  bytes);
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_global(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *gmem_src,
    size_t bytes) {
    return nvshmemi_memcpy_tma_global_global<SCOPE, Blocking::Yes>(registration, gmem_dst, gmem_src,
                                                                   bytes);
}

#else /* non-sm90: compile-time fallback returning -1 so call sites can instantiate. */

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_global_nbi(
    const nvshmemi_tma_smem_registration_t & /*registration*/, void * /*gmem_dst*/,
    const void * /*gmem_src*/, size_t /*bytes*/) {
    return -1;
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_global(
    const nvshmemi_tma_smem_registration_t & /*registration*/, void * /*gmem_dst*/,
    const void * /*gmem_src*/, size_t /*bytes*/) {
    return -1;
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_global_shared(
    const nvshmemi_tma_smem_registration_t & /*registration*/, void * /*smem_dst*/,
    const void * /*gmem_src*/, size_t /*bytes*/) {
    return -1;
}

#endif /* __CUDA_ARCH__ >= 900 */

/*
 * Dispatcher:  TMA bulk copy based on source memory kind:
 *   - source in smem: direct TMA smem -> remote gmem
 *   - source in gmem: staged TMA gmem -> smem -> remote gmem
 * Returns the underlying rc (0 on success; -1 on alignment/size
 * mismatch or non-supported target).
 */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_memcpy_tma(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *source,
    size_t nbytes) {
    if (__isShared(source)) {
        return nvshmemi_memcpy_tma_shared_global<SCOPE>(gmem_dst, source, nbytes);
    } else {
        return nvshmemi_memcpy_tma_global_global<SCOPE>(registration, gmem_dst, source, nbytes);
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_memcpy_tma_nbi(
    const nvshmemi_tma_smem_registration_t &registration, void *gmem_dst, const void *source,
    size_t nbytes) {
    if (__isShared(source)) {
        return nvshmemi_memcpy_tma_shared_global_nbi<SCOPE>(gmem_dst, source, nbytes);
    }
    return nvshmemi_memcpy_tma_global_global_nbi<SCOPE>(registration, gmem_dst, source, nbytes);
}

/* qpair specific APIs */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_quiet(int pe = NVSHMEMX_PE_ALL,
                                                             nvshmemx_qp_handle_t *qp_handle = NULL,
                                                             int num_qps = NVSHMEMX_QP_ALL) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    /* Handle operations retain their SMEM completion barriers until quiet. */
    nvshmemi_handle_quiet<SCOPE>();
#endif

    /* Drain TMA unconditionally BEFORE the connectivity-based quiet.  In a
     * mixed topology (some peers via NVLink P2P, others via IB/transport),
     * TMA ops to P2P peers are not covered by nvshmemi_transfer_quiet, so the
     * drain must not be gated on job_connectivity. */
    nvshmemi_tma_drain_if_registered();

    if (!nvshmemi_use_ldst_path()) {
        /* Network path: Require membar.sys to ensure P2P store visibility. (use_membar = true). */
        nvshmemi_transfer_quiet<SCOPE>(true, pe, qp_handle, num_qps);
    } else {
        /* __threadfence_system is required for P2P store visibility. */
        if (!myIdx) {
            __threadfence_system();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps);
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_WARP>(int pe,
                                                                   nvshmemx_qp_handle_t *qp_handle,
                                                                   int num_qps);
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>(int pe,
                                                                    nvshmemx_qp_handle_t *qp_handle,
                                                                    int num_qps);

/*
 * nvshmemi_flush - Wait until all source buffers used by preceding
 * non-blocking puts issued from the flushing thread are safe to reuse.
 *
 * Guarantees reusability only.  Does NOT guarantee that the data is
 * visible at the remote PE.  Callers who need remote visibility must still
 * use nvshmemi_quiet() / nvshmem_quiet().
 *
 * Transport behaviour:
 *   NVLink via TMA: commit the pending bulk group, cp.async.bulk.wait_group.read 0
 *     to stall until the TMA hardware has consumed the source smem, then
 *     fence.proxy.async.shared::cta to order that drain against subsequent
 *     generic accesses to the smem buffer.  Cheaper than membar.sys because
 *     we do not need remote visibility, only source-buffer reusability.
 *   NVLink via st.global (P2P): no-op - st.global puts block until the store
 *     enters the memory subsystem, so the source is already consumed.  The
 *     nvshmemi_tma_*() helpers below become no-ops when no TMA has been
 *     issued by this thread, so they are safe to invoke unconditionally.
 *   Network (IB/RoCE, EFA, proxy): drain send-side completions via
 *     nvshmemi_transfer_quiet(use_membar=false) - no __threadfence_system.
 */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_flush(int pe = NVSHMEMX_PE_ALL,
                                                             nvshmemx_qp_handle_t *qp_handle = NULL,
                                                             int num_qps = NVSHMEMX_QP_ALL) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    nvshmemi_threadgroup_sync<SCOPE>();
    /* TMA source-reuse drain.  Gated on smem registration so only CTAs that
     * actually issued TMA ops pay the wait_group.read cost; other CTAs skip
     * all three PTX instructions. */
    if (!myIdx) {
        if (nvshmemi_tma_smem_registered()) {
            nvshmemi_tma_bulk_commit_group();
            nvshmemi_tma_bulk_wait_group_read_0();
            nvshmemi_tma_fence_proxy_async_shared_cta();
        }
        if (!nvshmemi_use_ldst_path()) {
            /* Network path: drain send-side completions without issuing
             * __threadfence_system() (use_membar = false). */
            nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(false, pe, qp_handle, num_qps);
        }
    }
    /* P2P st.global path: no-op - source buffer already consumed. */
    nvshmemi_threadgroup_sync<SCOPE>();
}

template __device__ void nvshmemi_flush<NVSHMEMI_THREADGROUP_THREAD>(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps);
template __device__ void nvshmemi_flush<NVSHMEMI_THREADGROUP_WARP>(int pe,
                                                                   nvshmemx_qp_handle_t *qp_handle,
                                                                   int num_qps);
template __device__ void nvshmemi_flush<NVSHMEMI_THREADGROUP_BLOCK>(int pe,
                                                                    nvshmemx_qp_handle_t *qp_handle,
                                                                    int num_qps);

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_fence(int pe = NVSHMEMX_PE_ALL,
                                                             nvshmemx_qp_handle_t *qp_handle = NULL,
                                                             int num_qps = NVSHMEMX_QP_ALL) {
    /* Drain prior TMA-initiated puts.  Like nvshmemi_quiet, this must not be
     * gated on job_connectivity — in a mixed topology (some peers via NVLink
     * P2P + TMA, others via network) a user calling fence after a TMA put
     * expects the TMA op ordered relative to subsequent ops. */
    nvshmemi_tma_drain_if_registered();
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if (nvshmemi_tma_smem_registered()) {
        /* Ordering fabric operations require the earlier one to be completed before the next
         * one can be issued.
         * Draining pending operations here
         */
        nvshmemi_handle_quiet<SCOPE>();

        /* Order outstanding fabric PUTs without waiting for their completion;
         * completion remains the responsibility of nvshmem_quiet(). */
        fence_proxy_fabric2fabric_alias();
    }
#endif
    if (!nvshmemi_use_ldst_path()) {
        nvshmemi_transfer_fence<SCOPE>(pe, qp_handle, num_qps);
    }
    __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                               for data visibility in case of intra-node GPU transfers */
}

template __device__ void nvshmemi_fence<NVSHMEMI_THREADGROUP_THREAD>(
    int pe, nvshmemx_qp_handle_t *qp_handle, int num_qps);
template __device__ void nvshmemi_fence<NVSHMEMI_THREADGROUP_WARP>(int pe,
                                                                   nvshmemx_qp_handle_t *qp_handle,
                                                                   int num_qps);
template __device__ void nvshmemi_fence<NVSHMEMI_THREADGROUP_BLOCK>(int pe,
                                                                    nvshmemx_qp_handle_t *qp_handle,
                                                                    int num_qps);

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_g(const T *source, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr)) {
        T *source_actual = (T *)((char *)(peer_base_addr) +
                                 ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));
        return *source_actual;
    } else {
        return nvshmemi_transfer_rma_g<T>((void *)source, pe, qp_index);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_get_nbi(
    T *dest, const T *source, size_t nelems, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized<true>(pe, nelems * sizeof(T), SCOPE, source,
                                                        dest)) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        size_t nbytes = nelems * sizeof(T);
        const auto registration = nvshmemi_tma_get_smem_registration();
        if (registration.is_valid()) {
            if (__isShared(dest)) {
                if (nvshmemi_memcpy_tma_global_shared<SCOPE>(
                        registration, (void *)dest, (const void *)source_actual, nbytes) == 0) {
                    return;
                }
            } else {
                if (nvshmemi_memcpy_tma_global_global<SCOPE>(
                        registration, (void *)dest, (const void *)source_actual, nbytes) == 0) {
                    return;
                }
            }
        }
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual, nbytes);
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented<true>(pe, nelems * sizeof(T), SCOPE, source, dest)) {
        nvshmemi_handle_get<SCOPE>(source, dest, nelems * sizeof(T), pe, false);
#endif
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                          nelems * sizeof(T), pe, qp_index);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_get(
    T *dest, const T *source, size_t nelems, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized<true>(pe, nelems * sizeof(T), SCOPE, source,
                                                        dest)) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        size_t nbytes = nelems * sizeof(T);
        const auto registration = nvshmemi_tma_get_smem_registration();
        if (registration.is_valid()) {
            if (__isShared(dest)) {
                if (nvshmemi_memcpy_tma_global_shared<SCOPE>(
                        registration, (void *)dest, (const void *)source_actual, nbytes) == 0) {
                    nvshmemi_threadgroup_sync<SCOPE>();
                    return;
                }
            } else {
                if (nvshmemi_memcpy_tma_global_global<SCOPE>(
                        registration, (void *)dest, (const void *)source_actual, nbytes) == 0) {
                    nvshmemi_threadgroup_sync<SCOPE>();
                    return;
                }
            }
        }
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual, nbytes);
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented<true>(pe, nelems * sizeof(T), SCOPE, source, dest)) {
        nvshmemi_handle_get<SCOPE>(source, dest, nelems * sizeof(T), pe, true);
#endif
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                      nelems * sizeof(T), pe, qp_index);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_p(
    T *dest, const T value, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) && !nvshmemi_is_le_supported_and_prioritized(pe)) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented(pe)) {
        if (pe == nvshmemi_device_state_d.mype) {
            *dest = value;
        } else {
            nvshmemi_handle_p<T>((void *)dest, value, pe);
        }
#endif
    } else {
        nvshmemi_transfer_rma_p<T>((void *)dest, value, pe, qp_index);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemii_put_nbi(
    T *dest, const T *source, size_t nelems, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized<false>(pe, nelems * sizeof(T), SCOPE, dest,
                                                         source)) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        size_t nbytes = nelems * sizeof(T);
        /* TMA fast path when this CTA registered smem.  Fall through to P2P
         * stores on alignment/size/registration failure. */
        const auto registration = nvshmemi_tma_get_smem_registration();
        if (registration.is_valid() &&
            nvshmemi_memcpy_tma_nbi<SCOPE>(registration, (void *)dest_actual, (const void *)source,
                                           nbytes) == 0) {
            return;
        }
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source, nbytes);
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented<false>(pe, nelems * sizeof(T), SCOPE, dest, source)) {
        nvshmemi_handle_put<SCOPE>(source, dest, nelems * sizeof(T), pe, false);
#endif
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                          nelems * sizeof(T), pe, qp_index);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_put_nbi(
    T *dest, const T *source, size_t nelems, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_nbi<T, SCOPE>(dest, source, nelems, pe, qp_index);
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_put(
    T *dest, const T *source, size_t nelems, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized<false>(pe, nelems * sizeof(T), SCOPE, dest,
                                                         source)) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        size_t nbytes = nelems * sizeof(T);
        /* TMA when this CTA registered smem:
         * Both variants include internal syncs; the trailing sync below is
         * redundant in the TMA path but harmless. */
        const auto registration = nvshmemi_tma_get_smem_registration();
        if (registration.is_valid() &&
            nvshmemi_memcpy_tma<SCOPE>(registration, (void *)dest_actual, (const void *)source,
                                       nbytes) == 0) {
            nvshmemi_threadgroup_sync<SCOPE>();
            return;
        }
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source, nbytes);
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented<false>(pe, nelems * sizeof(T), SCOPE, dest, source)) {
        nvshmemi_handle_put<SCOPE>((void *)source, (void *)dest, nelems * sizeof(T), pe, true);
#endif
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                      nelems * sizeof(T), pe, qp_index);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_signal_op(
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const bool can_use_handle = nvshmemi_is_le_implemented(pe);
#endif
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const bool can_use_atomic_handle = nvshmemi_is_le_atomic_implemented(pe, sig_addr);
#else
    const bool can_use_atomic_handle = false;
#endif
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET && nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized(pe)) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = signal;
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (sig_op == NVSHMEMI_AMO_SIGNAL_SET && can_use_handle) {
        nvshmemi_handle_p((void *)sig_addr, signal, pe);
#endif
    } else if (sig_op == NVSHMEMI_AMO_SIGNAL_ADD && nvshmemi_use_ldst_path() &&
               nvshmemi_peer_reachable(peer_base_addr) &&
               (!can_use_atomic_handle || !nvshmemi_is_le_supported_and_prioritized(pe))) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        /* sig_op == NVSHMEM_SIGNAL_ADD */
        atomicAdd_system((unsigned long long *)dest_actual, signal);
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (sig_op == NVSHMEMI_AMO_SIGNAL_ADD && can_use_atomic_handle) {
        nvshmemi_handle_atomic_nonfetch<uint64_t, NVSHMEMI_AMO_SIGNAL_ADD>(sig_addr, signal, pe);
#endif
    } else {
        nvshmemi_transfer_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe,
                                                 (nvshmemi_amo_t)sig_op, qp_index);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemii_put_signal(
    T *dest, const T *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op,
    int pe, bool is_nbi, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_peer_reachable(peer_base_addr) &&
        !nvshmemi_is_le_supported_and_prioritized<false>(pe, nelems * sizeof(T), SCOPE, dest,
                                                         source)) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        size_t nbytes = nelems * sizeof(T);
        /* TMA fast path when this CTA registered smem.
         * If the put route is via TMA, we need to ensure completion before we can issue the
         * signal_op, so we use a blocking TMA put here.
         */
        const auto registration = nvshmemi_tma_get_smem_registration();
        if (registration.is_valid() &&
            nvshmemi_memcpy_tma<SCOPE>(registration, (void *)dest_actual, (const void *)source,
                                       nbytes) == 0) {
            nvshmemi_threadgroup_sync<SCOPE>();
            if (!myIdx) {
                __threadfence_system();
                nvshmemi_signal_op(sig_addr, signal, sig_op, pe, qp_index);
            }
            return;
        }
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source, nbytes);
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myIdx) {
            __threadfence_system();
            nvshmemi_signal_op(sig_addr, signal, sig_op, pe, qp_index);
        }
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    } else if (nvshmemi_is_le_implemented<false>(pe, nelems * sizeof(T), SCOPE, dest, source)) {
        nvshmemi_handle_put<SCOPE>(source, dest, nelems * sizeof(T), pe, true);
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myIdx) {
            __threadfence_system();
            // Ensure that signal operation is ordered with respect
            // to earlier fabric / generic operations
            fence_proxy_fabric2fabric_alias();
            fence_proxy_generic2fabric_alias();

            nvshmemi_signal_op(sig_addr, signal, sig_op, pe, qp_index);
        }
#endif
    } else {
        nvshmemi_transfer_put_signal<SCOPE>((void *)dest, (void *)source, nelems * sizeof(T),
                                            (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe,
                                            is_nbi, qp_index);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_put_signal(
    T *dest, const T *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op,
    int pe, bool is_nbi, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_signal<T, SCOPE>(dest, source, nelems, sig_addr, signal, sig_op, pe, is_nbi,
                                   qp_index);
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t nvshmemi_heap_offset(const void *address) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address) -
                                 reinterpret_cast<uintptr_t>(nvshmemi_device_state_d.heap_base));
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_ptr_range_overflows(const void *ptr,
                                                                           size_t bytes) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    return bytes > cuda::std::numeric_limits<uintptr_t>::max() - address;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void *nvshmemi_mc_ptr(nvshmemi_team_t *team,
                                                               const void *ptr) {
    if (team == NULL || team->nvls_rsc_base_ptr == NULL) {
        return NULL;
    }
    ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;
    if (ptr >= nvshmemi_device_state_d.heap_base && offset < nvshmemi_device_state_d.heap_size &&
        team->nvls_rsc_base_ptr != NULL) {
        void *mc_addr = (void *)__ldg((const long long unsigned *)team->nvls_rsc_base_ptr);
        if (mc_addr != NULL) {
            mc_addr = (void *)((char *)mc_addr + offset);
        }
        return mc_addr;
    } else {
        return NULL;
    }
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void *nvshmemi_ptr(const void *ptr, int pe) {
    if (pe >= 0 && pe < nvshmemi_device_state_d.npes && ptr >= nvshmemi_device_state_d.heap_base) {
        ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;

        if (offset < nvshmemi_device_state_d.heap_size) {
            void *peer_addr = (void *)__ldg(
                (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
            if (peer_addr != NULL) {
                peer_addr = (void *)((char *)peer_addr + offset);
            }
            return peer_addr;
        }
    }
    return NULL;
}

template <typename T, int UNROLL>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_store_128b_register(
    uint64_t dest_regs[2 * UNROLL], uint64_t ll_flag, uint64_t ll_flag_mask, T *src,
    uint8_t unroll_stride) {
    union {
        uint64_t regs8[2];
        uint32_t regs4[4];
    };
    int myIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    /* in both 16 and 8 byte cases, the flag thread needs to write two 8 byte chunks. */

#pragma unroll
    for (int unroll = 0; unroll < UNROLL; unroll++) {
        if (sizeof(T) >= 8) {
            uint64_t *src8 = (uint64_t *)(src + unroll * unroll_stride);
            asm("ld.b64 %0, [%1];" : "=l"(regs8[0]) : "l"(src8));
            asm("ld.b64 %0, [%1];" : "=l"(regs8[1]) : "l"(src8 + 1));
        } else if (sizeof(T) == 4) {
            uint32_t *src4 = (uint32_t *)(src + unroll * unroll_stride);
            asm("ld.b32 %0, [%1];" : "=r"(regs4[0]) : "l"(src4));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[1]) : "l"(src4 + 1));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[2]) : "l"(src4 + 2));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[3]) : "l"(src4 + 3));
        } else {
            uint32_t lower, upper;
            /* funnelshift_r will use shift to select the
             * proper four bytes depending on the alignment.
             * 0x10 = 2, 0x11 = 3, 0x01 = 1
             */
            uint8_t shift = (uintptr_t)src % 4;
            /* We need to align the buffer to the nearest 4 bytes
             * Shift down 2 bytes in the case of 2B alignment down
             * 1B in the case of 1B alignment. Shifting down is always
             * appropriate due to 256 B alignment of cudaMalloc().
             */
            uint32_t *src4 =
                (uint32_t *)((uintptr_t)(src + unroll * unroll_stride) & -(uintptr_t)4);
            /* 2 Byte aligned - 2B in lower, 2B in upper */
            /* 1 Byte aligned (0x01) - 1B in upper, 3B in lower */
            /* 1 Byte aligned (0x11) - 3B in upper, 1B in lower */
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 1));
            regs4[0] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 1));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 2));
            regs4[1] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 2));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 3));
            regs4[2] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 4));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 5));
            regs4[3] = __funnelshift_r(lower, upper, 8 * shift);
        }
        dest_regs[unroll * 2] = regs8[0];
        dest_regs[unroll * 2 + 1] = (regs8[1] & ll_flag) | ll_flag_mask;
    }
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_store_varlen_register(
    uint64_t dest_regs[2], uint64_t ll_flag, uint64_t ll_flag_mask, void *src, uint8_t nelems) {
    uint8_t i;
    union {
        uint64_t regs8[2];
        uint32_t regs4[4];
    };

    /* in both 16 and 8 byte cases, the flag thread needs to write two 8 byte chunks. */
    if (sizeof(T) >= 8) {
        uint64_t *src8 = (uint64_t *)src;
        for (i = 0; i < nelems; i++) {
            asm("ld.b64 %0, [%1];" : "=l"(regs8[i]) : "l"(src8 + i));
        }
    } else if (sizeof(T) == 4) {
        uint32_t *src4 = (uint32_t *)src;
        for (i = 0; i < nelems; i++) {
            asm("ld.b32 %0, [%1];" : "=r"(regs4[i]) : "l"(src4 + i));
        }
    } else {
        uint32_t lower, upper;
        /* funnelshift_r will use shift to select the
         * proper four bytes depending on the alignment.
         * 0x10 = 2, 0x11 = 3, 0x01 = 1
         */
        uint8_t shift = (uintptr_t)src % 4;
        /* We need to align the buffer to the nearest 4 bytes
         * Shift down 2 bytes in the case of 2B alignment down
         * 1B in the case of 1B alignment. Shifting down is always
         * appropriate due to 256 B alignment of cudaMalloc().
         */
        uint32_t *src4 = (uint32_t *)((uintptr_t)src & -(uintptr_t)4);
        /* 2 Byte aligned - 2B in lower, 2B in upper */
        /* 1 Byte aligned (0x01) - 1B in upper, 3B in lower */
        /* 1 Byte aligned (0x11) - 3B in upper, 1B in lower */
        for (i = 0; i < nelems; i++) {
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + i));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + i + 1));
            regs4[i] = __funnelshift_r(lower, upper, 8 * shift);
        }
    }

    dest_regs[0] = regs8[0];
    dest_regs[1] = (regs8[1] & ll_flag) | ll_flag_mask;
}

template <typename T, threadgroup_t SCOPE, int UNROLL>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_packLL128(T *psync, const T *source,
                                                                 size_t nelems, uint64_t ll_flag,
                                                                 nvshmemi_team_t *teami,
                                                                 int pe_count, int team_offset,
                                                                 int pe_group_offset) {
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const uint32_t warp_mask = 0x000000FF << (8 * (myIdx / 8));
    const int num_warps = nvshmemi_threadgroup_size<SCOPE>() / NVSHMEMI_WARP_SIZE;
    /* max 127 */
    const uint8_t remain = nelems % _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
    /* max ~ 2048 * 16 * 4*/
    const uint32_t source_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t psync_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t source_stride = num_warps * _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T);
    const uint32_t psync_stride = num_warps * _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T);
    int current_global_pe_index;
    const bool is_flag_thread = !((myIdx + 1) % _LL_128_FLAG_THREAD);

    /* We will always have a SCOPES worth of elements when doing unrolling */
    if (UNROLL > 1) {
        assert(remain == 0);
        assert(nelems % _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) == 0);
    }

    /* Both source_end and psync_end remove the last < 128 bytes.
     * These will be handled after the loop. */
    T *source_end = (T *)(source + nelems - remain);

    uint64_t regs[2 * UNROLL];
    T *source_ptr;
    T *psync_ptr;
    T *remote_psync_ptr;
    uint64_t ll_flag_mask;

    assert(nvshmemi_threadgroup_size<SCOPE>() % NVSHMEMI_WARP_SIZE == 0);
    assert((uintptr_t)psync % 16 == 0);
    assert(sizeof(T) <= 8);

    if (!is_flag_thread) {
        ll_flag = UINT64_MAX;
        ll_flag_mask = 0x0ULL;
    } else {
        ll_flag_mask = ll_flag;
    }
    __syncwarp();

    /* Initial offset by thread */
    source_ptr = (T *)source + source_offset;
    psync_ptr = (T *)psync + psync_offset;

    /* Only gate on source_end. psync_end is not important here. */
    for (; source_ptr < source_end; source_ptr += source_stride, psync_ptr += psync_stride) {
        nvshmemi_store_128b_register<T, UNROLL>(regs, ll_flag, ll_flag_mask, source_ptr,
                                                _LL_128_NUM_DATA_ELEMS_PER_PACKET(T));

        /* definitely possible we are not synchronized before loads to psync. Significant perf
         * overhead? */
        for (int pe = 0; pe < pe_count; pe++) {
            current_global_pe_index = nvshmemi_team_translate_pe_to_team_world_wrap(
                teami, team_offset + (pe_group_offset + pe) % pe_count);
            /*             if (VOLATILE && current_global_pe_index == my_pe_idx) {
                            continue;
                        } */
            remote_psync_ptr = (T *)nvshmemi_ptr(psync_ptr, current_global_pe_index);
#pragma unroll
            for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
                asm volatile(
                    "st.volatile.global.v2.b64 [%0], {%1,%2};" ::"l"((int4 *)remote_psync_ptr),
                    "l"(regs[unroll]), "l"(regs[unroll + 1]));
                remote_psync_ptr += _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T);
            }
        }
    }

    if (UNROLL == 1 && remain) {
        uint8_t num_elems_per_store;
        T *psync_end =
            psync + (nelems - remain) +
            (nelems / _LL_128_NUM_DATA_ELEMS_PER_PACKET(T)) * _LL_128_NUM_ELEMS_PER_FLAG(T);
        /* Last packet will contain < 128 bytes. the variable remain is usually > 0. */
        /* only need 1/4th of a warp */
        if (myIdx < 8) {
            source_ptr = source_end + source_offset;
            psync_ptr = psync_end + psync_offset;
            if (source_offset + _LL_128_NUM_ELEMS_PER_STORE(T) <= remain) {
                nvshmemi_store_128b_register<T, 1>(regs, UINT64_MAX, ll_flag_mask, source_ptr,
                                                   _LL_128_NUM_DATA_ELEMS_PER_PACKET(T));
            } else {
                num_elems_per_store = source_offset > remain ? 0 : remain - source_offset;
                nvshmemi_store_varlen_register<T>(regs, ll_flag, ll_flag_mask, source_ptr,
                                                  num_elems_per_store);
            }
            __syncwarp(warp_mask);
            for (int pe = 0; pe < pe_count; pe++) {
                current_global_pe_index = nvshmemi_team_translate_pe_to_team_world_wrap(
                    teami, team_offset + (pe_group_offset + pe) % pe_count);
                /*                 if (VOLATILE && current_global_pe_index == my_pe_idx) {
                                    continue;
                                } */
                remote_psync_ptr = (T *)nvshmemi_ptr(psync_ptr, current_global_pe_index);
                asm volatile(
                    "st.volatile.global.v2.b64 [%0], {%1,%2};" ::"l"((int4 *)remote_psync_ptr),
                    "l"(regs[0]), "l"(regs[1]));
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <int UNROLL>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ll128_load_psync(uint64_t reg[UNROLL * 2],
                                                                        uint64_t *src,
                                                                        uint64_t ll_flag,
                                                                        uint32_t warp_mask,
                                                                        bool is_flag_thread) {
    uint64_t *offset_src;
    int i;
    bool flag_arrived;
    do {
        flag_arrived = true;
#pragma unroll
        for (i = 0; i < UNROLL * 2; i += 2) {
            offset_src = src + _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(uint64_t) * (i / 2);
            asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                         : "=l"(reg[i]), "=l"(reg[i + 1])
                         : "l"((int4 *)offset_src));
            flag_arrived &= (reg[i + 1] == ll_flag);
            flag_arrived |= !is_flag_thread;
        }
        flag_arrived = __all_sync(warp_mask, flag_arrived != false);
    } while (!flag_arrived);
}

template <typename T, int UNROLL>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_recvLL128(T *dest, const T *psync,
                                                                 size_t nelems, uint64_t flag) {
    const int myIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    const uint32_t warp_mask = 0x000000FF << (8 * (myIdx / 8));
    /* max 127 */
    const uint8_t remain = nelems % _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
    /* max ~ 2048 * 16 * 4*/
    const uint32_t dest_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t psync_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const bool is_flag_thread = !((myIdx + 1) % _LL_128_FLAG_THREAD);
    uint8_t source_copy_len = 16 >> is_flag_thread;

    /* We will always have a SCOPES worth of elements when doing unrolling */
    if (UNROLL > 1) {
        assert(remain == 0);
        assert(nelems % _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) == 0);
    }

    /* Both dest_end and psync_end remove the last < 128 bytes.
     * These will be handled after the loop. */
    T *dest_end = (T *)(dest + nelems - remain);

    uint64_t regs[2 * UNROLL];
    T *dest_ptr;
    T *cur_dest_ptr;
    T *psync_ptr;

    assert((uintptr_t)psync % 16 == 0);
    assert(sizeof(T) <= 8);

    /* Initial offset by thread */
    dest_ptr = (T *)dest + dest_offset;
    psync_ptr = (T *)psync + psync_offset;
    for (; dest_ptr < dest_end; dest_ptr += _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T),
                                psync_ptr += _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T)) {
        if (UNROLL > 1) {
            nvshmemi_ll128_load_psync<UNROLL>(regs, (uint64_t *)psync_ptr, flag, 0xFFFFFFFF,
                                              is_flag_thread);
        } else {
            nvshmemi_ll128_load_psync<UNROLL>(regs, (uint64_t *)psync_ptr, flag, warp_mask,
                                              is_flag_thread);
        }

        cur_dest_ptr = dest_ptr;
#pragma unroll
        for (int i = 0; i < UNROLL * 2; i += 2) {
            asm("st.global.b64 [%0], %1;" ::"l"((uint64_t *)cur_dest_ptr), "l"(regs[i]));
            if (!is_flag_thread) {
                asm("st.global.b64 [%0], %1;" ::"l"((uint64_t *)cur_dest_ptr + 1),
                    "l"(regs[i + 1]));
            }
            if (UNROLL > 1) {
                __syncwarp();
            } else {
                __syncwarp(warp_mask);
            }
            cur_dest_ptr += _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
        }
    }

    if (UNROLL == 1 && remain) {
        T *psync_end =
            (T *)psync + (nelems - remain) +
            (nelems / _LL_128_NUM_DATA_ELEMS_PER_PACKET(T)) * _LL_128_NUM_ELEMS_PER_FLAG(T);
        if (remain && myIdx < 8) {
            dest_ptr = dest_end + dest_offset;
            psync_ptr = psync_end + psync_offset;
            nvshmemi_ll128_load_psync<1>(regs, (uint64_t *)psync_ptr, flag, warp_mask,
                                         is_flag_thread);
            if (dest_offset < remain) {
                source_copy_len = dest_offset + _LL_128_NUM_ELEMS_PER_STORE(T) < remain
                                      ? _LL_128_STORE_SIZE
                                      : ((remain - dest_offset) * sizeof(T));
                nvshmemi_memcpy_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(dest_ptr, regs,
                                                                         source_copy_len);
            }
        }
    }
    __syncwarp();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_mcast_recvLL(T *dest, const uint64_t *src,
                                                                    size_t nelems, uint32_t flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = (nelems * sizeof(T)) / sizeof(uint32_t);
    if (TYPE_IS_FLOAT(T)) {
        for (int i = myIdx; i < num_subelems; i += groupSize) {
            float data1, flag1;
            volatile uint32_t flagu32;
            do {
                asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];"
                             : "=f"(data1), "=f"(flag1)
                             : "l"(&src[i]));
                asm("cvt.rni.u32.f32 %0, %1;" : "=r"(flagu32) : "f"(flag1));
            } while ((flagu32 != flag));
            *(float *)((char *)dest + i * sizeof(float)) = data1;
        }
    } else {
        for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
            uint32_t flag1, flag2, data1, data2;
            do {
                asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                             : "l"(&src[i]));
            } while ((flag1 != flag) || (flag2 != flag));
            *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
            *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
        }
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_mcast_packLL(uint64_t *dest, const T *source,
                                                                    size_t nelems,
                                                                    uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    const size_t num_subelems = (nelems * sizeof(T)) / sizeof(float);
    float flagf32;
    if (TYPE_IS_FLOAT(T)) {
        asm("cvt.rn.f32.u32 %0, %1;" : "=f"(flagf32) : "r"(ll_flag));
    }
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        if (TYPE_IS_FLOAT(T)) {
            float val1 = *(float *)((char *)source + i * sizeof(float));
            float val2 = *(float *)((char *)source + (i + 1) * sizeof(float));
            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&dest[i]), "f"(val1),
                "f"(flagf32), "f"(val2), "f"(flagf32)
                : "memory");
        } else {
            uint32_t val1 = *(uint32_t *)((char *)source + i * sizeof(uint32_t));
            uint32_t val2 = *(uint32_t *)((char *)source + (i + 1) * sizeof(uint32_t));
            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&dest[i]), "r"(val1),
                "r"(ll_flag), "r"(val2), "r"(ll_flag)
                : "memory");
        }
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_recvLL(T *dest, const uint64_t *src,
                                                              size_t nelems, uint32_t flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = (nelems * sizeof(T)) / sizeof(uint32_t);
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        uint32_t flag1, flag2, data1, data2;
        do {
            asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                         : "l"(&src[i]));

        } while ((flag1 != flag) || (flag2 != flag));
        *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
        *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_packLL_naive(uint64_t *dest, const T *source,
                                                                    size_t nelems,
                                                                    uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = (nelems * sizeof(T)) / sizeof(uint32_t);
    for (int i = myIdx * 2; i < num_subelems; i += groupSize * 2) {
        size_t dst_offset = 2 * i * sizeof(uint32_t);
        size_t src_offset = i * sizeof(uint32_t);
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                         (uint32_t *)((char *)dest + dst_offset)),
                     "r"(*(uint32_t *)((char *)source + src_offset)), "r"(ll_flag),
                     "r"(*(uint32_t *)((char *)source + src_offset + sizeof(uint32_t))),
                     "r"(ll_flag));
    }
}

template <typename T, threadgroup_t SCOPE, int UNROLL>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_packLL(T *psync, const T *source,
                                                              size_t nelems, uint32_t ll_flag,
                                                              nvshmemi_team_t *teami, int pe_count,
                                                              int team_offset) {
    const size_t num_subelems = (nelems * sizeof(T)) / sizeof(uint32_t);
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    /* each thread will write 2 subelements (8 bytes) */
    const size_t element_start = myIdx * UNROLL * 2;
    const size_t element_stride = groupSize * UNROLL * 2;
    size_t element_offset;
    uint32_t regs[2 * UNROLL];

    int current_global_pe_index;
    uint32_t *current_dest_address;

    /* We need to be sure to fill each unrolled loop */
    assert(num_subelems % UNROLL * 2 == 0);
    assert((uintptr_t)psync % 16 == 0);

    for (element_offset = element_start; element_offset < num_subelems;
         element_offset += element_stride) {
#pragma unroll
        for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
            if (sizeof(T) == sizeof(uint64_t)) {
                asm("ld.global.v2.u32 {%0,%1}, [%2];"
                    : "=r"(regs[unroll]), "=r"(regs[unroll + 1])
                    : "l"((uint32_t *)source + element_offset + unroll));
            } else {
                regs[unroll] = *((uint32_t *)source + element_offset + unroll);
                regs[unroll + 1] = *((uint32_t *)source + element_offset + unroll + 1);
            }
        }
        for (int pe = 0; pe < pe_count; pe++) {
            current_global_pe_index =
                nvshmemi_team_translate_pe_to_team_world_wrap(teami, team_offset + pe);
            current_dest_address =
                (uint32_t *)nvshmemi_ptr(psync, current_global_pe_index) + element_offset * 2;

#pragma unroll
            for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
                asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                                 (int4 *)current_dest_address),
                             "r"(regs[unroll]), "r"(ll_flag), "r"(regs[unroll + 1]), "r"(ll_flag));
                current_dest_address += sizeof(int4) / sizeof(uint32_t);
            }
        }
    }
}

/* CFT Handle specific functions */
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)

// Function copies data from global to shared memory using TMA
// Returns after the copy is completed
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tma_g2s_copy_thread(
    int myIdx, void *smem_buf, __mbarrier_t *tma_bar_ptr, const void *src, uint32_t copy_bytes) {
    if ((myIdx % warpSize) == 0) {
        // async copy from global to shared only supports mbarrier completion mechanism
        cp_async_bulk_global_to_shared(smem_buf, src, tma_bar_ptr, copy_bytes);

        __mbarrier_token_t token_tma_g2s = barrier_arrive1_tx(tma_bar_ptr, copy_bytes);
        // Wait for previous transfer. Retry in case of false. (waiting can fail)
        while (!barrier_try_wait_token(tma_bar_ptr, token_tma_g2s)) {
        }
    }
}

template <int outstanding_copy_count>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tma_s2g_copy_thread(int myIdx,
                                                                           void *smem_buf,
                                                                           void *dst,
                                                                           uint32_t copy_bytes) {
    if ((myIdx % warpSize) == 0) {
        cp_async_bulk_shared_to_global(dst, smem_buf, copy_bytes);
        nvshmemi_tma_bulk_commit_group();

        // wait untill only 1 outstanding cp is pending
        // This will ensure that copy of previous iteration is completed
        if constexpr (outstanding_copy_count == 0) {
            nvshmemi_tma_bulk_wait_group_0();
        } else {
            // wait till outstanding_copy_count copies are left
            cp_async_bulk_wait_group_read<outstanding_copy_count>();
        }
    }
}

// Function issues try_put using fabric handles.
// NOTE: It does not wait for the copy to be completed.
template <le_fabric_handle_kind cft_handle_kind>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_try_put_wrapper_thread(
    int myIdx, const void *smem_buf, nvshmemi_fabric_handle<cft_handle_kind> dst_handle,
    size_t byte_offset_dst, handle_barrier_t *handle_bar, uint32_t copy_bytes) {
    if ((myIdx % warpSize) == 0) {
        handle_bar->ensure_handle_tx_capacity(copy_bytes);
        fabric_try_put_async<cft_handle_kind>(dst_handle.id(),
                                              dst_handle.offset() + byte_offset_dst, smem_buf,
                                              copy_bytes, handle_bar);
        fabric_submit();
        handle_bar->record_pending_handle(copy_bytes);
    }
}

// Function issues try_get using fabric handles. The caller waits separately.
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_issue_get_wrapper_thread(
    int myIdx, void *dst_smem_buf,
    nvshmemi_fabric_handle<le_fabric_handle_kind::Unicast> src_handle, size_t byte_offset_src,
    handle_barrier_t *handle_bar, uint32_t copy_bytes) {
    if ((myIdx % warpSize) == 0) {
        handle_bar->ensure_handle_get_tx_capacity(copy_bytes);
        fabric_try_get_async(src_handle.id(), src_handle.offset() + byte_offset_src, dst_smem_buf,
                             copy_bytes, handle_bar);
        fabric_submit();
        handle_bar->record_pending_get_handle(copy_bytes);
    }
}

// Function waits for a previously issued try_get to complete.
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_wait_get_wrapper_thread(
    int myIdx, handle_barrier_t *handle_bar, uint32_t copy_bytes) {
    if ((myIdx % warpSize) == 0) {
        // wait till the get is completed
        uint64_t curr_state = handle_bar->arrive_relaxed(copy_bytes);
        handle_bar->wait_primary(curr_state);
    }
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_issue_get_arrive_wrapper_thread(
    int myIdx, void *dst_smem_buf,
    nvshmemi_fabric_handle<le_fabric_handle_kind::Unicast> src_handle, size_t byte_offset_src,
    handle_barrier_t *handle_bar, uint32_t copy_bytes) {
    nvshmemi_issue_get_wrapper_thread(myIdx, dst_smem_buf, src_handle, byte_offset_src, handle_bar,
                                      copy_bytes);

    if ((myIdx % warpSize) == 0) {
        handle_bar->arrive_relaxed(copy_bytes);
    }
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_wait_get_phase_wrapper_thread(
    int myIdx, handle_barrier_t *handle_bar, int phase) {
    if ((myIdx % warpSize) == 0) {
        handle_bar->wait_primary_by_parity(phase);
    }
}

// Function issues try_get using fabric handles and waits for the copy to be completed
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_try_get_wrapper_thread(
    int myIdx, void *dst_smem_buf,
    nvshmemi_fabric_handle<le_fabric_handle_kind::Unicast> src_handle, size_t byte_offset_src,
    handle_barrier_t *handle_bar, uint32_t copy_bytes) {
    nvshmemi_issue_get_wrapper_thread(myIdx, dst_smem_buf, src_handle, byte_offset_src, handle_bar,
                                      copy_bytes);
    nvshmemi_wait_get_wrapper_thread(myIdx, handle_bar, copy_bytes);
}

/* Multi-warp threadgroups use separate producer/consumer warp leaders, so
 * double-buffering can overlap TMA loads with fabric puts. */
template <le_fabric_handle_kind cft_handle_kind, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put_TX_size_threadgroup_specialized(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    static_assert(SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK,
                  "specialized handle put requires a multi-warp threadgroup");
    assert((len % CFT_HANDLE_TX_SIZE) == 0);
    assert(is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size, 2 * TMA_COPY_NUM_STAGES));
    assert(nvshmemi_threadgroup_size<SCOPE>() >= 2 * warpSize);
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP) {
        assert(((blockDim.x * blockDim.y * blockDim.z) % (4 * warpSize)) == 0);
    }

    const bool is_src_shared = __isShared(src);
    if (is_src_shared) {
        assert(nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)src));
    }

    uint32_t tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    /* Use one lane from each of two different warps so the producer and consumer
     * can make independent progress.  Using two lanes from the same warp would
     * still couple them through warp scheduling/divergence. */
    bool is_load = (tid == 0);
    bool is_store = (tid == warpSize);

    uint32_t tid_in_blk = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t warp_idx_in_block = tid_in_blk / warpSize;
    uint32_t thrdgrp_idx_in_block = tid_in_blk / nvshmemi_threadgroup_size<SCOPE>();
    uintptr_t smem_base = registration.base;

    cuda::std::array<uint8_t *, TMA_COPY_NUM_STAGES> smem_data_buf;
    smem_data_buf[0] =
        reinterpret_cast<uint8_t *>(nvshmemi_tma_data_buffer(smem_base) +
                                    thrdgrp_idx_in_block * smem_chunk_size * TMA_COPY_NUM_STAGES);
    smem_data_buf[1] = smem_data_buf[0] + smem_chunk_size;

    /* ready_bar[i]: load leader signals that smem_data_buf[i] contains the next
     * staged chunk.  done_bar[i]: store leader signals that fabric has finished
     * reading smem_data_buf[i], so the load leader may reuse that buffer. */
    cuda::std::array<uint64_t *, TMA_COPY_NUM_STAGES> ready_bar;
    cuda::std::array<uint64_t *, TMA_COPY_NUM_STAGES> done_bar;
    uint32_t tma_slot_base = thrdgrp_idx_in_block * 2 * TMA_COPY_NUM_STAGES;
    ready_bar[0] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base);
    ready_bar[1] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 1);
    done_bar[0] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 2);
    done_bar[1] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 3);

    cuda::std::array<handle_barrier_t *, TMA_COPY_NUM_STAGES> handle_bar;
    uint32_t handle_slot_base = thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES;
    handle_bar[0] = nvshmemi_handle_barrier_slot(smem_base, handle_slot_base);
    handle_bar[1] = nvshmemi_handle_barrier_slot(smem_base, handle_slot_base + 1);

    if (is_load) {
        /* Initialize both the TMA producer/consumer barriers and the fabric
         * completion barriers before either warp starts issuing async work. */
        if (!is_src_shared) {
            nvshmemi_tma_mbarrier_init(ready_bar[0]);
            nvshmemi_tma_mbarrier_init(ready_bar[1]);
            nvshmemi_tma_mbarrier_init(done_bar[0]);
            nvshmemi_tma_mbarrier_init(done_bar[1]);
        }
        handle_bar[0]->prepare_handle(warp_idx_in_block);
        handle_bar[1]->prepare_handle(warp_idx_in_block);
        // Ensure mbarriers init visibility to async proxy
        fence_async_proxy();
    }

    nvshmemi_threadgroup_sync<SCOPE>();

    if (is_load && !is_src_shared) {
        const char *src_ptr = reinterpret_cast<const char *>(src);
        size_t remaining = len;

        /* Producer: stage local global memory into the next shared-memory buffer.
         * Before reusing a buffer from two iterations ago, wait for the consumer
         * to mark that fabric has read it. */
        for (size_t i = 0; remaining > 0; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            int phase = static_cast<int>((i >> 1) & 1);
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < remaining ? smem_chunk_size : remaining);

            if (i >= TMA_COPY_NUM_STAGES) {
                nvshmemi_tma_mbarrier_try_wait(done_bar[curr_buf_idx], phase ^ 1);
            }

            nvshmemi_tma_bulk_global_to_shared(smem_data_buf[curr_buf_idx], src_ptr, copy_bytes,
                                               ready_bar[curr_buf_idx]);
            nvshmemi_tma_mbarrier_arrive_expect_tx(ready_bar[curr_buf_idx], copy_bytes);

            src_ptr += copy_bytes;
            remaining -= copy_bytes;
        }
    } else if (is_store) {
        auto dst_handle = nvshmemi_fabric_handle_for_le_id<cft_handle_kind>(dest_le_id, dst);
        size_t remaining = len;
        uint32_t byte_offset_dst = 0;
        /* Consumer: wait for the producer to fill each buffer, issue the fabric
         * put from that buffer, then wait until fabric has consumed the shared
         * memory before signaling done_bar for reuse. */
        for (size_t i = 0; remaining > 0; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            int phase = static_cast<int>((i >> 1) & 1);
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < remaining ? smem_chunk_size : remaining);

            const void *fabric_src = nvshmemi_ptr_add(src, byte_offset_dst);
            if (!is_src_shared) {
                nvshmemi_tma_mbarrier_try_wait(ready_bar[curr_buf_idx], phase);
                fabric_src = &(smem_data_buf[curr_buf_idx][0]);
            }
            nvshmemi_try_put_wrapper_thread<cft_handle_kind>(
                tid, fabric_src, dst_handle, byte_offset_dst, handle_bar[curr_buf_idx], copy_bytes);

            handle_bar[curr_buf_idx]->fabric_wait_sync_reads();
            if (!is_src_shared) {
                nvshmemi_tma_mbarrier_arrive_expect_tx(done_bar[curr_buf_idx], 1);
                nvshmemi_tma_mbarrier_complete_tx(done_bar[curr_buf_idx], 1);
            }

            byte_offset_dst += copy_bytes;
            remaining -= copy_bytes;
        }

        assert(byte_offset_dst == len);
        /* NBI unicast PUTs retain the completion barriers for nvshmem_quiet().
         * Multicast users remain blocking because their collective lifetime is
         * not owned by the point-to-point quiet path. */
        if (is_blocking || cft_handle_kind != le_fabric_handle_kind::Unicast) {
            for (uint32_t buf_idx = 0; buf_idx < TMA_COPY_NUM_STAGES; buf_idx++) {
                handle_bar[buf_idx]->drain_pending_handle(true);
            }
        }
        if (!is_src_shared) {
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(ready_bar[0]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(ready_bar[1]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(done_bar[0]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(done_bar[1]));
        }
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_use_threadgroup_specialized_handle_path(
    const nvshmemi_tma_smem_registration_t &registration, size_t smem_chunk_size) {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        return nvshmemi_threadgroup_size<SCOPE>() >= 2 * warpSize &&
               is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size, 2 * TMA_COPY_NUM_STAGES);
    }
    return false;
}

/* From this function onwards, we should never move data using pointers
 * Only handles should be used or raise error if not possible
 * SCOPE : non-thread scopes
 * Handle properties are emulated in blackwell. This is required until hardware
 * support for handle features are available.
 * Generic non-thread scopes are single-issuer, so use one full per-threadgroup
 * tile to reduce per-chunk TMA/fabric overhead instead of double-buffering.
 */
template <le_fabric_handle_kind cft_handle_kind, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put_TX_size_generic(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    /* Threadgroups share the per-block TMA SMEM region and handle barriers.
     * Resource availability is checked before using the selected threadgroup.
     */
    static_assert(SCOPE != NVSHMEMI_THREADGROUP_THREAD,
                  "CFT handle operations are not supported for thread scope");

    assert((len % CFT_HANDLE_TX_SIZE) == 0);
    assert(is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size));
    const bool is_src_shared = __isShared(src);
    if (is_src_shared) {
        assert(nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)src));
    }

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t byte_offset_src = 0;
    size_t byte_offset_dst = 0;
    size_t single_buffer_size = smem_chunk_size * TMA_COPY_NUM_STAGES;
    uint32_t copy_bytes = (uint32_t)(single_buffer_size < len ? single_buffer_size : len);

    uint32_t tid_in_blk = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t warp_idx_in_block = tid_in_blk / warpSize;
    uintptr_t smem_base = registration.base;
    // Copy-style handle operations allocate one SMEM/barrier pair per calling threadgroup.
    uint32_t thrdgrp_idx_in_block = tid_in_blk / nvshmemi_threadgroup_size<SCOPE>();

    /* Only following threads enter this code path:
     * warp scope: only lane0
     * block scope: thread id 0
     */
    if (!myIdx) {
        // Compute the per-threadgroup data buffer offset within the full data region.
        uint8_t *smem_data_buf = reinterpret_cast<uint8_t *>(
            nvshmemi_tma_data_buffer(smem_base) + thrdgrp_idx_in_block * single_buffer_size);
        handle_barrier_t *handle_bar;

        /* We use mbarriers for 2 purposes:
         * 1. Sync fabric handle operations using handle_bar
         * 2. Sync TMA transfers using tma_bar_ptr
         */
        handle_bar =
            nvshmemi_handle_barrier_slot(smem_base, thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES);
        __mbarrier_t *tma_bar_ptr = reinterpret_cast<__mbarrier_t *>(
            nvshmemi_tma_barrier_slot(smem_base, warp_idx_in_block));

        handle_bar->prepare_handle(warp_idx_in_block);
        // Ensure mbarriers init visibility to async proxy
        fence_async_proxy();
        if (!is_src_shared) {
            __mbarrier_init(tma_bar_ptr, 1);
            nvshmemi_tma_fence_proxy_async_shared_cta();
            nvshmemi_tma_g2s_copy_thread(myIdx, smem_data_buf, tma_bar_ptr,
                                         nvshmemi_ptr_add(src, byte_offset_src), copy_bytes);
            byte_offset_src += copy_bytes;
        }

        auto dst_handle = nvshmemi_fabric_handle_for_le_id<cft_handle_kind>(dest_le_id, dst);
        while (byte_offset_dst < len) {
            // move data from shared memory to destination global memory
            copy_bytes =
                (uint32_t)(single_buffer_size < (len - byte_offset_dst) ? single_buffer_size
                                                                        : (len - byte_offset_dst));

            const void *fabric_src =
                is_src_shared ? nvshmemi_ptr_add(src, byte_offset_dst) : smem_data_buf;
            nvshmemi_try_put_wrapper_thread<cft_handle_kind>(
                myIdx, fabric_src, dst_handle, byte_offset_dst, handle_bar, copy_bytes);

            byte_offset_dst += copy_bytes;

            // Ensure fabric has consumed the source SMEM before it can be reused.
            handle_bar->fabric_wait_sync_reads();

            // load next chunk of source data into shared memory
            if (!is_src_shared && byte_offset_src < len) {
                copy_bytes = (uint32_t)(single_buffer_size < (len - byte_offset_src)
                                            ? single_buffer_size
                                            : (len - byte_offset_src));

                nvshmemi_tma_g2s_copy_thread(myIdx, smem_data_buf, tma_bar_ptr,
                                             nvshmemi_ptr_add(src, byte_offset_src), copy_bytes);
                byte_offset_src += copy_bytes;
            }
        }

        assert(byte_offset_dst == len);
        assert(is_src_shared || byte_offset_src == len);

        if (is_blocking || cft_handle_kind != le_fabric_handle_kind::Unicast) {
            handle_bar->drain_pending_handle(true);
        }
        if (!is_src_shared) {
            __mbarrier_inval(tma_bar_ptr);
        }
    }
}

template <le_fabric_handle_kind cft_handle_kind, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put_TX_size(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        if (nvshmemi_use_threadgroup_specialized_handle_path<SCOPE>(registration,
                                                                    smem_chunk_size)) {
            nvshmemi_handle_put_TX_size_threadgroup_specialized<cft_handle_kind, SCOPE>(
                registration, dst, src, len, dest_le_id, is_blocking, smem_chunk_size);
            return;
        }
    }

    nvshmemi_handle_put_TX_size_generic<cft_handle_kind, SCOPE>(
        registration, dst, src, len, dest_le_id, is_blocking, smem_chunk_size);
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void
nvshmemi_handle_put_TX_size<le_fabric_handle_kind::Unicast, NVSHMEMI_THREADGROUP_WARP>(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    nvshmemi_handle_put_TX_size_generic<le_fabric_handle_kind::Unicast, NVSHMEMI_THREADGROUP_WARP>(
        registration, dst, src, len, dest_le_id, is_blocking, smem_chunk_size);
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void
nvshmemi_handle_put_TX_size<le_fabric_handle_kind::Multicast, NVSHMEMI_THREADGROUP_WARP>(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    nvshmemi_handle_put_TX_size_generic<le_fabric_handle_kind::Multicast,
                                        NVSHMEMI_THREADGROUP_WARP>(
        registration, dst, src, len, dest_le_id, is_blocking, smem_chunk_size);
}

// Function used for 1 single sub 16B data transfer
template <le_fabric_handle_kind cft_handle_kind, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put_sub_TX_size(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, CUlogicalEndpointId dest_le_id, bool is_blocking,
    size_t smem_chunk_size) {
    /* Threadgroups share the per-block TMA SMEM region and handle barriers.
     * Resource availability is checked before using the selected threadgroup.
     */
    static_assert(SCOPE != NVSHMEMI_THREADGROUP_THREAD,
                  "CFT handle operations are not supported for thread scope");

    // This function is only used for sub 16B data transfers
    // For larger transfers, use the regular nvshmemi_handle_put16B function
    assert(len < CFT_HANDLE_TX_SIZE);
    assert(is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size));
    const bool is_src_shared = __isShared(src);
    if (is_src_shared) {
        assert(nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)src));
    }

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    uint32_t tid_in_blk = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t warp_idx_in_block = tid_in_blk / warpSize;
    // Effectively group ID in scope (e.g., warpID if scope is warp, blockID if scope is block)
    uintptr_t smem_base = registration.base;
    uint32_t thrdgrp_idx_in_block = tid_in_blk / nvshmemi_threadgroup_size<SCOPE>();

    // Copy data to shared memory using threads
    uint8_t *smem_data_buf = reinterpret_cast<uint8_t *>(nvshmemi_tma_data_buffer(smem_base) +
                                                         thrdgrp_idx_in_block * smem_chunk_size);

    if (!is_src_shared) {
        for (uint32_t i = myIdx; i < len; i += groupSize) {
            smem_data_buf[i] = reinterpret_cast<const uint8_t *>(src)[i];
        }
    }

    fence_async_proxy();
    nvshmemi_threadgroup_sync<SCOPE>();

    if (!myIdx) {
        handle_barrier_t *handle_bar;

        handle_bar =
            nvshmemi_handle_barrier_slot(smem_base, thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES);
        handle_bar->prepare_handle(warp_idx_in_block);
        // Ensure mbarriers init visibility to async proxy
        fence_async_proxy();

        auto dst_handle = nvshmemi_fabric_handle_for_le_id<cft_handle_kind>(dest_le_id, dst);
        const void *fabric_src = is_src_shared ? src : smem_data_buf;
        nvshmemi_try_put_wrapper_thread<cft_handle_kind>(myIdx, fabric_src, dst_handle, 0,
                                                         handle_bar, (uint32_t)len);

        handle_bar->fabric_wait_sync_reads();
        if (is_blocking || cft_handle_kind != le_fabric_handle_kind::Unicast) {
            handle_bar->drain_pending_handle(true);
        }
    }
}

/*
 * this function is in threadscope, every thread can call this function
 *
 */
template <typename T, int SMEM_CHUNK_SIZE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_p_emulated(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst, const T src,
    int pe) {
    // This scalar p() path stages one payload in the first 16B of the common
    // per-thread handle slot. Larger payloads should use the regular handle put path.
    static_assert(sizeof(T) <= CFT_HANDLE_TX_SIZE, "CFT handle p supports payloads up to 16B");

    unsigned mask = __activemask();
    unsigned active_threads = __popc(mask);

    uint32_t thrd_idx_in_blk =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int lane_idx = thrd_idx_in_blk % warpSize;
    int leader_lane = __ffs(mask) - 1;
    uintptr_t smem_base = registration.base;
    int warp_idx_in_block = thrd_idx_in_blk / warpSize;

    /*
     * Keep scalar PUTs in the same per-warp slot layout as staged copy paths:
     * each warp owns TMA_COPY_NUM_STAGES consecutive handle barriers.  Using
     * warp_idx_in_block directly would alias another warp's copy barrier
     * (for example, scalar warp 2 slot 2 collides with copy warp 1 stage 0).
     * Scalar PUTs reuse their owning warp's stage-0 barrier.
     */
    handle_barrier_t *handle_bar =
        nvshmemi_handle_barrier_slot(smem_base, warp_idx_in_block * TMA_COPY_NUM_STAGES);

    /*
     * cp_mask works on byte positions within an aligned 16B lane. The
     * destination can be unaligned, so align the fabric offset down and keep
     * dst_byte_offset as the position of the first destination byte inside
     * that aligned lane.
     *
     * Example, dst_offset = 0x108 and sizeof(T) = 8:
     *   aligned_dst_offset = 0x100
     *   dst_byte_offset    = 8
     *   first mask         = 0xFF00, covering bytes 8..15
     *   second_payload     = 0, so this thread issues one cp_mask put
     */
    // TODO: we expect smem to have active threads per block amount of space.
    auto dst_handle = nvshmemi_fabric_handle_for_pe(pe, dst);
    const uint64_t dst_offset = dst_handle.offset();
    const uint32_t dst_byte_offset = static_cast<uint32_t>(dst_offset & (CFT_HANDLE_TX_SIZE - 1));
    const uint64_t aligned_dst_offset = dst_offset - dst_byte_offset;
    const uint32_t payload_bytes = static_cast<uint32_t>(sizeof(T));

    /*
     * A <=16B scalar write can still straddle two aligned 16B lanes. In that
     * case we split this thread's payload across two cp_mask puts.
     *
     * Example, dst_offset = 0x10c and sizeof(T) = 8:
     *   dst_byte_offset       = 12
     *   first_payload_bytes   = 4, bytes 12..15 in lane 0x100, mask 0xF000
     *   second_payload_bytes  = 4, bytes  0..3  in lane 0x110, mask 0x000F
     *
     * crossing_mask marks the active warp lanes that need the second put.
     * active_tx_count counts 16B fabric completion events, not logical payload
     * bytes: every active thread has one put, crossing threads have one extra.
     */
    const uint32_t first_payload_bytes = (payload_bytes < (CFT_HANDLE_TX_SIZE - dst_byte_offset))
                                             ? payload_bytes
                                             : (CFT_HANDLE_TX_SIZE - dst_byte_offset);
    const uint32_t second_payload_bytes = payload_bytes - first_payload_bytes;
    const unsigned crossing_mask = __ballot_sync(mask, second_payload_bytes != 0);
    const uint32_t active_tx_count = active_threads + __popc(crossing_mask);
    const uint32_t completion_bytes = active_tx_count * CFT_HANDLE_TX_SIZE;

    /*
     * Scalar PUTs share handle barriers and staging SMEM with copy paths.
     * Acquire the slot before overwriting SMEM so a previous owner is drained,
     * and retain pending completions from this owner in the same batch.
     */
    if (lane_idx == leader_lane) {
        handle_bar->prepare_handle(warp_idx_in_block);
        handle_bar->ensure_handle_tx_capacity(completion_bytes);
    }
    __syncwarp(mask);

    uint8_t *smem_slot = nvshmemi_handle_thread_smem_slot(smem_base, thrd_idx_in_blk);
    const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(&src);
    // Place source bytes at the same byte positions selected by cp_mask.  If
    // the payload crosses a 16B boundary, the bytes for the second lane wrap to
    // the low positions of this staging slot.
    for (uint32_t i = 0; i < payload_bytes; i++) {
        smem_slot[(dst_byte_offset + i) & (CFT_HANDLE_TX_SIZE - 1)] = src_bytes[i];
    }

    // Ensure async proxy can see the smem data
    fence_async_proxy();

    uint16_t first_bytemask =
        byte_range_to_bytemask_low_first(dst_byte_offset, first_payload_bytes);
    fabric_try_put_async<le_fabric_handle_kind::Unicast>(dst_handle.id(), aligned_dst_offset,
                                                         smem_slot, first_bytemask, handle_bar);
    if (second_payload_bytes) {
        uint16_t second_bytemask = size_to_bytemask_low_first(second_payload_bytes);
        fabric_try_put_async<le_fabric_handle_kind::Unicast>(
            dst_handle.id(), aligned_dst_offset + CFT_HANDLE_TX_SIZE, smem_slot, second_bytemask,
            handle_bar);
    }
    fabric_submit();

    __syncwarp(mask);

    if (lane_idx == leader_lane) {
        handle_bar->record_pending_handle(completion_bytes);
        handle_bar->drain_pending_handle(true);
    }
    __syncwarp(mask);
}

/* Generic GET is used by warp scope.  Keep two fabric GET slots in flight and
 * use each handle barrier as the ready signal before TMA drains that SMEM tile
 * to local global memory. */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_get_emulated_generic(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, int pe, [[maybe_unused]] bool is_blocking,
    size_t smem_chunk_size) {
    /* Threadgroups share the per-block TMA SMEM region and handle barriers.
     * Resource availability is checked before using the selected threadgroup.
     */
    static_assert(SCOPE != NVSHMEMI_THREADGROUP_THREAD,
                  "CFT handle operations are not supported for thread scope");

    assert((len % CFT_HANDLE_TX_SIZE) == 0);
    assert(is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size));
    const bool is_dst_shared = __isShared(dst);
    if (is_dst_shared) {
        assert(nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)dst));
    }

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t tid_in_blk = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t warp_idx_in_block = tid_in_blk / warpSize;
    uintptr_t smem_base = registration.base;
    uint32_t thrdgrp_idx_in_block = tid_in_blk / nvshmemi_threadgroup_size<SCOPE>();

    auto src_handle = nvshmemi_fabric_handle_for_pe(pe, src);
    if (!myIdx) {
        uint8_t *smem_data_buf[TMA_COPY_NUM_STAGES];
        size_t threadgroup_buffer_size = smem_chunk_size * TMA_COPY_NUM_STAGES;
        smem_data_buf[0] = reinterpret_cast<uint8_t *>(
            nvshmemi_tma_data_buffer(smem_base) + thrdgrp_idx_in_block * threadgroup_buffer_size);
        smem_data_buf[1] = smem_data_buf[0] + smem_chunk_size;
        cuda::std::array<handle_barrier_t *, TMA_COPY_NUM_STAGES> handle_bar;
        handle_bar[0] =
            nvshmemi_handle_barrier_slot(smem_base, thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES);
        handle_bar[1] =
            nvshmemi_handle_barrier_slot(smem_base, thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES + 1);

        handle_bar[0]->prepare_handle(warp_idx_in_block);
        handle_bar[1]->prepare_handle(warp_idx_in_block);
        // Ensure mbarriers init visibility to async proxy
        fence_async_proxy();

        size_t num_chunks = NVSHMEMI_TEAM_ROUND_UP_DIV(len, smem_chunk_size);
        size_t preissue_chunks =
            num_chunks < TMA_COPY_NUM_STAGES ? num_chunks : TMA_COPY_NUM_STAGES;

        /* Single issuer: wait the handle barrier, drain the ready SMEM tile,
         * wait for source-read before reuse, then reissue the freed slot. */
        for (size_t i = 0; i < preissue_chunks; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            size_t byte_offset_src = i * smem_chunk_size;
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < (len - byte_offset_src) ? smem_chunk_size
                                                                     : (len - byte_offset_src));
            void *fabric_dst = is_dst_shared ? nvshmemi_ptr_add(dst, byte_offset_src)
                                             : &(smem_data_buf[curr_buf_idx][0]);

            nvshmemi_issue_get_wrapper_thread(myIdx, fabric_dst, src_handle, byte_offset_src,
                                              handle_bar[curr_buf_idx], copy_bytes);
        }

        for (size_t i = 0; i < num_chunks; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            size_t byte_offset_dst = i * smem_chunk_size;
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < (len - byte_offset_dst) ? smem_chunk_size
                                                                     : (len - byte_offset_dst));

            if ((myIdx % warpSize) == 0) {
                handle_bar[curr_buf_idx]->drain_pending_handle(false);
            }
            if (!is_dst_shared) {
                unsigned int data_addr = nvshmemi_tma_cvta_to_shared(smem_data_buf[curr_buf_idx]);
                nvshmemi_tma_bulk_shared_to_global(nvshmemi_ptr_add(dst, byte_offset_dst),
                                                   data_addr, copy_bytes);
                nvshmemi_tma_bulk_commit_group();
            }

            size_t next_chunk = i + TMA_COPY_NUM_STAGES;
            if (next_chunk < num_chunks) {
                size_t next_byte_offset_src = next_chunk * smem_chunk_size;
                uint32_t next_buf_idx = static_cast<uint32_t>(next_chunk & 1);
                uint32_t next_copy_bytes = (uint32_t)(smem_chunk_size < (len - next_byte_offset_src)
                                                          ? smem_chunk_size
                                                          : (len - next_byte_offset_src));
                void *next_fabric_dst = is_dst_shared ? nvshmemi_ptr_add(dst, next_byte_offset_src)
                                                      : &(smem_data_buf[next_buf_idx][0]);

                if (!is_dst_shared) {
                    nvshmemi_tma_bulk_wait_group_read_0();
                }
                nvshmemi_issue_get_wrapper_thread(myIdx, next_fabric_dst, src_handle,
                                                  next_byte_offset_src, handle_bar[next_buf_idx],
                                                  next_copy_bytes);
            }
        }

        if (!is_dst_shared) {
            nvshmemi_tma_bulk_wait_group_0();
        }

        // invalidate the barrier
        if ((myIdx % warpSize) == 0) {
            handle_bar[0]->drain_pending_handle(true);
            handle_bar[1]->drain_pending_handle(true);
        }
        if (is_dst_shared) {
            fence_proxy_fabric2generic_alias();
        }

    }  // end
}

/* Multi-warp GET uses separate producer/consumer warp leaders.  The producer
 * keeps two fabric GETs in flight, the consumer waits the handle barrier
 * directly as the ready signal, and done_bar protects SMEM buffer reuse after
 * TMA drains. */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_get_emulated_threadgroup_specialized(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, int pe, [[maybe_unused]] bool is_blocking,
    size_t smem_chunk_size) {
    static_assert(SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK,
                  "specialized handle get requires a multi-warp threadgroup");
    assert((len % CFT_HANDLE_TX_SIZE) == 0);
    const bool is_dst_shared = __isShared(dst);
    if (is_dst_shared) {
        assert(nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)dst));
    }
    assert(is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size, 2 * TMA_COPY_NUM_STAGES));
    assert(nvshmemi_threadgroup_size<SCOPE>() >= 2 * warpSize);
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP) {
        assert(((blockDim.x * blockDim.y * blockDim.z) % (4 * warpSize)) == 0);
    }

    uint32_t tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    bool is_load = (tid == 0);
    bool is_store = (tid == warpSize);

    uint32_t tid_in_blk = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t warp_idx_in_block = tid_in_blk / warpSize;
    uint32_t thrdgrp_idx_in_block = tid_in_blk / nvshmemi_threadgroup_size<SCOPE>();
    uintptr_t smem_base = registration.base;

    cuda::std::array<uint8_t *, TMA_COPY_NUM_STAGES> smem_data_buf;
    smem_data_buf[0] =
        reinterpret_cast<uint8_t *>(nvshmemi_tma_data_buffer(smem_base) +
                                    thrdgrp_idx_in_block * smem_chunk_size * TMA_COPY_NUM_STAGES);
    smem_data_buf[1] = smem_data_buf[0] + smem_chunk_size;

    /* handle_bar[i]: fabric GET has completed into smem_data_buf[i].
     * ready_bar[i]: producer has submitted the fabric GET for this phase.
     * done_bar[i]: TMA has consumed smem_data_buf[i], so fabric may reuse it. */
    cuda::std::array<uint64_t *, TMA_COPY_NUM_STAGES> ready_bar;
    cuda::std::array<uint64_t *, TMA_COPY_NUM_STAGES> done_bar;
    uint32_t tma_slot_base = thrdgrp_idx_in_block * 2 * TMA_COPY_NUM_STAGES;
    ready_bar[0] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base);
    ready_bar[1] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 1);
    done_bar[0] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 2);
    done_bar[1] = nvshmemi_tma_barrier_slot(smem_base, tma_slot_base + 3);

    cuda::std::array<handle_barrier_t *, TMA_COPY_NUM_STAGES> handle_bar;
    uint32_t handle_slot_base = thrdgrp_idx_in_block * TMA_COPY_NUM_STAGES;
    handle_bar[0] = nvshmemi_handle_barrier_slot(smem_base, handle_slot_base);
    handle_bar[1] = nvshmemi_handle_barrier_slot(smem_base, handle_slot_base + 1);

    if (is_load) {
        nvshmemi_tma_mbarrier_init(ready_bar[0]);
        nvshmemi_tma_mbarrier_init(ready_bar[1]);
        nvshmemi_tma_mbarrier_init(done_bar[0]);
        nvshmemi_tma_mbarrier_init(done_bar[1]);
        handle_bar[0]->prepare_handle(warp_idx_in_block);
        handle_bar[1]->prepare_handle(warp_idx_in_block);
        // Ensure mbarriers init visibility to async proxy
        fence_async_proxy();
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    size_t num_chunks = NVSHMEMI_TEAM_ROUND_UP_DIV(len, smem_chunk_size);
    size_t preissue_chunks = num_chunks < TMA_COPY_NUM_STAGES ? num_chunks : TMA_COPY_NUM_STAGES;

    if (is_load) {
        auto src_handle = nvshmemi_fabric_handle_for_pe(pe, src);

        /* Producer: keep both handle GET slots in flight when possible.  Publish
         * readiness only after fabric.submit; before reissuing a slot, wait until
         * TMA has consumed that SMEM buffer. */
        for (size_t i = 0; i < preissue_chunks; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            size_t byte_offset_src = i * smem_chunk_size;
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < (len - byte_offset_src) ? smem_chunk_size
                                                                     : (len - byte_offset_src));
            void *fabric_dst = is_dst_shared ? nvshmemi_ptr_add(dst, byte_offset_src)
                                             : &(smem_data_buf[curr_buf_idx][0]);

            nvshmemi_issue_get_wrapper_thread(tid, fabric_dst, src_handle, byte_offset_src,
                                              handle_bar[curr_buf_idx], copy_bytes);
            nvshmemi_tma_mbarrier_arrive_expect_tx(ready_bar[curr_buf_idx], 1);
            nvshmemi_tma_mbarrier_complete_tx(ready_bar[curr_buf_idx], 1);
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    if (is_load) {
        auto src_handle = nvshmemi_fabric_handle_for_pe(pe, src);
        for (size_t next_chunk = TMA_COPY_NUM_STAGES; next_chunk < num_chunks; next_chunk++) {
            size_t next_byte_offset_src = next_chunk * smem_chunk_size;
            uint32_t next_buf_idx = static_cast<uint32_t>(next_chunk & 1);
            int next_phase = static_cast<int>((next_chunk >> 1) & 1);
            uint32_t next_copy_bytes = (uint32_t)(smem_chunk_size < (len - next_byte_offset_src)
                                                      ? smem_chunk_size
                                                      : (len - next_byte_offset_src));
            void *next_fabric_dst = is_dst_shared ? nvshmemi_ptr_add(dst, next_byte_offset_src)
                                                  : &(smem_data_buf[next_buf_idx][0]);

            nvshmemi_tma_mbarrier_try_wait(done_bar[next_buf_idx], next_phase ^ 1);
            nvshmemi_issue_get_wrapper_thread(tid, next_fabric_dst, src_handle,
                                              next_byte_offset_src, handle_bar[next_buf_idx],
                                              next_copy_bytes);
            nvshmemi_tma_mbarrier_arrive_expect_tx(ready_bar[next_buf_idx], 1);
            nvshmemi_tma_mbarrier_complete_tx(ready_bar[next_buf_idx], 1);
        }
    } else if (is_store) {
        size_t remaining = len;
        uint32_t byte_offset_dst = 0;

        /* Consumer: wait for fabric to fill each SMEM tile, issue the local
         * S2G TMA copy, then signal done after TMA has read that tile. */
        for (size_t i = 0; remaining > 0; i++) {
            uint32_t curr_buf_idx = static_cast<uint32_t>(i & 1);
            int curr_phase = static_cast<int>((i >> 1) & 1);
            uint32_t copy_bytes =
                (uint32_t)(smem_chunk_size < remaining ? smem_chunk_size : remaining);

            if ((tid % warpSize) == 0) {
                nvshmemi_tma_mbarrier_try_wait(ready_bar[curr_buf_idx], curr_phase);
                handle_bar[curr_buf_idx]->drain_pending_handle(false);
            }
            if (!is_dst_shared) {
                unsigned int data_addr = nvshmemi_tma_cvta_to_shared(smem_data_buf[curr_buf_idx]);
                nvshmemi_tma_bulk_shared_to_global(nvshmemi_ptr_add(dst, byte_offset_dst),
                                                   data_addr, copy_bytes);
                nvshmemi_tma_bulk_commit_group();
                nvshmemi_tma_bulk_wait_group_read_0();
            }
            nvshmemi_tma_mbarrier_arrive_expect_tx(done_bar[curr_buf_idx], 1);
            nvshmemi_tma_mbarrier_complete_tx(done_bar[curr_buf_idx], 1);

            byte_offset_dst += copy_bytes;
            remaining -= copy_bytes;
        }

        assert(byte_offset_dst == len);
        if (!is_dst_shared) {
            nvshmemi_tma_bulk_wait_group_0();
        }
        if ((tid % warpSize) == 0) {
            handle_bar[0]->drain_pending_handle(true);
            handle_bar[1]->drain_pending_handle(true);
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(ready_bar[0]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(ready_bar[1]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(done_bar[0]));
            __mbarrier_inval(reinterpret_cast<__mbarrier_t *>(done_bar[1]));
        }
        if (is_dst_shared) {
            fence_proxy_fabric2generic_alias();
        }
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_get_emulated(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, int pe, bool is_blocking, size_t smem_chunk_size) {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        if (nvshmemi_use_threadgroup_specialized_handle_path<SCOPE>(registration,
                                                                    smem_chunk_size)) {
            nvshmemi_handle_get_emulated_threadgroup_specialized<SCOPE>(
                registration, dst, src, len, pe, is_blocking, smem_chunk_size);
            return;
        }
    }

    nvshmemi_handle_get_emulated_generic<SCOPE>(registration, dst, src, len, pe, is_blocking,
                                                smem_chunk_size);
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void
nvshmemi_handle_get_emulated<NVSHMEMI_THREADGROUP_WARP>(
    const nvshmemi_tma_smem_registration_t &registration, void *__restrict__ dst,
    const void *__restrict__ src, size_t len, int pe, bool is_blocking, size_t smem_chunk_size) {
    nvshmemi_handle_get_emulated_generic<NVSHMEMI_THREADGROUP_WARP>(registration, dst, src, len, pe,
                                                                    is_blocking, smem_chunk_size);
}

/* Handle PUT is supported for non-thread scopes.  The destination address must
 * be CFT_HANDLE_TX_SIZE-aligned; a sub-TX-size payload tail is handled below. */
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_put(const void *src, void *dst,
                                                                  size_t len, int pe,
                                                                  bool is_blocking) {
    if constexpr (SCOPE != NVSHMEMI_THREADGROUP_THREAD) {
        if (pe == nvshmemi_device_state_d.mype) {
            nvshmemi_memcpy_threadgroup<SCOPE>(dst, src, len);
            return;
        }

        // address must be aligned to CFT_HANDLE_TX_SIZE
        assert(nvshmemi_is_addr_offset_aligned(dst, CFT_HANDLE_TX_SIZE));
        assert(nvshmemi_ld_and_check_valid_le_id(pe));
        CUlogicalEndpointId dest_le_id = nvshmemi_ld_and_get_le_id((pe));
        size_t adjusted_size = (len / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;
        const auto registration = nvshmemi_tma_get_smem_registration();

        uint32_t num_threadgroups =
            NVSHMEMI_TEAM_ROUND_UP_DIV(nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_BLOCK>(),
                                       nvshmemi_threadgroup_size<SCOPE>());
        size_t smem_chunk_size = nvshmemi_tma_align_down_16(
            nvshmemi_smem_data_buf_size(registration, TMA_COPY_NUM_STAGES) / num_threadgroups);

        if (adjusted_size) {
            nvshmemi_handle_put_TX_size<le_fabric_handle_kind::Unicast, SCOPE>(
                registration, dst, src, adjusted_size, dest_le_id, is_blocking, smem_chunk_size);
        }

        size_t remaining_size = len - adjusted_size;
        if (remaining_size) {
            nvshmemi_threadgroup_sync<SCOPE>();
            nvshmemi_handle_put_sub_TX_size<le_fabric_handle_kind::Unicast, SCOPE>(
                registration, nvshmemi_ptr_add(dst, adjusted_size),
                nvshmemi_ptr_add(src, adjusted_size), remaining_size, dest_le_id, is_blocking,
                smem_chunk_size);
        }
    }
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_p(void *__restrict__ dst, const T src,
                                                                int pe) {
    if (pe == nvshmemi_device_state_d.mype) {
        *reinterpret_cast<volatile T *>(dst) = src;
        return;
    }

    const auto registration = nvshmemi_tma_get_smem_registration();
    nvshmemi_handle_p_emulated<T, NVSHMEMI_SMEM_BUF_SIZE>(registration, dst, src, pe);
}

#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET
template <le_fabric_atomic_op Op, typename T>
struct nvshmemi_fabric_atomic_type {
    static_assert(Op != le_fabric_atomic_op::Add, "fabric ADD requires its type specialization");
    static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t),
                  "unsupported fabric atomic type");
    static constexpr le_fabric_atomic_type value =
        sizeof(T) == sizeof(uint32_t) ? le_fabric_atomic_type::B32 : le_fabric_atomic_type::B64;
};

template <typename T>
struct nvshmemi_fabric_atomic_type<le_fabric_atomic_op::Add, T> {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t),
                  "unsupported fabric ADD type");
    static constexpr le_fabric_atomic_type value =
        std::is_same_v<T, float>        ? le_fabric_atomic_type::F32
        : std::is_same_v<T, double>     ? le_fabric_atomic_type::F64
        : sizeof(T) == sizeof(uint32_t) ? le_fabric_atomic_type::U32
                                        : le_fabric_atomic_type::U64;
};

template <le_fabric_atomic_op Op, typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_handle_atomic_once(const nvshmemi_tma_smem_registration_t &registration,
                            T *__restrict__ dst, T value, T compare, int pe) {
    const unsigned active_mask = __activemask();
    const uint32_t active_threads = __popc(active_mask);
    const uint32_t thread_idx =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const uint32_t lane_idx = thread_idx % warpSize;
    const uint32_t leader_lane = __ffs(active_mask) - 1;
    const uint32_t warp_idx_in_block = thread_idx / warpSize;

    const uintptr_t smem_base = registration.base;
    /* fabric.try_atom always operates on a 16B destination block. Its regular
     * source occupies 16B, while CAS requires a 32B-aligned 32B source whose
     * lower and upper halves contain the compare and swap blocks respectively.
     * Each thread's region is at least 16B-aligned, so rounding its source up
     * to 32B consumes at most the first 16B of that same region.
     *
     * Give every thread a fixed 64B region. If its base is 32B-aligned, the
     * source/result occupy [0, 48). If it is only 16B-aligned, they occupy
     * [16, 64). Scalar PUT always uses [0, 16).
     *
     * Although source plus result need only 48B, the 64B stride contains the
     * possible alignment adjustment and isolates this thread from both
     * scalar PUT and atomic staging owned by every other thread. */
    const uintptr_t thread_smem =
        reinterpret_cast<uintptr_t>(nvshmemi_handle_thread_smem_slot(smem_base, thread_idx));
    uint8_t *src_smem =
        reinterpret_cast<uint8_t *>((thread_smem + NVSHMEMI_HANDLE_ATOMIC_SMEM_ALIGNMENT - 1) &
                                    ~(NVSHMEMI_HANDLE_ATOMIC_SMEM_ALIGNMENT - 1));
    void *return_value_smem = src_smem + 2 * CFT_HANDLE_TX_SIZE;

    /* Reuse stage 0 of this physical warp's two-slot CFT barrier allocation. */
    handle_barrier_t *handle_bar =
        nvshmemi_handle_barrier_slot(smem_base, warp_idx_in_block * TMA_COPY_NUM_STAGES);

    if (lane_idx == leader_lane) {
        /* prepare_handle() preserves pending batches owned by this same warp.
         * Drain explicitly before overwriting staging memory that such a batch
         * may still be reading. The atomic itself is blocking, so retaining the
         * earlier batch provides no benefit here. */
        handle_bar->drain_pending_handle(true);
        handle_bar->prepare_handle(warp_idx_in_block);
        handle_bar->ensure_handle_tx_capacity(active_threads * CFT_HANDLE_TX_SIZE);
    }
    __syncwarp(active_mask);

    /* Every instruction operates on a 16B destination block. Initialize the
     * unused lanes with the operation's identity so only the requested scalar
     * changes. CAS uses compare=swap=0 for unused lanes, which is also an
     * identity whether the old lane is zero or nonzero. */
    constexpr size_t source_bytes =
        Op == le_fabric_atomic_op::Cas ? 2 * CFT_HANDLE_TX_SIZE : CFT_HANDLE_TX_SIZE;
    constexpr uint8_t identity = Op == le_fabric_atomic_op::And ? 0xff : 0;
    for (size_t i = 0; i < source_bytes; i++) {
        src_smem[i] = identity;
    }
    if constexpr (Op == le_fabric_atomic_op::Cas) {
        *reinterpret_cast<T *>(src_smem) = compare;
        *reinterpret_cast<T *>(src_smem + CFT_HANDLE_TX_SIZE) = value;
    } else {
        *reinterpret_cast<T *>(src_smem) = value;
    }
    fence_async_proxy();

    const auto dst_handle = nvshmemi_fabric_handle_for_pe(pe, dst);
    fabric_try_atom_async<Op, nvshmemi_fabric_atomic_type<Op, T>::value>(
        dst_handle.id(), dst_handle.offset(), return_value_smem, src_smem, handle_bar);
    fabric_submit();
    __syncwarp(active_mask);

    if (lane_idx == leader_lane) {
        handle_bar->record_pending_handle(active_threads * CFT_HANDLE_TX_SIZE);
        handle_bar->drain_pending_handle(true);
    }
    __syncwarp(active_mask);

    return *reinterpret_cast<T *>(return_value_smem);
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_atomic_bits_equal(T lhs, T rhs) {
    const uint8_t *lhs_bytes = reinterpret_cast<const uint8_t *>(&lhs);
    const uint8_t *rhs_bytes = reinterpret_cast<const uint8_t *>(&rhs);
    for (size_t i = 0; i < sizeof(T); i++) {
        if (lhs_bytes[i] != rhs_bytes[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_handle_atomic_swap_impl(
    const nvshmemi_tma_smem_registration_t &registration, T *__restrict__ dst, T value, int pe) {
    T compare{};
    while (true) {
        T old = nvshmemi_handle_atomic_once<le_fabric_atomic_op::Cas>(registration, dst, value,
                                                                      compare, pe);
        if (nvshmemi_atomic_bits_equal(old, compare)) {
            return old;
        }
        compare = old;
    }
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE __half
nvshmemi_handle_half_atomic_add_impl(const nvshmemi_tma_smem_registration_t &registration,
                                     __half *__restrict__ dst, __half value, int pe) {
    /* Handle atomics require dst to be 16B-aligned, so the scalar half is the
     * low lane of this word. Use bitwise word operations to keep the adjacent
     * half unchanged, including negative zero and NaN payloads. */
    uint32_t *word = reinterpret_cast<uint32_t *>(dst);
    uint32_t expected = nvshmemi_handle_atomic_once<le_fabric_atomic_op::Or>(
        registration, word, uint32_t{0}, uint32_t{0}, pe);
    const __half_raw value_raw = static_cast<__half_raw>(value);

    while (true) {
        const __half_raw old_raw = {static_cast<uint16_t>(expected)};
        const __half new_value = __hadd(__half(old_raw), __half(value_raw));
        const uint32_t desired =
            (expected & 0xffff0000U) | static_cast<uint32_t>(static_cast<__half_raw>(new_value).x);
        const uint32_t observed = nvshmemi_handle_atomic_once<le_fabric_atomic_op::Cas>(
            registration, word, desired, expected, pe);
        if (observed == expected) {
            return __half(old_raw);
        }
        expected = observed;
    }
}

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_handle_atomic_fetch_impl(const nvshmemi_tma_smem_registration_t &registration,
                                  T *__restrict__ dst, T value, T compare, int pe) {
    if constexpr (Amo == NVSHMEMI_AMO_ADD || Amo == NVSHMEMI_AMO_FETCH_ADD ||
                  Amo == NVSHMEMI_AMO_INC || Amo == NVSHMEMI_AMO_FETCH_INC ||
                  Amo == NVSHMEMI_AMO_SIGNAL_ADD) {
        if constexpr (std::is_same_v<T, __half>) {
            return nvshmemi_handle_half_atomic_add_impl(registration, dst, value, pe);
        } else {
            return nvshmemi_handle_atomic_once<le_fabric_atomic_op::Add>(registration, dst, value,
                                                                         T{}, pe);
        }
    } else if constexpr (Amo == NVSHMEMI_AMO_AND || Amo == NVSHMEMI_AMO_FETCH_AND) {
        return nvshmemi_handle_atomic_once<le_fabric_atomic_op::And>(registration, dst, value, T{},
                                                                     pe);
    } else if constexpr (Amo == NVSHMEMI_AMO_OR || Amo == NVSHMEMI_AMO_FETCH_OR ||
                         Amo == NVSHMEMI_AMO_FETCH) {
        return nvshmemi_handle_atomic_once<le_fabric_atomic_op::Or>(registration, dst, value, T{},
                                                                    pe);
    } else if constexpr (Amo == NVSHMEMI_AMO_XOR || Amo == NVSHMEMI_AMO_FETCH_XOR) {
        return nvshmemi_handle_atomic_once<le_fabric_atomic_op::Xor>(registration, dst, value, T{},
                                                                     pe);
    } else if constexpr (Amo == NVSHMEMI_AMO_COMPARE_SWAP) {
        return nvshmemi_handle_atomic_once<le_fabric_atomic_op::Cas>(registration, dst, value,
                                                                     compare, pe);
    } else {
        static_assert(Amo == NVSHMEMI_AMO_SWAP || Amo == NVSHMEMI_AMO_SET,
                      "unsupported fabric atomic operation");
        return nvshmemi_handle_atomic_swap_impl(registration, dst, value, pe);
    }
}
#endif

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_get(const void *src, void *dst,
                                                                  size_t len, int pe,
                                                                  bool is_blocking) {
    if constexpr (SCOPE != NVSHMEMI_THREADGROUP_THREAD) {
#if __CUDA_ARCH__ >= 1000
        if (pe == nvshmemi_device_state_d.mype) {
            nvshmemi_memcpy_threadgroup<SCOPE>(dst, src, len);
            return;
        }

        assert(nvshmemi_is_addr_offset_aligned(src, CFT_HANDLE_TX_SIZE));
        assert(nvshmemi_ld_and_check_valid_le_id(pe));
        const auto registration = nvshmemi_tma_get_smem_registration();

        uint32_t num_threadgroups =
            NVSHMEMI_TEAM_ROUND_UP_DIV(nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_BLOCK>(),
                                       nvshmemi_threadgroup_size<SCOPE>());
        size_t smem_chunk_size = nvshmemi_tma_align_down_16(
            nvshmemi_smem_data_buf_size(registration, TMA_COPY_NUM_STAGES) / num_threadgroups);

        nvshmemi_handle_get_emulated<SCOPE>(registration, dst, src, len, pe, is_blocking,
                                            smem_chunk_size);
#else
        assert(0 && "nvshmemi_handle_get: not supported");
#endif
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE size_t nvshmemi_handle_mcast_memcpy_threadgroup(
    nvshmemi_team_t *team, T *__restrict__ dst, const T *__restrict__ src, size_t len) {
#if __CUDA_ARCH__ >= 1000
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        assert(0 && "nvshmemi_handle_mcast_memcpy_threadgroup: not implemented for thread scope");
        return 0;
    } else {
        // address must be aligned to CFT_HANDLE_TX_SIZE
        assert(nvshmemi_is_addr_offset_aligned(dst, CFT_HANDLE_TX_SIZE));
        const auto registration = nvshmemi_tma_get_smem_registration();

        size_t smem_chunk_size;
        if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
            // Preserve the block-specialized path's full-stage allocation.
            smem_chunk_size = nvshmemi_smem_data_buf_size(registration, TMA_COPY_NUM_STAGES);
        } else {
            /* Threadgroups in a block share each SMEM stage.  Give every calling
             * threadgroup the largest non-overlapping, 16B-aligned chunk, matching
             * the unicast handle put/get paths. */
            uint32_t num_threadgroups =
                NVSHMEMI_TEAM_ROUND_UP_DIV(nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_BLOCK>(),
                                           nvshmemi_threadgroup_size<SCOPE>());
            smem_chunk_size = nvshmemi_tma_align_down_16(
                nvshmemi_smem_data_buf_size(registration, TMA_COPY_NUM_STAGES) / num_threadgroups);
        }

        // round low to nearest multiple of CFT_HANDLE_TX_SIZE
        size_t adjusted_size = (len / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;
        assert(IS_VALID_LE_ID(team->mc_leid_with_flag));
        CUlogicalEndpointId mc_le_id = PARSE_LE_ID(team->mc_leid_with_flag);
        if (adjusted_size) {
            nvshmemi_handle_put_TX_size<le_fabric_handle_kind::Multicast, SCOPE>(
                registration, dst, src, adjusted_size, mc_le_id, true, smem_chunk_size);
        }

        len -= adjusted_size;
        if (len) {
            nvshmemi_threadgroup_sync<SCOPE>();
            nvshmemi_handle_put_sub_TX_size<le_fabric_handle_kind::Multicast, SCOPE>(
                registration, nvshmemi_ptr_add(dst, adjusted_size),
                nvshmemi_ptr_add(src, adjusted_size), len, mc_le_id, true, smem_chunk_size);
            len = 0;
        }
    }
#else
    assert(0 && "nvshmemi_handle_mcast_memcpy_threadgroup: not supported");
#endif
    return len;
}

#endif  // LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE constexpr bool nvshmemi_handle_atomic_is_supported() {
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if constexpr (Amo == NVSHMEMI_AMO_ADD || Amo == NVSHMEMI_AMO_FETCH_ADD ||
                  Amo == NVSHMEMI_AMO_INC || Amo == NVSHMEMI_AMO_FETCH_INC ||
                  Amo == NVSHMEMI_AMO_SIGNAL_ADD) {
        return std::is_same_v<T, __half> || sizeof(T) == sizeof(uint32_t) ||
               sizeof(T) == sizeof(uint64_t);
    } else {
        return sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t);
    }
#else
    return false;
#endif
}

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_can_use_handle_atomic(T *target, int pe) {
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if constexpr (!nvshmemi_handle_atomic_is_supported<T, Amo>()) {
        return false;
    } else {
        return nvshmemi_is_le_atomic_implemented(pe, target);
    }
#else
    (void)target;
    (void)pe;
    return false;
#endif
}

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_handle_atomic_fetch(T *target, T value,
                                                                        T compare, int pe) {
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    assert((nvshmemi_handle_atomic_is_supported<T, Amo>()));
    const auto registration = nvshmemi_tma_get_smem_registration();
    return nvshmemi_handle_atomic_fetch_impl<T, Amo>(registration, target, value, compare, pe);
#else
    /* Compile-only stub. nvshmemi_can_use_handle_atomic() returns false under
     * the same feature gate, so normal dispatch cannot execute this path. */
    (void)target;
    (void)value;
    (void)compare;
    (void)pe;
    return T{};
#endif
}

template <typename T, nvshmemi_amo_t Amo>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_handle_atomic_nonfetch(T *target, T value,
                                                                              int pe) {
#if LE_ATOMIC_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const auto registration = nvshmemi_tma_get_smem_registration();
    (void)nvshmemi_handle_atomic_fetch_impl<T, Amo>(registration, target, value, T{}, pe);
#else
    /* Compile-only stub; see nvshmemi_handle_atomic_fetch() above. */
    (void)target;
    (void)value;
    (void)pe;
#endif
}

#endif /* __CUDA__ARCH__ */
#endif /* _NVSHMEM_COMMON_DEVICE_CUH_ */
