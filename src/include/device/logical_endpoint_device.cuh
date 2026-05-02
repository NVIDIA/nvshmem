/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __logical_endpoint_device_cuh__
#define __logical_endpoint_device_cuh__

#ifdef __CUDA_ARCH__

#if defined(__CUDACC_RTC__)

#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

typedef unsigned int CUlogicalEndpointId;

#define LE_HW_SW_REQUIREMENTS_MET 0
inline constexpr int CFT_HANDLE_TX_SIZE = 16;

__device__ __forceinline__ bool nvshmemi_ld_and_check_valid_le_id(int) { return false; }

__device__ __forceinline__ CUlogicalEndpointId nvshmemi_ld_and_get_le_id(int) {
    return (CUlogicalEndpointId)-1;
}

__device__ __forceinline__ bool nvshmemi_is_addr_offset_aligned(const void *, size_t) {
    return false;
}

__device__ __forceinline__ bool nvshmemi_is_le_implemented(int, size_t, threadgroup_t,
                                                           const void *, const void *) {
    return false;
}

__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int) { return false; }

__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int, size_t,
                                                                         threadgroup_t,
                                                                         const void *, const void *) {
    return false;
}

#else

#include "device_host/logical_endpoint_types.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

inline constexpr int NVSHMEMI_SMEM_BUF_SIZE = 512; // 16*32 - 16B per thread, 1 buf per warp

__device__ __forceinline__ bool nvshmemi_ld_and_check_valid_le_id(int pe) {
#if defined(CFT_HANDLES_ENABLED)
    const long long unsigned *le_ids =
        reinterpret_cast<const long long unsigned *>(nvshmemi_device_state_d.unicast_le_ids_);
    return IS_VALID_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return false;
#endif
}

__device__ __forceinline__ CUlogicalEndpointId nvshmemi_ld_and_get_le_id(int pe) {
#if defined(CFT_HANDLES_ENABLED)
    const long long unsigned *le_ids =
        reinterpret_cast<const long long unsigned *>(nvshmemi_device_state_d.unicast_le_ids_);
    return (CUlogicalEndpointId)PARSE_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return (CUlogicalEndpointId)-1;
#endif
}

__device__ bool nvshmemi_tma_smem_registered();
#if LE_HW_SW_REQUIREMENTS_MET && defined(CFT_HANDLES_ENABLED)
__device__ __forceinline__ size_t nvshmemi_smem_data_buf_size(size_t num_buffers);
__device__ constexpr bool nvshmemi_tma_is_16b_aligned(size_t value);
#endif

__device__ __forceinline__ bool nvshmemi_is_addr_offset_aligned(const void *addr, size_t size) {
    if (addr == nullptr) return false;
    if (size == 0) return true;
    return ((((uintptr_t)addr - (uintptr_t)nvshmemi_device_state_d.heap_base) % size) == 0);
}

__device__ __forceinline__ bool nvshmemi_is_le_implemented(int pe, size_t size, threadgroup_t scope,
                                                           const void *le_addr,
                                                           const void *tma_addr) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(CFT_HANDLES_ENABLED)
    return ((scope == NVSHMEMI_THREADGROUP_BLOCK) && nvshmemi_tma_smem_registered() &&
            (nvshmemi_smem_data_buf_size(TMA_COPY_NUM_STAGES) >= CFT_HANDLE_TX_SIZE) &&
            nvshmemi_is_addr_offset_aligned(le_addr, CFT_HANDLE_TX_SIZE) &&
            nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)tma_addr) &&
            nvshmemi_ld_and_check_valid_le_id(pe) && ((size % CFT_HANDLE_TX_SIZE) == 0));
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int pe) {
#if defined(CFT_HANDLES_ENABLED) && defined(PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_ld_and_check_valid_le_id(pe);
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int pe, size_t size,
                                                                         threadgroup_t scope,
                                                                         const void *le_addr,
                                                                         const void *tma_addr) {
#if defined(CFT_HANDLES_ENABLED) && defined(PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_is_le_implemented(pe, size, scope, le_addr, tma_addr);
#else
    return false;
#endif
}

#if LE_HW_SW_REQUIREMENTS_MET
#include <cuda_awbarrier_primitives.h>  // __mbarrier_*
#include "non_abi/device/pt-to-pt/tma_device.cuh"

enum class le_fabric_handle_kind {
    Unicast,
    Multicast,
};

// Fabric Handle ​
template <le_fabric_handle_kind K>
struct cft_handle {
    // Constructs Fabric Handle from Logical Endpoint Identifier and Offset:
    __host__ __device__ cft_handle() : id_(0), offset_(0), type_(K) {}
    __host__ __device__ cft_handle(CUlogicalEndpointId id, uint64_t off)
        : id_(id), offset_(off), type_(K) {}

    // Reads Logical Endpoint Identifier
    __host__ __device__ CUlogicalEndpointId id() const noexcept { return id_; }

    // Reads Offset
    __host__ __device__ uint64_t offset() const noexcept { return offset_; }

    // Swaps Logical Endpoint Identifier
    __host__ __device__ CUlogicalEndpointId swap_id(CUlogicalEndpointId id) {
        CUlogicalEndpointId old_id = id_;
        id_ = id;
        return old_id;
    }
    // Swaps Offset
    __host__ __device__ uint64_t swap_offset(uint64_t off) {
        uint64_t old_offset = offset_;
        offset_ = off;
        return old_offset;
    }

    __host__ __device__ bool is_unicast() const noexcept {
        return type_ == le_fabric_handle_kind::Unicast;
    }
    __host__ __device__ bool is_multicast() const noexcept {
        return type_ == le_fabric_handle_kind::Multicast;
    }

   private:
    CUlogicalEndpointId id_;
    le_fabric_handle_kind type_;
    uint64_t offset_;
};

template <le_fabric_handle_kind K>
using nvshmemi_fabric_handle = cft_handle<K>;

template <le_fabric_handle_kind K>
__host__ __device__ __forceinline__ nvshmemi_fabric_handle<K>
nvshmemi_fabric_handle_for_le_id(CUlogicalEndpointId peer_le_id, const void *heap_addr) {
    return nvshmemi_fabric_handle<K>(
        peer_le_id,
        (uint64_t)(reinterpret_cast<const char *>(heap_addr) -
                   reinterpret_cast<const char *>(nvshmemi_device_state_d.heap_base)));
}

__host__ __device__ __forceinline__ nvshmemi_fabric_handle<le_fabric_handle_kind::Unicast>
nvshmemi_fabric_handle_for_pe(int pe, const void *heap_addr) {
    return nvshmemi_fabric_handle_for_le_id<le_fabric_handle_kind::Unicast>(
        nvshmemi_ld_and_get_le_id(pe), heap_addr);
}

/** PTX mbarrier.try_wait* … b64 rdy|cndInv — both predicate results as bool. */
struct mbarrier_try_wait_rdy_cnd {
    bool rdy;
    bool cnd_inv;
};

// to be allocated in shared memory
struct handle_barrier_t {
    alignas(16) __mbarrier_t bar;

    // for fabric programming, mbarrier must be initialized with layout::v1
    inline __device__ void init(int arvCnt) {
        // SMEM address for [%0]: use b64/"l" when PTX uses .address_size 64 (e.g. sm_100+).
        const unsigned long long smem_addr =
            static_cast<unsigned long long>(__cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile ("mbarrier.init.shared.layout::v1.b64 [%0], %1;"
                :: "l"(smem_addr), "r"(arvCnt) : "memory");
    }

    // track completion in complete_tx::16B
    inline __device__ uint64_t arrive_relaxed(uint32_t size_bytes) {
        uint64_t state;
        uint32_t adjusted_size = ((size_bytes + (CFT_HANDLE_TX_SIZE - 1)) / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;
        const unsigned long long smem_addr =
            static_cast<unsigned long long>(__cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        /* Note: Expected TX is updated in chunks of 16B and not bytes
         * try_put will increment complete_tx in chunks of 16B
         */
        asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 %0, [%1], %2;"
                     : "=l"(state)
                     : "l"(smem_addr), "r"(adjusted_size / CFT_HANDLE_TX_SIZE)
                     : "memory");
        return state;
    }

    // mbarrier.try_wait.phase_type::primary…b64 rdy|cndInv, [mbar], state
    inline __device__ mbarrier_try_wait_rdy_cnd try_wait_token_rdy_cnd(uint64_t state) {
        unsigned rdy_u = 0, cnd_u = 0;
        const unsigned long long smem_addr =
            static_cast<unsigned long long>(__cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile(
            "{\n\t"
            ".reg .pred p_rdy;\n\t"
            ".reg .pred p_cnd;\n\t"
            "mbarrier.try_wait.phase_type::primary.acquire.cta.shared::cta.b64 "
            "p_rdy|p_cnd, [%2], %3;\n\t"
            "selp.b32 %0, 1, 0, p_rdy;\n\t"
            "selp.b32 %1, 1, 0, p_cnd;\n\t"
            "}"
            : "=r"(rdy_u), "=r"(cnd_u)
            : "l"(smem_addr), "l"(state)
            : "memory");
        return {rdy_u != 0u, cnd_u != 0u};
    }

    // mbarrier.try_wait.parity.phase_type::primary…b64 rdy|cndInv, [mbar], phaseParity
    inline __device__ mbarrier_try_wait_rdy_cnd try_wait_parity_rdy_cnd(int phase_parity) {
        unsigned rdy_u = 0, cnd_u = 0;
        const unsigned long long smem_addr =
            static_cast<unsigned long long>(__cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile(
            "{\n\t"
            ".reg .pred p_rdy;\n\t"
            ".reg .pred p_cnd;\n\t"
            "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 "
            "p_rdy|p_cnd, [%2], %3;\n\t"
            "selp.b32 %0, 1, 0, p_rdy;\n\t"
            "selp.b32 %1, 1, 0, p_cnd;\n\t"
            "}"
            : "=r"(rdy_u), "=r"(cnd_u)
            : "l"(smem_addr), "r"(phase_parity)
            : "memory");
        return {rdy_u != 0u, cnd_u != 0u};
    }

    inline __device__ bool try_wait_phase(int phase_parity) {
        return try_wait_parity_rdy_cnd(phase_parity).rdy;
    }

    /** Same as try_wait(state).rdy; optional err reports cnd_inv (e.g. invalid wait). */
    inline __device__ bool try_wait_token_with_err(uint64_t state, uint8_t *err) {
        mbarrier_try_wait_rdy_cnd t = try_wait_token_rdy_cnd(state);
        if (err) *err = t.cnd_inv ? uint8_t{1} : uint8_t{0};
        return t.rdy;
    }

    inline __device__ bool try_wait_token(uint64_t state) {
        uint8_t err = 0;
	    while(!try_wait_token_with_err(state, &err)) {}
        assert(err == 0);
        return true;
    }

    // wait till data has been read from shared memory
    inline __device__ void fabric_wait_sync_reads()
    {
        asm volatile("fabric.wait.sync_restrict::reads;\n" ::: "memory");
    }

    inline __device__ void inval() { __mbarrier_inval(&bar); }

};

/*
 * Fabric operations
 */

inline __device__ uint16_t size_to_bytemask_low_first(unsigned size_bytes) {
    if (size_bytes >= 16) return 0xFFFFu;
    return static_cast<uint16_t>((1u << size_bytes) - 1u);
}

 /* try put */
template <le_fabric_handle_kind K>
inline constexpr bool dependent_false_v = false;

template <le_fabric_handle_kind K>
__device__ inline void
fabric_try_put_async(CUlogicalEndpointId dst_le_id, uint64_t dst_data_off,
                     const void* src_in_shared_memory, uint32_t  size_bytes,
                     handle_barrier_t* bar)
{
    static_assert(dependent_false_v<K>, "Unknown handle kind");
}

template <>
__device__ inline void
fabric_try_put_async<le_fabric_handle_kind::Unicast>(CUlogicalEndpointId dst_le_id, uint64_t dst_data_off,
                     const void* src_in_shared_memory, uint32_t  size_bytes, handle_barrier_t* hbar)
{
    // Issue single instruction for size_bytes multiple of 16B
    uint32_t adjusted_size = (size_bytes / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;

    // .shared::cta operands need SMEM offsets from __cvta_generic_to_shared (generic ptr is wrong).
    unsigned long long src_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(src_in_shared_memory));
    const unsigned long long bar_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(
            reinterpret_cast<void*>(&(hbar->bar))));
    if (adjusted_size) {
        asm volatile(
            "fabric.try_put.async.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
            "[%0, %1], [%2], %3, [%4];\n"
            :
            : "r"(dst_le_id),             // %0: .b32 dstLeId
              "l"(dst_data_off),          // %1: .b64 dstDataOff
              "l"(src_smem),              // %2: .ptr .shared src
              "r"(adjusted_size),         // %3: .b32 size
              "l"(bar_smem)               // %4: .ptr .shared .b64 mbarrier
            : "memory");
    }

    // Remainder is done using cp_mask variant
    size_bytes -= adjusted_size;
    dst_data_off += adjusted_size;
    src_smem += adjusted_size;
    assert(size_bytes <= CFT_HANDLE_TX_SIZE);
    if (size_bytes) {
        uint16_t bytemask = size_to_bytemask_low_first(size_bytes);
        /* Note: completion is tracked in 16B units, so on using cp_mask
         * we still specify size as 16B but only store based on bytemask
         * which is essential for complete_tx tracking
         */
        asm volatile(
            "fabric.try_put.async.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
            "[%0, %1], [%2], %3, [%4], %5;\n"
            :
            : "r"(dst_le_id),
              "l"(dst_data_off),
              "l"(src_smem),
              "r"(CFT_HANDLE_TX_SIZE),
              "l"(bar_smem),
              "h"(bytemask)
            : "memory");
    }
}

/* try_get here uses cp_async_bulk_global_to_shared to copy data from global to shared memory.
 * the completion mechanism is mbarrier. mbarrier.complete_tx is implicitly called which will
 * increment the tx_count of mbarrier by the number of BYTES copied.
 * Since we use expect_tx() to update tx_count in chunks of CFT_HANDLE_TX_SIZE (16B) and not bytes
 * we call an additional barrier_arrive_relaxed() to update tx_count by size -
 * (size/CFT_HANDLE_TX_SIZE) to match the number of bytes copied.
 */
__device__ __forceinline__ void
fabric_try_get_async(CUlogicalEndpointId src_le_id, uint64_t src_data_off,
                     void* dst_in_shared_memory, uint32_t  size_bytes,
                     handle_barrier_t* hbar)
{
    // PTX requires a .shared address when .dst = .shared::cta
    unsigned long long dst_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(dst_in_shared_memory));
    const unsigned long long bar_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(
            reinterpret_cast<void*>(&(hbar->bar))));
    asm volatile(
        // fabric.try_get.async.dst.completion_mechanism{.level::cache_hint}.sem.sco.b128
        "fabric.try_get.async.shared::cta."
        "mbarrier::complete_tx::bytes.mbarrier::report::fabric."
        "relaxed.sys.b128 "
        "[%0], [%1, %2], %3, [%4];\n"
        :
        : "l"(dst_smem),       // %0: dst (.ptr .shared)
          "r"(src_le_id),      // %1: srcLeId (.b32)
          "l"(src_data_off),   // %2: srcDataOff (.b64)
          "r"(size_bytes),     // %3: size (.b32)
          "l"(bar_smem)        // %4: mbarrier (.ptr .shared)
        : "memory");
    // remainder expect_tx is done in arrive_relaxed()
    barrier_expect_tx(&(hbar->bar),
                      size_bytes - (size_bytes / CFT_HANDLE_TX_SIZE));
}


inline __device__ void fabric_submit() {
   asm volatile ("fabric.submit;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2generic_release_system() {
   asm volatile ("fence.proxy.generic::fabric.alias.release.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2generic_acquire_system() {
   asm volatile ("fence.proxy.generic::fabric.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2generic_alias() {
    fence_proxy_fabric2generic_acquire_system();
    fence_proxy_fabric2generic_release_system();
}

inline __device__ void fence_proxy_generic2fabric_release_system() {
   asm volatile ("fence.proxy.fabric::generic.alias.release.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_generic2fabric_acquire_system() {
   asm volatile ("fence.proxy.fabric::generic.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_generic2fabric_alias() {
    fence_proxy_generic2fabric_acquire_system();
    fence_proxy_generic2fabric_release_system();
}

inline __device__ void fence_proxy_fabric2fabric_release_system() {
   asm volatile ("fence.proxy.fabric::fabric.alias.release.sys;\n" ::: "memory");
}
inline __device__ void fence_proxy_fabric2fabric_acquire_system() {
   asm volatile ("fence.proxy.fabric::fabric.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2fabric_alias() {
    fence_proxy_fabric2fabric_acquire_system();
    fence_proxy_fabric2fabric_release_system();
}
#endif  // LE_HW_SW_REQUIREMENTS_MET

#endif  // __CUDACC_RTC__

#endif  // __CUDA_ARCH__
#endif  // __logical_endpoint_device_cuh__
