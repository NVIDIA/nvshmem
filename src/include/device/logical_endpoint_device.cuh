/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __logical_endpoint_device_cuh__
#define __logical_endpoint_device_cuh__

#include "non_abi/nvshmem_build_options.h"

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

template <bool REQUIRE_TX_SIZE_MULTIPLE>
__device__ __forceinline__ bool nvshmemi_is_le_implemented(int, size_t, threadgroup_t, const void *,
                                                           const void *) {
    return false;
}

__device__ __forceinline__ bool nvshmemi_is_le_implemented(int) { return false; }

__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int) { return false; }

template <bool REQUIRE_TX_SIZE_MULTIPLE>
__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int, size_t, threadgroup_t,
                                                                         const void *,
                                                                         const void *) {
    return false;
}

__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int) { return false; }

#else

#include "device_host/logical_endpoint_types.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

inline constexpr int NVSHMEMI_SMEM_BUF_SIZE = 512;  // 16*32 - 16B per thread, 1 buf per warp

__device__ __forceinline__ bool nvshmemi_ld_and_check_valid_le_id(int pe) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const long long unsigned* le_ids =
        reinterpret_cast<const long long unsigned*>(nvshmemi_device_state_d.unicast_le_ids_);
    return IS_VALID_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return false;
#endif
}

__device__ __forceinline__ CUlogicalEndpointId nvshmemi_ld_and_get_le_id(int pe) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const long long unsigned* le_ids =
        reinterpret_cast<const long long unsigned*>(nvshmemi_device_state_d.unicast_le_ids_);
    return (CUlogicalEndpointId)PARSE_LE_ID((uint64_t)__ldg(le_ids + pe));
#else
    return (CUlogicalEndpointId)-1;
#endif
}

__device__ bool nvshmemi_tma_smem_registered();
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
__device__ __forceinline__ size_t nvshmemi_smem_data_buf_size(size_t num_buffers);
__device__ constexpr bool nvshmemi_tma_is_16b_aligned(size_t value);

template <threadgroup_t SCOPE>
__device__ __forceinline__ size_t nvshmemi_handle_smem_chunk_size() {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        return 0;
    } else if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        return nvshmemi_smem_data_buf_size(TMA_COPY_NUM_STAGES);
    } else {
        size_t warps_per_threadgroup =
            (nvshmemi_threadgroup_size<SCOPE>() + NVSHMEMI_WARP_SIZE - 1) / NVSHMEMI_WARP_SIZE;
        return warps_per_threadgroup * NVSHMEMI_SMEM_BUF_SIZE;
    }
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ bool is_thrdgrp_smem_rsc_available(size_t smem_chunk_size,
                                                              size_t barrier_slots_per_group) {
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        return false;
    }

    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_WARPGROUP) {
        const size_t threads_per_cta = blockDim.x * blockDim.y * blockDim.z;
        if ((threads_per_cta % (4 * NVSHMEMI_WARP_SIZE)) != 0) return false;
    }

    if (smem_chunk_size == 0 || barrier_slots_per_group == 0) return false;

    size_t stage_bytes = nvshmemi_smem_data_buf_size(TMA_COPY_NUM_STAGES);
    size_t max_thrdgrps_by_smem = stage_bytes / smem_chunk_size;
    size_t max_thrdgrps_by_barrier = NVSHMEMI_NUM_HANDLE_BARRIER_SLOTS / barrier_slots_per_group;
    size_t max_thrdgrps = max_thrdgrps_by_smem < max_thrdgrps_by_barrier ? max_thrdgrps_by_smem
                                                                         : max_thrdgrps_by_barrier;
    uint32_t tid_in_block = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>();
    uint32_t thrdgrp_idx = tid_in_block / nvshmemi_threadgroup_size<SCOPE>();

    return thrdgrp_idx < max_thrdgrps;
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ bool is_thrdgrp_smem_rsc_available(size_t smem_chunk_size) {
    return is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size, TMA_COPY_NUM_STAGES);
}

template <threadgroup_t SCOPE, int SMEM_CHUNK_SIZE>
__device__ __forceinline__ bool is_thrdgrp_smem_rsc_available() {
    return is_thrdgrp_smem_rsc_available<SCOPE>((size_t)SMEM_CHUNK_SIZE);
}

__device__ __forceinline__ bool is_thrdgrp_smem_rsc_available(threadgroup_t scope) {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_BLOCK:
            return is_thrdgrp_smem_rsc_available<NVSHMEMI_THREADGROUP_BLOCK>(
                nvshmemi_handle_smem_chunk_size<NVSHMEMI_THREADGROUP_BLOCK>());
        case NVSHMEMI_THREADGROUP_WARP:
            return is_thrdgrp_smem_rsc_available<NVSHMEMI_THREADGROUP_WARP>(
                nvshmemi_handle_smem_chunk_size<NVSHMEMI_THREADGROUP_WARP>());
        case NVSHMEMI_THREADGROUP_WARPGROUP:
            return is_thrdgrp_smem_rsc_available<NVSHMEMI_THREADGROUP_WARPGROUP>(
                nvshmemi_handle_smem_chunk_size<NVSHMEMI_THREADGROUP_WARPGROUP>());
        default:
            return false;
    }
}
#endif

__device__ __forceinline__ bool nvshmemi_is_addr_offset_aligned(const void* addr, size_t size) {
    if (addr == nullptr) return false;
    if (size == 0) return true;
    return ((((uintptr_t)addr - (uintptr_t)nvshmemi_device_state_d.heap_base) % size) == 0);
}

template <bool REQUIRE_TX_SIZE_MULTIPLE>
__device__ __forceinline__ bool nvshmemi_is_le_implemented(int pe, size_t size, threadgroup_t scope,
                                                           const void* le_addr,
                                                           const void* tma_addr) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if constexpr (REQUIRE_TX_SIZE_MULTIPLE) {
        if (size == 0 || (size % CFT_HANDLE_TX_SIZE) != 0) return false;
    }

    /* The handle path stages global local buffers through shared memory; shared
     * local buffers can be consumed or produced by fabric directly. */
    return ((scope != NVSHMEMI_THREADGROUP_THREAD) && nvshmemi_tma_smem_registered() &&
            is_thrdgrp_smem_rsc_available(scope) &&
            nvshmemi_is_addr_offset_aligned(le_addr, CFT_HANDLE_TX_SIZE) &&
            nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)tma_addr) &&
            nvshmemi_ld_and_check_valid_le_id(pe));
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_le_implemented(int pe) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    const size_t required_smem_size =
        static_cast<size_t>(CFT_HANDLE_TX_SIZE) * blockDim.x * blockDim.y * blockDim.z;
    return (nvshmemi_tma_smem_registered() &&
            nvshmemi_smem_data_buf_size(1) >= required_smem_size &&
            nvshmemi_ld_and_check_valid_le_id(pe));
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_multicast_le_implemented(uint64_t le_id_with_flag,
                                                                     size_t size,
                                                                     threadgroup_t scope) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if (size == 0 || !nvshmemi_tma_smem_registered() || !IS_VALID_LE_ID(le_id_with_flag) ||
        ((size % CFT_HANDLE_TX_SIZE) != 0)) {
        return false;
    }

    return is_thrdgrp_smem_rsc_available(scope);
#else
    return false;
#endif
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ bool nvshmemi_is_multicast_reduce_le_implemented(
    uint64_t le_id_with_flag, size_t size) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if constexpr (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        return false;
    }

    const size_t threadgroup_size = nvshmemi_threadgroup_size<SCOPE>();
    if ((threadgroup_size % NVSHMEMI_WARP_SIZE) != 0) {
        return false;
    }

    const size_t warps_per_threadgroup = threadgroup_size / NVSHMEMI_WARP_SIZE;
    const size_t smem_chunk_size = warps_per_threadgroup * NVSHMEMI_SMEM_BUF_SIZE;
    const size_t barrier_slots_per_group = warps_per_threadgroup * TMA_COPY_NUM_STAGES;

    return (size != 0 && warps_per_threadgroup != 0 && nvshmemi_tma_smem_registered() &&
            IS_VALID_LE_ID(le_id_with_flag) &&
            is_thrdgrp_smem_rsc_available<SCOPE>(smem_chunk_size, barrier_slots_per_group) &&
            ((size % CFT_HANDLE_TX_SIZE) == 0));
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_le_prioritized(int pe) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_ld_and_check_valid_le_id(pe);
#else
    return false;
#endif
}

template <bool REQUIRE_TX_SIZE_MULTIPLE>
__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int pe, size_t size,
                                                                         threadgroup_t scope,
                                                                         const void* le_addr,
                                                                         const void* tma_addr) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_is_le_implemented<REQUIRE_TX_SIZE_MULTIPLE>(pe, size, scope, le_addr, tma_addr);
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_le_supported_and_prioritized(int pe) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_is_le_implemented(pe);
#else
    return false;
#endif
}

__device__ __forceinline__ bool nvshmemi_is_multicast_le_supported_and_prioritized(
    uint64_t le_id_with_flag, size_t size, threadgroup_t scope) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_is_multicast_le_implemented(le_id_with_flag, size, scope);
#else
    return false;
#endif
}

template <threadgroup_t SCOPE>
__device__ __forceinline__ bool nvshmemi_is_multicast_reduce_le_supported_and_prioritized(
    uint64_t le_id_with_flag, size_t size) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT) && \
    LE_HW_SW_REQUIREMENTS_MET
    return nvshmemi_is_multicast_reduce_le_implemented<SCOPE>(le_id_with_flag, size);
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
__host__ __device__ __forceinline__ nvshmemi_fabric_handle<K> nvshmemi_fabric_handle_for_le_id(
    CUlogicalEndpointId peer_le_id, const void* heap_addr) {
    return nvshmemi_fabric_handle<K>(
        peer_le_id, (uint64_t)(reinterpret_cast<const char*>(heap_addr) -
                               reinterpret_cast<const char*>(nvshmemi_device_state_d.heap_base)));
}

__host__ __device__ __forceinline__ nvshmemi_fabric_handle<le_fabric_handle_kind::Unicast>
nvshmemi_fabric_handle_for_pe(int pe, const void* heap_addr) {
    return nvshmemi_fabric_handle_for_le_id<le_fabric_handle_kind::Unicast>(
        nvshmemi_ld_and_get_le_id(pe), heap_addr);
}

enum class mbarrier_primary_wait_status : uint8_t {
    complete,
    complete_with_report,
};

struct mbarrier_primary_wait_raw_result {
    bool wait_complete;
    bool report;
};

// to be allocated in shared memory
struct alignas(16) handle_barrier_t {
    __mbarrier_t bar;

    /*
     * A handle barrier occupies a 16-byte reserved SMEM slot while the
     * hardware mbarrier itself occupies 8 bytes.  Keep pending fabric handle
     * completion state in the remaining bytes so a caller can return or defer
     * waiting without invalidating the barrier that will report completion.
     *
     * pending_handle_bytes is counted in complete_tx::16B units (expressed as
     * bytes) rather than logical payload bytes.  This matters for cp_mask
     * operations, where a sub-16-byte PUT still reports one 16-byte
     * completion event.
     */
    uint32_t pending_handle_bytes;
    uint16_t pending_handle_owner;
    uint8_t pending_handle_active;

    inline __device__ void reset_pending_handle_state() {
        pending_handle_bytes = 0;
        pending_handle_owner = 0;
        pending_handle_active = 0;
    }

    inline __device__ void init_raw(int arvCnt) {
        const unsigned long long smem_addr = static_cast<unsigned long long>(
            __cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile("mbarrier.init.shared.layout::v1.b64 [%0], %1;" ::"l"(smem_addr), "r"(arvCnt)
                     : "memory");
    }

    inline __device__ void inval_raw() { __mbarrier_inval(&bar); }

    inline __device__ uint32_t handle_completion_bytes(uint32_t size_bytes) const {
        return ((size_bytes + (CFT_HANDLE_TX_SIZE - 1)) / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;
    }

    inline __device__ bool has_pending_handle() const { return pending_handle_active != 0; }

    inline __device__ bool pending_handle_is_owned_by(uint32_t owner) const {
        return has_pending_handle() && pending_handle_owner == owner;
    }

    /* Complete the current handle operation batch. Keeping the barrier valid advances it
     * to the next phase so more handle operations can be accumulated in the same slot. */
    inline __device__ void drain_pending_handle(bool invalidate = true) {
        if (!has_pending_handle()) return;

        if (pending_handle_bytes != 0) {
            uint64_t state = arrive_relaxed(pending_handle_bytes);
            wait_primary(state);
            pending_handle_bytes = 0;
        }

        if (invalidate) {
            inval_raw();
            reset_pending_handle_state();
        }
    }

    /* Reuse a live handle barrier when the same threadgroup issues another
     * operation. A different owner implies a slot collision with another handle
     * path, which is resolved by completing the old batch first. */
    inline __device__ void prepare_handle(uint32_t owner) {
        if (has_pending_handle() && pending_handle_owner != owner) drain_pending_handle(true);

        if (!has_pending_handle()) {
            init_raw(1);
            pending_handle_bytes = 0;
            pending_handle_owner = static_cast<uint16_t>(owner);
            pending_handle_active = 1;
        }
    }

    inline __device__ void ensure_handle_tx_capacity(uint32_t size_bytes) {
        uint32_t completion_bytes = handle_completion_bytes(size_bytes);
        if (pending_handle_bytes != 0 &&
            pending_handle_bytes + completion_bytes >= TMA_PUT_MAX_BATCH_SIZE) {
            drain_pending_handle(false);
        }
    }

    inline __device__ void ensure_handle_get_tx_capacity(uint32_t size_bytes) {
        if (pending_handle_bytes != 0 &&
            pending_handle_bytes + size_bytes >= TMA_GET_MAX_BATCH_SIZE) {
            drain_pending_handle(false);
        }
    }

    inline __device__ void record_pending_handle(uint32_t size_bytes) {
        pending_handle_bytes += handle_completion_bytes(size_bytes);
    }

    inline __device__ void record_pending_get_handle(uint32_t size_bytes) {
        pending_handle_bytes += size_bytes;
    }

    // for fabric programming, mbarrier must be initialized with layout::v1
    inline __device__ void init(int arvCnt) {
        /* Other handle paths share these slots.  They must not reinitialize a
         * barrier that is still carrying pending fabric completions. */
        drain_pending_handle(true);
        init_raw(arvCnt);
    }

    // only 1 thread in warp initializes the barrier
    inline __device__ void init(int arvCnt, int myIdx) {
        if (myIdx % warpSize == 0) {
            init(arvCnt);
        }
    }

    // track completion in complete_tx::16B
    inline __device__ uint64_t arrive_relaxed(uint32_t size_bytes) {
        uint64_t state;
        uint32_t adjusted_size =
            ((size_bytes + (CFT_HANDLE_TX_SIZE - 1)) / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;
        const unsigned long long smem_addr = static_cast<unsigned long long>(
            __cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        /* Note: Expected TX is updated in chunks of 16B and not bytes
         * try_put will increment complete_tx in chunks of 16B
         */
        asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 %0, [%1], %2;"
                     : "=l"(state)
                     : "l"(smem_addr), "r"(adjusted_size / CFT_HANDLE_TX_SIZE)
                     : "memory");
        return state;
    }

    inline __device__ mbarrier_primary_wait_raw_result try_wait_primary_raw_result(uint64_t state) {
        unsigned wait_complete = 0;
        unsigned report = 0;
        const unsigned long long smem_addr = static_cast<unsigned long long>(
            __cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile(
            "{\n\t"
            ".reg .pred p_wait_complete;\n\t"
            ".reg .pred p_report;\n\t"
            "mbarrier.try_wait.phase_type::primary.acquire.cta.shared::cta.b64 "
            "p_wait_complete|p_report, [%2], %3;\n\t"
            "selp.b32 %0, 1, 0, p_wait_complete;\n\t"
            "selp.b32 %1, 1, 0, p_report;\n\t"
            "}"
            : "=r"(wait_complete), "=r"(report)
            : "l"(smem_addr), "l"(state)
            : "memory");
        return {wait_complete != 0, report != 0};
    }

    inline __device__ mbarrier_primary_wait_raw_result
    try_wait_primary_by_parity_raw_result(int phase_parity) {
        unsigned wait_complete = 0;
        unsigned report = 0;
        const unsigned long long smem_addr = static_cast<unsigned long long>(
            __cvta_generic_to_shared(reinterpret_cast<void*>(&bar)));
        asm volatile(
            "{\n\t"
            ".reg .pred p_wait_complete;\n\t"
            ".reg .pred p_report;\n\t"
            "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 "
            "p_wait_complete|p_report, [%2], %3;\n\t"
            "selp.b32 %0, 1, 0, p_wait_complete;\n\t"
            "selp.b32 %1, 1, 0, p_report;\n\t"
            "}"
            : "=r"(wait_complete), "=r"(report)
            : "l"(smem_addr), "r"(phase_parity)
            : "memory");
        return {wait_complete != 0, report != 0};
    }

    inline __device__ bool try_wait_primary_raw(uint64_t state, uint8_t& report) {
        mbarrier_primary_wait_raw_result result = try_wait_primary_raw_result(state);
        report = result.report ? uint8_t{1} : uint8_t{0};
        return result.wait_complete;
    }

    inline __device__ bool try_wait_primary_by_parity_raw(int phase_parity, uint8_t& report) {
        mbarrier_primary_wait_raw_result result =
            try_wait_primary_by_parity_raw_result(phase_parity);
        report = result.report ? uint8_t{1} : uint8_t{0};
        return result.wait_complete;
    }

    inline __device__ mbarrier_primary_wait_status wait_primary_status(uint64_t state) {
        uint8_t report = 0;
        while (!try_wait_primary_raw(state, report)) {
        }
        return report ? mbarrier_primary_wait_status::complete_with_report
                      : mbarrier_primary_wait_status::complete;
    }

    inline __device__ mbarrier_primary_wait_status wait_primary_by_parity_status(int phase_parity) {
        uint8_t report = 0;
        while (!try_wait_primary_by_parity_raw(phase_parity, report)) {
        }
        return report ? mbarrier_primary_wait_status::complete_with_report
                      : mbarrier_primary_wait_status::complete;
    }

    inline __device__ void wait_primary(uint64_t state) {
        [[maybe_unused]] mbarrier_primary_wait_status status = wait_primary_status(state);
        assert(status == mbarrier_primary_wait_status::complete);
    }

    inline __device__ void wait_primary_by_parity(int phase_parity) {
        [[maybe_unused]] mbarrier_primary_wait_status status =
            wait_primary_by_parity_status(phase_parity);
        assert(status == mbarrier_primary_wait_status::complete);
    }

    // wait till data has been read from shared memory
    inline __device__ void fabric_wait_sync_reads() {
        asm volatile("fabric.wait.sync_restrict::reads;\n" ::: "memory");
    }

    inline __device__ void inval() {
        inval_raw();
        reset_pending_handle_state();
    }
    inline __device__ void inval(int myIdx) {
        if (myIdx % warpSize == 0) inval();
    }
};

static_assert(sizeof(handle_barrier_t) == 16,
              "handle_barrier_t must fit in one 16-byte handle barrier SMEM slot");
static_assert(alignof(handle_barrier_t) == 16,
              "handle_barrier_t must be aligned to a 16-byte handle barrier SMEM slot");

/*
 * Fabric operations
 */

inline __device__ uint16_t size_to_bytemask_low_first(unsigned size_bytes) {
    if (size_bytes >= 16) return 0xFFFFu;
    return static_cast<uint16_t>((1u << size_bytes) - 1u);
}

inline __device__ uint16_t byte_range_to_bytemask_low_first(unsigned byte_offset,
                                                            unsigned size_bytes) {
    assert(byte_offset < CFT_HANDLE_TX_SIZE);
    assert(size_bytes <= CFT_HANDLE_TX_SIZE);
    assert(byte_offset + size_bytes <= CFT_HANDLE_TX_SIZE);
    return static_cast<uint16_t>(size_to_bytemask_low_first(size_bytes) << byte_offset);
}

/* try put */
template <le_fabric_handle_kind K>
inline constexpr bool dependent_false_v = false;

template <le_fabric_handle_kind K>
__device__ inline void fabric_try_put_async(CUlogicalEndpointId dst_le_id, uint64_t dst_data_off,
                                            const void* src_in_shared_memory, uint32_t size_bytes,
                                            handle_barrier_t* bar) {
    static_assert(dependent_false_v<K>, "Unknown handle kind");
}

template <le_fabric_handle_kind K>
__device__ inline void fabric_try_put_async(CUlogicalEndpointId dst_le_id, uint64_t dst_data_off,
                                            const void* src_in_shared_memory, uint16_t bytemask,
                                            handle_barrier_t* bar) {
    static_assert(dependent_false_v<K>, "Unknown handle kind");
}

template <>
__device__ inline void fabric_try_put_async<le_fabric_handle_kind::Unicast>(
    CUlogicalEndpointId dst_le_id, uint64_t dst_data_off, const void* src_in_shared_memory,
    uint16_t bytemask, handle_barrier_t* hbar) {
    if (!bytemask) return;
    assert((dst_data_off & (CFT_HANDLE_TX_SIZE - 1)) == 0);

    // .shared::cta operands need SMEM offsets from __cvta_generic_to_shared (generic ptr is wrong).
    unsigned long long src_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(src_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));

    /* Note: completion is tracked in 16B units, so on using cp_mask
     * we still specify size as 16B but only store based on bytemask
     * which is essential for complete_tx tracking
     */
    asm volatile(
        "fabric.try_put.async.shared::cta."
        "mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
        "[%0, %1], [%2], %3, [%4], %5;\n"
        :
        : "r"(dst_le_id), "l"(dst_data_off), "l"(src_smem), "r"(CFT_HANDLE_TX_SIZE), "l"(bar_smem),
          "h"(bytemask)
        : "memory");
}

template <>
__device__ inline void fabric_try_put_async<le_fabric_handle_kind::Unicast>(
    CUlogicalEndpointId dst_le_id, uint64_t dst_data_off, const void* src_in_shared_memory,
    uint32_t size_bytes, handle_barrier_t* hbar) {
    // Issue single instruction for size_bytes multiple of 16B
    uint32_t adjusted_size = (size_bytes / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;

    // .shared::cta operands need SMEM offsets from __cvta_generic_to_shared (generic ptr is wrong).
    unsigned long long src_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(src_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));
    if (adjusted_size) {
        asm volatile(
            "fabric.try_put.async.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
            "[%0, %1], [%2], %3, [%4];\n"
            :
            : "r"(dst_le_id),      // %0: .b32 dstLeId
              "l"(dst_data_off),   // %1: .b64 dstDataOff
              "l"(src_smem),       // %2: .ptr .shared src
              "r"(adjusted_size),  // %3: .b32 size
              "l"(bar_smem)        // %4: .ptr .shared .b64 mbarrier
            : "memory");
    }

    // Remainder is done using cp_mask variant
    size_bytes -= adjusted_size;
    dst_data_off += adjusted_size;
    assert(size_bytes <= CFT_HANDLE_TX_SIZE);
    if (size_bytes) {
        uint16_t bytemask = size_to_bytemask_low_first(size_bytes);
        fabric_try_put_async<le_fabric_handle_kind::Unicast>(
            dst_le_id, dst_data_off,
            reinterpret_cast<const void*>(reinterpret_cast<const char*>(src_in_shared_memory) +
                                          adjusted_size),
            bytemask, hbar);
    }
}

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && !defined(__clang_llvm_bitcode_lib__) && \
    !defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
__device__ inline void fabric_try_put_counted_async(CUlogicalEndpointId dst_le_id,
                                                    uint64_t dst_data_off, uint64_t dst_count_off,
                                                    const void* src_in_shared_memory,
                                                    uint32_t size_bytes, handle_barrier_t* hbar) {
    const unsigned long long src_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(src_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));
    if (size_bytes) {
        asm volatile(
            "fabric.try_put.async.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric."
            "counted::bytes.relaxed.sys.b128 "
            "[%0, %1, %2], [%3], %4, [%5];\n"
            :
            : "r"(dst_le_id), "l"(dst_data_off), "l"(dst_count_off), "l"(src_smem), "r"(size_bytes),
              "l"(bar_smem)
            : "memory");
    }
}
#endif

template <>
__device__ inline void fabric_try_put_async<le_fabric_handle_kind::Multicast>(
    CUlogicalEndpointId dst_le_id, uint64_t dst_data_off, const void* src_in_shared_memory,
    uint32_t size_bytes, handle_barrier_t* hbar) {
    // Issue single instruction for size_bytes multiple of 16B
    uint32_t adjusted_size = (size_bytes / CFT_HANDLE_TX_SIZE) * CFT_HANDLE_TX_SIZE;

    // .shared::cta operands need SMEM offsets from __cvta_generic_to_shared (generic ptr is wrong).
    unsigned long long src_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(src_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));
    if (adjusted_size) {
        asm volatile(
            "fabric.try_put.async.multimem.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
            "[%0, %1], [%2], %3, [%4];\n"
            :
            : "r"(dst_le_id),      // %0: .b32 dstLeId
              "l"(dst_data_off),   // %1: .b64 dstDataOff
              "l"(src_smem),       // %2: .ptr .shared src
              "r"(adjusted_size),  // %3: .b32 size
              "l"(bar_smem)        // %4: .ptr .shared .b64 mbarrier
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
            "fabric.try_put.async.multimem.shared::cta."
            "mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
            "[%0, %1], [%2], %3, [%4], %5;\n"
            :
            : "r"(dst_le_id), "l"(dst_data_off), "l"(src_smem), "r"(CFT_HANDLE_TX_SIZE),
              "l"(bar_smem), "h"(bytemask)
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
__device__ __forceinline__ void fabric_try_get_async(CUlogicalEndpointId src_le_id,
                                                     uint64_t src_data_off,
                                                     void* dst_in_shared_memory,
                                                     uint32_t size_bytes, handle_barrier_t* hbar) {
    // PTX requires a .shared address when .dst = .shared::cta
    unsigned long long dst_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(dst_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));
    asm volatile(
        // fabric.try_get.async.dst.completion_mechanism{.level::cache_hint}.sem.sco.b128
        "fabric.try_get.async.shared::cta."
        "mbarrier::complete_tx::bytes.mbarrier::report::fabric."
        "relaxed.sys.b128 "
        "[%0], [%1, %2], %3, [%4];\n"
        :
        : "l"(dst_smem),      // %0: dst (.ptr .shared)
          "r"(src_le_id),     // %1: srcLeId (.b32)
          "l"(src_data_off),  // %2: srcDataOff (.b64)
          "r"(size_bytes),    // %3: size (.b32)
          "l"(bar_smem)       // %4: mbarrier (.ptr .shared)
        : "memory");
    // remainder expect_tx is done in arrive_relaxed()
    barrier_expect_tx(&(hbar->bar), size_bytes - (size_bytes / CFT_HANDLE_TX_SIZE));
}

//  try_pullred

// All threads in warp have to call this function with the
// same member mask (0xFFFFFFFF)
#define FABRIC_TRY_PULLRED_ASYNC_PTX(RDXN_OP, RDXN_TYPE)                                   \
    asm volatile(                                                                          \
        "fabric.try_pullred.async.multimem.shared::cta."                                   \
        "mbarrier::complete_tx::bytes.mbarrier::report::fabric."                           \
        "relaxed.sys." #RDXN_OP "." #RDXN_TYPE                                             \
        ".sync "                                                                           \
        "[%0], [%1, %2], %3, [%4], 0xffffffff;"                                            \
        :                                                                                  \
        : "l"(dst_smem), "r"(src_le_id), "l"(src_data_off), "r"(size_bytes), "l"(bar_smem) \
        : "memory")

/*
 * // valid combinations for sm_100
 * .op.ty = {
 * {.and, .or, .xor} x {.b32, .b64},
 * {.min, .max}      x {.u32, .s32, .u64, .s64, .f16, .bf16},
 * {.add}            x {.u32, .u64, .bf16, .f16, .f32}, // No .f64
 * }
 *
 * // Note: add these if applicable
 * // sm_100a, sm_101a, sm_120a, sm_121a,
 * // and sm_100f, sm_101f, sm_110f or higher in the same family:
 * .op.ty = {
 *   {.min, .max}      x {.e4m3, .e5m2},
 *   {.add.acc::f16}   x {.e4m3, .e5m2},
 *   {.add.acc::f32}   x {.f16, .bf16}
 * }
 */

template <typename T, rdxn_ops_t RDXN_OP>
__device__ constexpr inline bool is_handle_pullred_supported() {
    if constexpr (((RDXN_OP == RDXN_OPS_AND) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_OR) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_XOR) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_AND) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_OR) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_XOR) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, int32_t>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, int32_t>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, int64_t>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, int64_t>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, half>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, half>) ||
                  ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, __nv_bfloat16>) ||
                  ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, __nv_bfloat16>) ||
                  ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, uint32_t>) ||
                  ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, uint64_t>) ||
                  ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, __nv_bfloat16>) ||
                  ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, half>) ||
                  ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, float>)) {
        return true;
    }
    return false;
}

template <typename T, rdxn_ops_t RDXN_OP>
__device__ inline void fabric_try_pullred_async(CUlogicalEndpointId src_le_id,
                                                uint64_t src_data_off, void* dst_in_shared_memory,
                                                uint32_t size_bytes, handle_barrier_t* hbar) {
    // PTX requires a .shared address when .dst = .shared::cta
    unsigned long long dst_smem =
        static_cast<unsigned long long>(__cvta_generic_to_shared(dst_in_shared_memory));
    const unsigned long long bar_smem = static_cast<unsigned long long>(
        __cvta_generic_to_shared(reinterpret_cast<void*>(&(hbar->bar))));

    if constexpr ((RDXN_OP == RDXN_OPS_AND) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(and, b32);  // and.b32
    } else if constexpr ((RDXN_OP == RDXN_OPS_OR) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(or, b32);  // or.b32
    } else if constexpr ((RDXN_OP == RDXN_OPS_XOR) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(xor, b32);  // xor.b32
    } else if constexpr ((RDXN_OP == RDXN_OPS_AND) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(and, b64);  // and.b64
    } else if constexpr ((RDXN_OP == RDXN_OPS_OR) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(or, b64);  // or.b64
    } else if constexpr ((RDXN_OP == RDXN_OPS_XOR) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(xor, b64);  // xor.b64
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, u32);  // min.u32
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, u32);  // max.u32
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, int32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, s32);  // min.s32
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, int32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, s32);  // max.s32
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, u64);  // min.u64
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, u64);  // max.u64
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, int64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, s64);  // min.s64
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, int64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, s64);  // max.s64
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, half>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, f16);  // min.f16
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, half>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, f16);  // max.f16
    } else if constexpr ((RDXN_OP == RDXN_OPS_MIN) && std::is_same_v<T, __nv_bfloat16>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(min, bf16);  // min.bf16
    } else if constexpr ((RDXN_OP == RDXN_OPS_MAX) && std::is_same_v<T, __nv_bfloat16>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(max, bf16);  // max.bf16
    } else if constexpr ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, uint32_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(add, u32);  // add.u32
    } else if constexpr ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, uint64_t>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(add, u64);  // add.u64
    } else if constexpr ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, __nv_bfloat16>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(add, bf16);  // add.bf16
    } else if constexpr ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, half>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(add, f16);  // add.f16
    } else if constexpr ((RDXN_OP == RDXN_OPS_SUM) && std::is_same_v<T, float>) {
        FABRIC_TRY_PULLRED_ASYNC_PTX(add, f32);  // add.f32
    } else {
        assert(false && "Unsupported reduce operation");
    }
    // remainder expect_tx is done in arrive_relaxed()
    if ((threadIdx.x % warpSize) == 0) {
        barrier_expect_tx(&(hbar->bar), size_bytes - (size_bytes / CFT_HANDLE_TX_SIZE));
    }
}

inline __device__ void fabric_submit() { asm volatile("fabric.submit;\n" ::: "memory"); }

inline __device__ void fence_proxy_fabric2generic_release_system() {
    asm volatile("fence.proxy.generic::fabric.alias.release.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2generic_acquire_system() {
    asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2generic_alias() {
    fence_proxy_fabric2generic_acquire_system();
    fence_proxy_fabric2generic_release_system();
}

inline __device__ void fence_proxy_generic2fabric_release_system() {
    asm volatile("fence.proxy.fabric::generic.alias.release.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_generic2fabric_acquire_system() {
    asm volatile("fence.proxy.fabric::generic.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_generic2fabric_alias() {
    fence_proxy_generic2fabric_acquire_system();
    fence_proxy_generic2fabric_release_system();
}

inline __device__ void fence_proxy_fabric2fabric_release_system() {
    asm volatile("fence.proxy.fabric::fabric.alias.release.sys;\n" ::: "memory");
}
inline __device__ void fence_proxy_fabric2fabric_acquire_system() {
    asm volatile("fence.proxy.fabric::fabric.alias.acquire.sys;\n" ::: "memory");
}

inline __device__ void fence_proxy_fabric2fabric_alias() {
    fence_proxy_fabric2fabric_acquire_system();
    fence_proxy_fabric2fabric_release_system();
}
#endif  // LE_HW_SW_REQUIREMENTS_MET

#endif  // __CUDACC_RTC__

#endif  // __CUDA_ARCH__
#endif  // __logical_endpoint_device_cuh__
