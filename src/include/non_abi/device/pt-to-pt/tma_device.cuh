/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef TMA_DEVICE_CUH
#define TMA_DEVICE_CUH

#include <cuda_runtime.h>
#include "device_host/nvshmem_types.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

#ifdef __CUDA_ARCH__

__device__ constexpr size_t nvshmemi_tma_alignment_16() { return (size_t)1 << 4; }

__device__ constexpr size_t nvshmemi_tma_alignment_mask_16() {
    return nvshmemi_tma_alignment_16() - 1;
}

__device__ constexpr bool nvshmemi_tma_is_16b_aligned(size_t value) {
    return (value & nvshmemi_tma_alignment_mask_16()) == 0;
}

__device__ constexpr size_t nvshmemi_tma_align_down_16(size_t value) {
    return value & ~nvshmemi_tma_alignment_mask_16();
}

#if __CUDA_ARCH__ >= 900
#include <cuda_awbarrier_primitives.h>

/*
 * Elect one leader from the active threads in the calling warp.
 * Returns true for exactly one thread.  Uses PTX elect.sync (sm_90+).
 *
 * NOTE: This elects a leader within the CALLING warp.  When used inside a
 * multi-warp block, pair it with a warp_id == 0 guard (see
 * nvshmemi_tma_block_is_elected below) to restrict the issue to one thread
 * in the entire block.
 */
__device__ __forceinline__ bool nvshmemi_tma_elect_warp() {
    uint32_t is_leader;
    asm volatile(R"({
.reg .pred elect_p;
.reg .u32 elect_id;
elect.sync elect_id|elect_p, 0xffffffff;
selp.u32 %0, 1, 0, elect_p;
})"
                 : "=r"(is_leader));
    return (bool)is_leader;
}

/*
 * Select exactly one thread across an entire CTA to issue a TMA operation.
 * Matches the is_elected() pattern from the CUDA Programming Guide:
 *
 *   uint uniform_warp_id = __shfl_sync(0xffffffff, warp_id, 0);
 *   return (uniform_warp_id == 0) && elect_sync(0xffffffff);
 *
 * The __shfl_sync broadcast makes the warp_id check compiler-visible as a
 * warp-uniform value, preventing the compiler from inserting a peeling loop
 * over all active threads (which causes warp serialization).  Using
 * if (threadIdx.x == 0) alone is NOT sufficient for this reason.
 *
 * Call this from ALL threads in the block; returns true for exactly one.
 */
__device__ __forceinline__ bool nvshmemi_tma_block_is_elected() {
    unsigned int tid =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    unsigned int warp_id = tid / warpSize;
    /* Broadcast warp_id from lane 0 to make it a compiler-known uniform value */
    unsigned int uniform_warp_id = __shfl_sync(0xffffffff, warp_id, 0);
    return (uniform_warp_id == 0) && nvshmemi_tma_elect_warp();
}

/*
 * PTX helper: convert a generic pointer to a shared-memory-space 32-bit address.
 */
__device__ __forceinline__ unsigned int nvshmemi_tma_cvta_to_shared(const void *ptr) {
    unsigned int smem_addr;
    asm(R"({
.reg .u64 smem_u64;
cvta.to.shared.u64 smem_u64, %1;
cvt.u32.u64 %0, smem_u64;
})"
        : "=r"(smem_addr)
        : "l"((uint64_t)(uintptr_t)ptr));
    return smem_addr;
}

/*
 * PTX helper: issue cp.async.bulk from shared to global memory.
 */
__device__ __forceinline__ void nvshmemi_tma_bulk_shared_to_global(void *gmem_dst,
                                                                   unsigned int smem_addr,
                                                                   uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :
                 : "l"((uint64_t)(uintptr_t)gmem_dst), "r"(smem_addr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void nvshmemi_tma_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_read_0() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

/* Full completion wait: smem read AND global write both done. */
__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_0() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

inline __device__ void barrier_expect_tx(__mbarrier_t *barrier, uint32_t txCount) {
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(txCount)
        : "memory");
}

inline __device__ void cp_async_bulk_global_to_shared(void *dest, const void *src,
                                                      __mbarrier_t *barrier, uint32_t size) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dest));
    uint64_t gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
    uint32_t smem_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(smem_ptr), "l"(gmem_ptr), "r"(size), "r"(smem_barrier_ptr)
        : "memory");
}

inline __device__ void cp_async_bulk_shared_to_global(void *dest, const void *src, uint32_t size) {
    nvshmemi_tma_bulk_shared_to_global(dest, nvshmemi_tma_cvta_to_shared(src), size);
}

template <int n>
inline __device__ void cp_async_bulk_wait_group_read() {
    static_assert(n >= 0 && n <= 24, "n must be between 0 and 24");

#define NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(N)                      \
    if constexpr (n == N) {                                                 \
        asm volatile("cp.async.bulk.wait_group.read " #N ";" ::: "memory"); \
    }

    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(0)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(1)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(2)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(3)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(4)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(5)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(6)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(7)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(8)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(9)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(10)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(11)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(12)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(13)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(14)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(15)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(16)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(17)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(18)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(19)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(20)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(21)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(22)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(23)
    NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE(24)

#undef NVSHMEMI_CP_ASYNC_BULK_WAIT_GROUP_READ_CASE
}

inline __device__ __mbarrier_token_t barrier_arrive1_tx(__mbarrier_t *barrier,
                                                        uint32_t expected_tx_count) {
    __mbarrier_token_t token;

    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(token)
                 : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                   "r"(expected_tx_count)
                 : "memory");
    return token;
}

inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token) {
    int ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(ready)
        : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "l"(token)
        : "memory");
    return ready;
}

/*
 * PTX helper: fence.proxy.async.shared::cta.  Must be issued after threads write
 * to smem and before TMA reads that smem, so the async proxy engine sees the
 * writes.  Paired with cp.async.bulk.wait_group.read to establish source-buffer
 * reuse ordering after an outbound TMA put.
 */
__device__ __forceinline__ void nvshmemi_tma_fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

/*
 * mbarrier helpers for pairing with cp.async.bulk gmem->smem (inbound TMA).
 * Layout: mbarrier is an 8-byte object in shared memory, 8-byte aligned.
 * Usage pattern for single-thread pipelined TMA load:
 *   mbarrier_init(bar);                       // arrive count = 1
 *   fence.proxy.async.shared::cta             // make init visible
 *   for each chunk (phase toggles 0/1):
 *     mbarrier_arrive_expect_tx(bar, bytes);  // arrive + expect N bytes
 *     cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes ...
 *     mbarrier_try_wait(bar, phase);          // spin until barrier releases
 *
 * All ops are issued from a single thread (the caller); mbarrier ops are
 * thread-safe when the calling thread is the only arriver.
 */
__device__ __forceinline__ void nvshmemi_tma_mbarrier_init(uint64_t *mbar) {
    unsigned int addr = nvshmemi_tma_cvta_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
}

__device__ __forceinline__ void nvshmemi_tma_mbarrier_arrive_expect_tx(uint64_t *mbar,
                                                                       uint32_t bytes) {
    unsigned int addr = nvshmemi_tma_cvta_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void nvshmemi_tma_mbarrier_try_wait(uint64_t *mbar, int phase) {
    unsigned int addr = nvshmemi_tma_cvta_to_shared(mbar);
    asm volatile(R"({
.reg .pred p;
waitL_%=:
mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;
@!p bra waitL_%=;
})" ::"r"(addr),
                 "r"(phase)
                 : "memory");
}

/*
 * mbarrier.complete_tx: manually add `tx_count` to a barrier's tx counter.
 * Paired with mbarrier.arrive.expect_tx to release a barrier that is not
 * being fulfilled by a cp.async.bulk completion.
 */
__device__ __forceinline__ void nvshmemi_tma_mbarrier_complete_tx(uint64_t *mbar,
                                                                  uint32_t tx_count) {
    unsigned int addr = nvshmemi_tma_cvta_to_shared(mbar);
    asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(addr),
                 "r"(tx_count)
                 : "memory");
}

/*
 * PTX helper: issue cp.async.bulk gmem -> smem with mbarrier completion tracking.
 * The mbarrier's tx_count is incremented by `bytes` on completion.
 */
__device__ __forceinline__ void nvshmemi_tma_bulk_global_to_shared(void *smem_dst,
                                                                   const void *gmem_src,
                                                                   uint32_t bytes, uint64_t *mbar) {
    unsigned int dst_addr = nvshmemi_tma_cvta_to_shared(smem_dst);
    unsigned int mbar_addr = nvshmemi_tma_cvta_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(dst_addr),
        "l"((uint64_t)(uintptr_t)gmem_src), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

/*
 * nvshmemi_memcpy_tma_shared_global - Copy data from local shared memory to
 * local or remote global memory via TMA bulk async copy.
 *
 * Template parameters:
 *   SCOPE   - Threadgroup scope for the operation:
 *     THREAD - Single calling thread issues the entire transfer.
 *     WARP   - Elected leader of the calling warp issues the transfer; warp syncs.
 *     BLOCK  - Elected leader of warp 0 issues the full transfer as a single
 *              cp.async.bulk op; all threads sync via __syncthreads().
 *              A single large op amortises per-op TMA latency better than
 *              splitting the buffer across warps.
 *   BLOCKING - When true, waits for full completion (smem read + global write)
 *              before returning.  When false (NBI), returns immediately after
 *              issuing the transfer.
 *
 * Requirements:
 *   - SM >= 90 (Hopper or newer)
 *   - gmem_dst must be 16-byte aligned
 *   - smem_src must be 16-byte aligned
 *   - bytes must be a multiple of 16 and > 0
 *   - The caller must issue fence.proxy.async.shared::cta before calling this
 *     function to make any prior shared-memory stores visible to the TMA async
 *     proxy engine.  Without this fence, the TMA engine may read stale data.
 *     Note: a single fence before a sequence of puts is sufficient if smem is
 *     not modified between puts (e.g. one fence per collective).
 *
 * For the non-blocking variant (BLOCKING=false):
 *   - To reuse the smem buffer for the next chunk before all remote writes
 *     complete: call cp.async.bulk.wait_group.read 0 followed by
 *     __syncthreads().  This waits only for the smem READ phase to finish,
 *     leaving the remote write in flight.
 *   - For full completion (smem read + remote write both done): call
 *     nvshmem_quiet() from a thread then __syncthreads().
 *
 * Returns 0 on success.
 */
template <threadgroup_t SCOPE, bool BLOCKING>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
    if (bytes == 0) return 0;
    /* Validate cp.async.bulk requirements.  Fall back to 0/error rather than
     * invoking undefined hardware behavior.
     * TODO: handle head/tail of unaligned messages with ld/st so that callers
     * with arbitrary alignment and size can still use the TMA fast path for the
     * aligned middle portion. */
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)gmem_dst)) return -1;
    if (!nvshmemi_tma_is_16b_aligned((size_t)(uintptr_t)smem_src)) return -1;
    if (!nvshmemi_tma_is_16b_aligned(bytes)) return -1;
    if (bytes > (size_t)UINT32_MAX) return -1;

    const unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
        nvshmemi_tma_bulk_commit_group();
        if (BLOCKING) {
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
    } else if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        if (nvshmemi_tma_elect_warp()) {
            nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
            nvshmemi_tma_bulk_commit_group();
            if (BLOCKING) {
                nvshmemi_tma_bulk_wait_group_0();
                __threadfence_system();
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    } else if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        /* One elected thread across the entire block issues the full transfer
         * as a single cp.async.bulk op.  nvshmemi_tma_block_is_elected() uses
         * elect.sync + __shfl_sync so the compiler sees a warp-uniform predicate
         * and does not insert a serialising peeling loop. */
        if (nvshmemi_tma_block_is_elected()) {
            nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
            nvshmemi_tma_bulk_commit_group();
            if (BLOCKING) {
                nvshmemi_tma_bulk_wait_group_0();
                __threadfence_system();
            }
        }
        __syncthreads();
    }

    return 0;
}

/* Convenience aliases matching the original two-function API. */
template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
    return nvshmemi_memcpy_tma_shared_global<SCOPE, true>(gmem_dst, smem_src, bytes);
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global_nbi(void *gmem_dst, const void *smem_src,
                                                            size_t bytes) {
    return nvshmemi_memcpy_tma_shared_global<SCOPE, false>(gmem_dst, smem_src, bytes);
}

#else

/*
 * Provide compile-time fallbacks for non-Hopper targets so call sites in the
 * generic device path can be instantiated for the full arch matrix. Runtime
 * gating in nvshmemi_tma_smem_registered() keeps these paths unreachable.
 */
__device__ __forceinline__ void nvshmemi_tma_bulk_commit_group() {}

__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_read_0() {}

__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_0() {}

__device__ __forceinline__ void nvshmemi_tma_fence_proxy_async_shared_cta() {}

template <threadgroup_t SCOPE, bool BLOCKING>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void * /* gmem_dst */,
                                                        const void * /* smem_src */,
                                                        size_t /* bytes */) {
    return -1;
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
    return nvshmemi_memcpy_tma_shared_global<SCOPE, true>(gmem_dst, smem_src, bytes);
}

template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global_nbi(void *gmem_dst, const void *smem_src,
                                                            size_t bytes) {
    return nvshmemi_memcpy_tma_shared_global<SCOPE, false>(gmem_dst, smem_src, bytes);
}

#endif /* __CUDA_ARCH__ >= 900 */
#endif /* __CUDA_ARCH__ */
#endif /* TMA_DEVICE_CUH */
