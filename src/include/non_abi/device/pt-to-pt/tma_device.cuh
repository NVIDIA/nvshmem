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
#if __CUDA_ARCH__ >= 900

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
    asm volatile(
        "{\n\t"
        ".reg .pred elect_p;\n\t"
        ".reg .u32  elect_id;\n\t"
        "elect.sync elect_id|elect_p, 0xffffffff;\n\t"
        "selp.u32 %0, 1, 0, elect_p;\n\t"
        "}\n\t"
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
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x +
                       threadIdx.z * blockDim.x * blockDim.y;
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
    asm("{ .reg .u64 smem_u64;"
        "  cvta.to.shared.u64 smem_u64, %1;"
        "  cvt.u32.u64 %0, smem_u64; }"
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
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :
                 : "l"((uint64_t)(uintptr_t)gmem_dst), "r"(smem_addr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void nvshmemi_tma_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_read_0() {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

/* Full completion wait: smem read AND global write both done. */
__device__ __forceinline__ void nvshmemi_tma_bulk_wait_group_0() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
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
 *     nvshmem_quiet() from every thread then __syncthreads().
 *
 * Returns 0 on success.
 */
template <threadgroup_t SCOPE, bool BLOCKING>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
    if (bytes == 0) return 0;

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
        nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
        nvshmemi_tma_bulk_commit_group();
        if (BLOCKING) {
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
    } else if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        if (nvshmemi_tma_elect_warp()) {
            unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
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
            unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
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

#endif /* __CUDA_ARCH__ >= 900 */
#endif /* __CUDA_ARCH__ */
#endif /* TMA_DEVICE_CUH */
