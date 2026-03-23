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

#endif /* __CUDA_ARCH__ >= 900 */

/*
 * nvshmemi_memcpy_tma_shared_global - Copy data from local shared memory to
 * local or remote global memory via TMA bulk async copy.
 *
 * Templated on threadgroup scope:
 *   THREAD - Single calling thread issues the entire transfer.
 *   WARP   - Lane 0 of the calling warp issues the transfer; warp syncs.
 *   BLOCK  - Transfer is striped across warps; each warp's lane 0 issues
 *            a chunk. Achieves higher throughput for large transfers.
 *
 * This is the blocking variant: waits for completion before returning.
 *
 * Requirements:
 *   - SM >= 90 (Hopper or newer)
 *   - gmem_dst must be 16-byte aligned
 *   - smem_src must be 16-byte aligned
 *   - bytes must be a multiple of 16 and > 0
 *   - The caller must issue fence.proxy.async.shared::cta before calling this
 *     function to make any prior shared-memory stores visible to the TMA async
 *     proxy engine.  Without this fence, the TMA engine may read stale data.
 *
 * Returns 0 on success, -1 if TMA is not available for this architecture.
 */
template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
#if __CUDA_ARCH__ >= 900
    if (bytes == 0) return 0;

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
        nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
        nvshmemi_tma_bulk_commit_group();
        nvshmemi_tma_bulk_wait_group_0();
        __threadfence_system();
    } else if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        if (nvshmemi_tma_elect_warp()) {
            unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
            nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
            nvshmemi_tma_bulk_commit_group();
            nvshmemi_tma_bulk_wait_group_0();
            __threadfence_system();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    } else if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int block_size = blockDim.x * blockDim.y * blockDim.z;
        int num_warps = (block_size + warpSize - 1) / warpSize;
        int warp_id = tid / warpSize;

        /* Divide transfer across warps, each chunk 16-byte aligned */
        size_t base_chunk = (bytes / num_warps) & ~(size_t)15;

        if (base_chunk >= 16) {
            /* Multi-warp path: elected leader of each warp handles its chunk */
            size_t offset = (size_t)warp_id * base_chunk;
            size_t this_chunk =
                (warp_id < num_warps - 1) ? base_chunk : (bytes - offset);

            if (nvshmemi_tma_elect_warp() && offset < bytes) {
                unsigned int smem_addr =
                    nvshmemi_tma_cvta_to_shared((const char *)smem_src + offset);
                nvshmemi_tma_bulk_shared_to_global((char *)gmem_dst + offset, smem_addr,
                                                   (uint32_t)this_chunk);
                nvshmemi_tma_bulk_commit_group();
                nvshmemi_tma_bulk_wait_group_0();
                __threadfence_system();
            }
        } else {
            /* Small transfer: elected leader of warp 0 handles everything */
            if (tid < warpSize && nvshmemi_tma_elect_warp()) {
                unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
                nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
                nvshmemi_tma_bulk_commit_group();
                nvshmemi_tma_bulk_wait_group_0();
                __threadfence_system();
            }
        }
        __syncthreads();
    }

    return 0;
#else
    (void)gmem_dst;
    (void)smem_src;
    (void)bytes;
    return -1;
#endif /* __CUDA_ARCH__ >= 900 */
}

/*
 * nvshmemi_memcpy_tma_shared_global_nbi - Non-blocking variant.
 *
 * Same scoping as nvshmemi_memcpy_tma_shared_global but does not wait for
 * the transfer to complete.
 *
 * Caller responsibilities:
 *   - fence.proxy.async.shared::cta must be issued before each call to make
 *     smem stores visible to the TMA async proxy (same as the blocking variant).
 *   - To reuse the smem buffer for the next chunk before all remote writes
 *     complete: call cp.async.bulk.wait_group.read 0 (warp-level; one thread
 *     per warp suffices) followed by __syncthreads().  This waits only for the
 *     smem READ phase to finish, leaving the remote write in flight.
 *   - For full completion (smem read + remote write both done): call
 *     nvshmemi_quiet<SCOPE>() or nvshmem_quiet() from the thread that issued
 *     the transfer.  nvshmem_quiet() is THREAD scope — for a multi-warp CTA
 *     using BLOCK-scope NBI, use nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>()
 *     so that every warp's in-flight groups are drained.
 *
 * Returns 0 on success, -1 if TMA is not available.
 */
template <threadgroup_t SCOPE>
__device__ inline int nvshmemi_memcpy_tma_shared_global_nbi(void *gmem_dst, const void *smem_src,
                                                            size_t bytes) {
#if __CUDA_ARCH__ >= 900
    if (bytes == 0) return 0;

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
        nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
        nvshmemi_tma_bulk_commit_group();
    } else if (SCOPE == NVSHMEMI_THREADGROUP_WARP) {
        if (nvshmemi_tma_elect_warp()) {
            unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
            nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
            nvshmemi_tma_bulk_commit_group();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    } else if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int block_size = blockDim.x * blockDim.y * blockDim.z;
        int num_warps = (block_size + warpSize - 1) / warpSize;
        int warp_id = tid / warpSize;

        size_t base_chunk = (bytes / num_warps) & ~(size_t)15;

        if (base_chunk >= 16) {
            size_t offset = (size_t)warp_id * base_chunk;
            size_t this_chunk =
                (warp_id < num_warps - 1) ? base_chunk : (bytes - offset);

            if (nvshmemi_tma_elect_warp() && offset < bytes) {
                unsigned int smem_addr =
                    nvshmemi_tma_cvta_to_shared((const char *)smem_src + offset);
                nvshmemi_tma_bulk_shared_to_global((char *)gmem_dst + offset, smem_addr,
                                                   (uint32_t)this_chunk);
                nvshmemi_tma_bulk_commit_group();
            }
        } else {
            if (tid < warpSize && nvshmemi_tma_elect_warp()) {
                unsigned int smem_addr = nvshmemi_tma_cvta_to_shared(smem_src);
                nvshmemi_tma_bulk_shared_to_global(gmem_dst, smem_addr, (uint32_t)bytes);
                nvshmemi_tma_bulk_commit_group();
            }
        }
        __syncthreads();
    }

    return 0;
#else
    (void)gmem_dst;
    (void)smem_src;
    (void)bytes;
    return -1;
#endif
}

#endif /* __CUDA_ARCH__ */
#endif /* TMA_DEVICE_CUH */
