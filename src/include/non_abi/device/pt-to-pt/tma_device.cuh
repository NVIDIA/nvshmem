/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef TMA_DEVICE_CUH
#define TMA_DEVICE_CUH

#include <cuda_runtime.h>
#include "device_host/nvshmem_types.h"

#ifdef __CUDA_ARCH__

/*
 * nvshmemi_memcpy_tma_shared_global - Copy data from local shared memory to
 * local or remote global memory via TMA bulk async copy.
 *
 * This function uses cp.async.bulk (SM90+) to perform an asynchronous bulk
 * copy from shared memory to global memory. This is the blocking variant:
 * the function waits for the copy to complete before returning.
 *
 * Requirements:
 *   - SM >= 90 (Hopper or newer)
 *   - gmem_dst must be 16-byte aligned
 *   - smem_src must be 16-byte aligned
 *   - bytes must be a multiple of 16 and > 0
 *
 * Parameters:
 *   gmem_dst - Destination in global memory (may be peer-mapped for NVLink)
 *   smem_src - Source in shared memory (must be within the calling CTA's smem)
 *   bytes    - Number of bytes to copy
 *
 * Returns 0 on success, -1 if TMA is not available for this architecture.
 */
__device__ inline int nvshmemi_memcpy_tma_shared_global(void *gmem_dst, const void *smem_src,
                                                        size_t bytes) {
#if __CUDA_ARCH__ >= 900
    if (bytes == 0) return 0;

    /* Convert the generic smem pointer to a shared-memory-space address */
    unsigned int smem_addr;
    asm("{ .reg .u64 smem_u64;"
        "  cvta.to.shared.u64 smem_u64, %1;"
        "  cvt.u32.u64 %0, smem_u64; }"
        : "=r"(smem_addr)
        : "l"((uint64_t)(uintptr_t)smem_src));

    /* Issue bulk async copy from shared to global */
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :
                 : "l"((uint64_t)(uintptr_t)gmem_dst), "r"(smem_addr), "r"((uint32_t)bytes)
                 : "memory");

    /* Commit the bulk group */
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");

    /* Wait for all outstanding bulk groups to complete reading from shared memory */
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");

    return 0;
#else
    /* TMA bulk async copy not available on this architecture */
    return -1;
#endif /* __CUDA_ARCH__ >= 900 */
}

/*
 * nvshmemi_memcpy_tma_shared_global_nbi - Non-blocking variant.
 *
 * Same as nvshmemi_memcpy_tma_shared_global but does not wait for completion.
 * The caller is responsible for issuing cp.async.bulk.wait_group or a
 * barrier to synchronize the transfer.
 *
 * Returns 0 on success, -1 if TMA is not available.
 */
__device__ inline int nvshmemi_memcpy_tma_shared_global_nbi(void *gmem_dst, const void *smem_src,
                                                            size_t bytes) {
#if __CUDA_ARCH__ >= 900
    if (bytes == 0) return 0;

    unsigned int smem_addr;
    asm("{ .reg .u64 smem_u64;"
        "  cvta.to.shared.u64 smem_u64, %1;"
        "  cvt.u32.u64 %0, smem_u64; }"
        : "=r"(smem_addr)
        : "l"((uint64_t)(uintptr_t)smem_src));

    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :
                 : "l"((uint64_t)(uintptr_t)gmem_dst), "r"(smem_addr), "r"((uint32_t)bytes)
                 : "memory");

    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");

    return 0;
#else
    return -1;
#endif
}

#endif /* __CUDA_ARCH__ */
#endif /* TMA_DEVICE_CUH */
