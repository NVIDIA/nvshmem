/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_BATCH_RMA_PENDING_QPS_CUH_
#define _NVSHMEMI_BATCH_RMA_PENDING_QPS_CUH_

#include <stdint.h>

#include "device_host_transport/nvshmem_common_batch_rma_pending_qps.hpp"

#ifdef __CUDA_ARCH__

/*
 * Tracks, per active region slot, transport QPs whose ready work was not submitted because the
 * region carries the batch-RMA hint. Region thresholds and region end drain this set and ask the
 * transport to submit all currently ready work on each marked QP. The two-level bitmap avoids
 * scanning every transport QP.
 */
__device__ __forceinline__ uint32_t *nvshmemi_batch_rma_pending_bitmap(void *slot_storage) {
    return static_cast<uint32_t *>(slot_storage);
}

__device__ __forceinline__ void *nvshmemi_batch_rma_pending_qps_for_slot(
    void *all_slots_storage, uint32_t qp_count, uint32_t region_slot_index) {
    return nvshmemi_batch_rma_pending_bitmap(all_slots_storage) +
           region_slot_index * nvshmemi_batch_rma_slot_chunk_count(qp_count);
}

__device__ __forceinline__ uint32_t *nvshmemi_batch_rma_summary_bitmap(void *slot_storage,
                                                                       uint32_t qp_count) {
    return nvshmemi_batch_rma_pending_bitmap(slot_storage) +
           nvshmemi_batch_rma_pending_chunk_count(qp_count);
}

__device__ __forceinline__ void nvshmemi_batch_rma_mark_qp_pending(void *slot_storage,
                                                                   uint32_t qp_count,
                                                                   uint32_t qp_index) {
    if (slot_storage == nullptr || qp_index >= qp_count) {
        return;
    }

    uint32_t pending_chunk_index = qp_index / nvshmemi_batch_rma_bits_per_chunk;
    uint32_t qp_bit = 1U << (qp_index % nvshmemi_batch_rma_bits_per_chunk);
    uint32_t summary_chunk_index = pending_chunk_index / nvshmemi_batch_rma_bits_per_chunk;
    uint32_t pending_chunk_bit = 1U << (pending_chunk_index % nvshmemi_batch_rma_bits_per_chunk);

    /* Pending-before-summary lets a concurrent submission either consume this mark or leave the
     * summary bit set for the next submission pass. */
    atomicOr(nvshmemi_batch_rma_pending_bitmap(slot_storage) + pending_chunk_index, qp_bit);
    atomicOr(nvshmemi_batch_rma_summary_bitmap(slot_storage, qp_count) + summary_chunk_index,
             pending_chunk_bit);
}

__device__ __forceinline__ uint32_t nvshmemi_batch_rma_take_summary_chunk(
    void *slot_storage, uint32_t qp_count, uint32_t summary_chunk_index) {
    return atomicExch(
        nvshmemi_batch_rma_summary_bitmap(slot_storage, qp_count) + summary_chunk_index, 0U);
}

__device__ __forceinline__ uint32_t
nvshmemi_batch_rma_take_pending_chunk(void *slot_storage, uint32_t pending_chunk_index) {
    return atomicExch(nvshmemi_batch_rma_pending_bitmap(slot_storage) + pending_chunk_index, 0U);
}

template <typename SubmitQp>
__device__ __forceinline__ void nvshmemi_batch_rma_submit_pending_qps(void *slot_storage,
                                                                      uint32_t qp_count,
                                                                      SubmitQp submit_qp) {
    uint32_t summary_chunk_count = nvshmemi_batch_rma_summary_chunk_count(qp_count);
    for (uint32_t summary_chunk_index = 0; summary_chunk_index < summary_chunk_count;
         summary_chunk_index++) {
        uint32_t summary_chunk =
            nvshmemi_batch_rma_take_summary_chunk(slot_storage, qp_count, summary_chunk_index);
        while (summary_chunk != 0) {
            uint32_t pending_chunk_bit = static_cast<uint32_t>(__ffs(summary_chunk) - 1);
            uint32_t pending_chunk_index =
                summary_chunk_index * nvshmemi_batch_rma_bits_per_chunk + pending_chunk_bit;
            uint32_t pending_chunk =
                nvshmemi_batch_rma_take_pending_chunk(slot_storage, pending_chunk_index);
            while (pending_chunk != 0) {
                uint32_t qp_bit = static_cast<uint32_t>(__ffs(pending_chunk) - 1);
                uint32_t qp_index =
                    pending_chunk_index * nvshmemi_batch_rma_bits_per_chunk + qp_bit;
                if (qp_index < qp_count) {
                    submit_qp(qp_index);
                }
                pending_chunk &= pending_chunk - 1;
            }
            summary_chunk &= summary_chunk - 1;
        }
    }
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_BATCH_RMA_PENDING_QPS_CUH_ */
