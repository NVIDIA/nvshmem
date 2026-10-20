/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_COMMON_BATCH_RMA_PENDING_QPS_HPP_
#define _NVSHMEMI_COMMON_BATCH_RMA_PENDING_QPS_HPP_

#include <stddef.h>
#include <stdint.h>

/*
 * Host/device-neutral storage layout and allocation sizing. CUDA-only bitmap mutation and
 * submission helpers live in non_abi/device/common/nvshmemi_batch_rma_pending_qps.cuh.
 */

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define NVSHMEMI_BATCH_RMA_HOST_DEVICE __host__ __device__
#else
#define NVSHMEMI_BATCH_RMA_HOST_DEVICE
#endif

static constexpr uint32_t nvshmemi_batch_rma_bits_per_chunk = 32U;

NVSHMEMI_BATCH_RMA_HOST_DEVICE static constexpr uint32_t nvshmemi_batch_rma_pending_chunk_count(
    uint32_t qp_count) {
    return (qp_count + nvshmemi_batch_rma_bits_per_chunk - 1) / nvshmemi_batch_rma_bits_per_chunk;
}

NVSHMEMI_BATCH_RMA_HOST_DEVICE static constexpr uint32_t nvshmemi_batch_rma_summary_chunk_count(
    uint32_t qp_count) {
    return nvshmemi_batch_rma_pending_chunk_count(nvshmemi_batch_rma_pending_chunk_count(qp_count));
}

NVSHMEMI_BATCH_RMA_HOST_DEVICE static constexpr uint32_t nvshmemi_batch_rma_slot_chunk_count(
    uint32_t qp_count) {
    return nvshmemi_batch_rma_pending_chunk_count(qp_count) +
           nvshmemi_batch_rma_summary_chunk_count(qp_count);
}

static inline bool nvshmemi_batch_rma_pending_qps_storage_size(uint32_t qp_count,
                                                               int region_slot_count,
                                                               size_t *size) {
    if (size == nullptr || region_slot_count <= 0 ||
        (region_slot_count & (region_slot_count - 1)) != 0) {
        return false;
    }

    size_t slot_chunk_count = nvshmemi_batch_rma_slot_chunk_count(qp_count);
    if (slot_chunk_count > SIZE_MAX / static_cast<size_t>(region_slot_count) / sizeof(uint32_t)) {
        return false;
    }

    *size = slot_chunk_count * static_cast<size_t>(region_slot_count) * sizeof(uint32_t);
    return true;
}

#undef NVSHMEMI_BATCH_RMA_HOST_DEVICE

#endif /* _NVSHMEMI_COMMON_BATCH_RMA_PENDING_QPS_HPP_ */
