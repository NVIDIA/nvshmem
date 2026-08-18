/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_TYPES_H_
#define _NVSHMEMI_REGION_TYPES_H_

#if !defined __CUDACC_RTC__
#include <limits.h>
#include <stdint.h>
#else
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#endif

#include "device_host/nvshmem_types.h"
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/nvshmemi_region_constants.h"

#define NVSHMEMI_REGION_SUPPORTED_HINTS ((uint32_t)NVSHMEMI_REGION_HINT_BATCH_RMA)
static_assert(NVSHMEMI_REGION_HINT_BATCH_RMA == NVSHMEMX_REGION_HINT_BATCH_RMA,
              "Public and internal batch RMA hint values must match.");

typedef enum {
    NVSHMEMI_REGION_OPERATION_NONE = 0,
    NVSHMEMI_REGION_OPERATION_NBI_RMA,
    NVSHMEMI_REGION_OPERATION_SENTINEL = INT_MAX,
} nvshmemi_region_operation_t;

template <nvshmemi_region_operation_t OPERATION>
struct nvshmemi_region_operation_traits {
    static constexpr uint32_t supported_hints = NVSHMEMX_REGION_HINT_NONE;
};

template <>
struct nvshmemi_region_operation_traits<NVSHMEMI_REGION_OPERATION_NBI_RMA> {
    static constexpr uint32_t supported_hints = NVSHMEMI_REGION_HINT_BATCH_RMA;
};

typedef struct {
    uint64_t issuer_id;
    uint64_t region_id;
    uint32_t hints;
} nvshmemi_region_info_t;

typedef struct {
    uint32_t operation_ticket;
} nvshmemi_region_batch_rma_state_t;

typedef enum {
    NVSHMEMI_REGION_SLOT_FREE = 0,
    NVSHMEMI_REGION_SLOT_INITIALIZING = 1,
    NVSHMEMI_REGION_SLOT_ACTIVE = 2,
    NVSHMEMI_REGION_SLOT_STATE_SENTINEL = INT_MAX,
} nvshmemi_region_slot_state_t;
static_assert(NVSHMEMI_REGION_SLOT_FREE == 0,
              "CUDA zero-initialization must produce free region slots.");

/* Keep lookup-hot fields in the first 32 bytes and isolate each slot in a 64-byte extent. */
struct alignas(NVSHMEMI_REGION_SLOT_BYTES) nvshmemi_region_slot {
    /* Active states encode the slot generation above the low phase bits. */
    unsigned long long state;
    uint64_t gridid;
    uint64_t block_id;
    uint32_t hints;
    nvshmemi_region_batch_rma_state_t batch_rma;
    uint64_t generation;
    uint64_t issuer_id;
};
static_assert(sizeof(nvshmemi_region_slot_t) == NVSHMEMI_REGION_SLOT_BYTES,
              "Region slot must have the configured extent.");
static_assert(alignof(nvshmemi_region_slot_t) == NVSHMEMI_REGION_SLOT_BYTES,
              "Region slot alignment must match its configured extent.");

#define PROXY_REGION_METADATA_ENTRIES                                        \
    ((sizeof(nvshmemi_region_info_t) + PROXY_CHANNEL_ENTRY_DATA_BYTES - 1) / \
     PROXY_CHANNEL_ENTRY_DATA_BYTES)
#define PROXY_REGION_METADATA_BYTES (PROXY_REGION_METADATA_ENTRIES * CHANNEL_ENTRY_BYTES)
#define PROXY_REGION_END_REQ_BYTES (CHANNEL_ENTRY_BYTES + PROXY_REGION_METADATA_BYTES)

static_assert(PROXY_REGION_METADATA_ENTRIES * PROXY_CHANNEL_ENTRY_DATA_BYTES >=
                  sizeof(nvshmemi_region_info_t),
              "Proxy region metadata must hold nvshmemi_region_info_t.");
static_assert((PROXY_REGION_METADATA_ENTRIES - 1) * PROXY_CHANNEL_ENTRY_DATA_BYTES <
                  sizeof(nvshmemi_region_info_t),
              "Proxy region metadata must use the minimum number of channel entries.");
static_assert((PROXY_GROUP_COUNT_MAX & PROXY_GROUP_REGION) == 0,
              "The region marker must not overlap valid proxy group counts.");
static_assert(PROXY_GROUP_SIZE_SINGLE <= PROXY_GROUP_COUNT_MAX,
              "The single-request proxy group size must fit in the count field.");

#endif /* _NVSHMEMI_REGION_TYPES_H_ */
