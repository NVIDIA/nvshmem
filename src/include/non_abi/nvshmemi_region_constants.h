/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_CONSTANTS_H_
#define _NVSHMEMI_REGION_CONSTANTS_H_

#define NVSHMEMI_REGION_MAX_SLOTS_DEFAULT 4096
/* Tolerate short collision clusters while bounding source-path lookup work. */
#define NVSHMEMI_REGION_SLOT_PROBE_LIMIT 8
#define NVSHMEMI_REGION_HINT_BATCH_RMA (1u << 0)

#endif
