/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_CONSTANTS_H_
#define _NVSHMEMI_CONSTANTS_H_

#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

#include "device_host/nvshmem_constants.h"

#define CHANNEL_BUF_SIZE (1 << CHANNEL_BUF_SIZE_LOG)
#define CHANNEL_BUF_SIZE_LOG 22
#define CHANNEL_ENTRY_BYTES 8

/*
 * SplitMix64 finalizer parameters used to hash AMO targets. Keep these shared
 * between host and device routing so a target selects the same endpoint.
 * Reference: https://prng.di.unimi.it/splitmix64.c
 */
#define NVSHMEMI_SPLITMIX64_SHIFT_1 30U
#define NVSHMEMI_SPLITMIX64_MULTIPLIER_1 0xbf58476d1ce4e5b9ULL
#define NVSHMEMI_SPLITMIX64_SHIFT_2 27U
#define NVSHMEMI_SPLITMIX64_MULTIPLIER_2 0x94d049bb133111ebULL
#define NVSHMEMI_SPLITMIX64_SHIFT_3 31U

typedef enum {
    PROXY_GLOBAL_EXIT_NOT_REQUESTED = 0,
    PROXY_GLOBAL_EXIT_INIT,
    PROXY_GLOBAL_EXIT_REQUESTED,
    PROXY_GLOBAL_EXIT_FINISHED,
    PROXY_GLOBAL_EXIT_MAX_STATE = INT_MAX
} nvshmemi_proxy_global_exit_status_t;

#define PROXY_DMA_REQ_BYTES 32
#define PROXY_CHANNEL_ENTRY_CONTROL_BYTES 1
#define PROXY_CHANNEL_ENTRY_DATA_BYTES (CHANNEL_ENTRY_BYTES - PROXY_CHANNEL_ENTRY_CONTROL_BYTES)
#define PROXY_GROUP_SIZE_SINGLE 0x01
/* groupsize uses the low seven bits for its count and reserves the high bit for region metadata. */
#define PROXY_GROUP_COUNT_MAX 0x7f
#define PROXY_GROUP_REGION 0x80
#define PROXY_AMO_REQ_BYTES 40
#define PROXY_INLINE_REQ_BYTES 24
#define PROXY_PUT_WITH_SIG_REQ_BYTES 48

#endif
