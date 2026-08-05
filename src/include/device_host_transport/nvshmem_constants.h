/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEM_CONSTANTS_H_
#define _NVSHMEM_CONSTANTS_H_

#if !defined __CUDACC_RTC__
#include <limits.h>
#include <stdint.h>
#else
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#endif
#include "non_abi/nvshmem_version.h"

#define CHANNEL_BUF_SIZE (1 << CHANNEL_BUF_SIZE_LOG)
#define CHANNEL_BUF_SIZE_LOG 22
#define CHANNEL_ENTRY_BYTES 8

/* This is not the NVSHMEM release version, it is the supported OpenSHMEM spec version. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3

#define NVSHMEM_VENDOR_VERSION                                                       \
    ((NVSHMEM_VENDOR_MAJOR_VERSION) * 10000 + (NVSHMEM_VENDOR_MINOR_VERSION) * 100 + \
     (NVSHMEM_VENDOR_PATCH_VERSION))

#define NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S) #S
#define NVSHMEMI_SUBST_AND_STRINGIFY(S) NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S)

#define NVSHMEM_VENDOR_STRING \
    "NVSHMEM v"                                       \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MAJOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MINOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_PATCH_VERSION)

#define NVSHMEM_MAX_NAME_LEN 256

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

typedef enum nvshmemi_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE,
    NVSHMEM_CMP_SENTINEL = INT_MAX,
} nvshmemx_cmp_type_t;

typedef enum nvshmemi_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE,
    NVSHMEM_THREAD_TYPE_SENTINEL = INT_MAX,
} nvshmemx_thread_support_t;

typedef enum {
    PROXY_GLOBAL_EXIT_NOT_REQUESTED = 0,
    PROXY_GLOBAL_EXIT_INIT,
    PROXY_GLOBAL_EXIT_REQUESTED,
    PROXY_GLOBAL_EXIT_FINISHED,
    PROXY_GLOBAL_EXIT_MAX_STATE = INT_MAX
} nvshmemx_proxy_status_t;

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

typedef enum {
    NVSHMEM_STATUS_NOT_INITIALIZED = 0,
    NVSHMEM_STATUS_IS_BOOTSTRAPPED,
    NVSHMEM_STATUS_IS_INITIALIZED,
    NVSHMEM_STATUS_LIMITED_MPG,
    NVSHMEM_STATUS_FULL_MPG,
    NVSHMEM_STATUS_INVALID = INT_MAX,
} nvshmemx_init_status_t;

typedef enum {
    NVSHMEMX_QP_HOST = 0,
    NVSHMEMX_QP_DEFAULT = 1,
    NVSHMEMX_QP_ANY = INT_MAX,
    NVSHMEMX_QP_ALL = INT_MAX,
} nvshmemx_qp_handle_index_t;

/* in the proxy code, pe is represented as a 16-bit integer, so we use the 16th bit to represent any
 * and all */
typedef enum {
    NVSHMEM_PE_INVALID = -1,
    NVSHMEMX_PE_ANY = (1 << 15),
    NVSHMEMX_PE_ALL = (1 << 15),
} nvshmem_pe_index_t;

#endif
