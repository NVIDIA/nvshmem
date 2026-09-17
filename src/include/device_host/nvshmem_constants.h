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

#include "device_host/nvshmem_version.h"

/* This is not the NVSHMEM release version, it is the supported OpenSHMEM spec version. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3

#define NVSHMEM_VENDOR_VERSION                                                       \
    ((NVSHMEM_VENDOR_MAJOR_VERSION) * 10000 + (NVSHMEM_VENDOR_MINOR_VERSION) * 100 + \
     (NVSHMEM_VENDOR_PATCH_VERSION))

#define NVSHMEM_MAX_NAME_LEN 256

typedef enum nvshmemx_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE,
    NVSHMEM_CMP_SENTINEL = INT_MAX,
} nvshmemx_cmp_type_t;

typedef enum nvshmemx_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE,
    NVSHMEM_THREAD_TYPE_SENTINEL = INT_MAX,
} nvshmemx_thread_support_t;

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

/* In the proxy code, PE is represented as a 16-bit integer, so we use the 16th bit to represent
 * any and all. */
typedef enum {
    NVSHMEM_PE_INVALID = -1,
    NVSHMEMX_PE_ANY = (1 << 15),
    NVSHMEMX_PE_ALL = (1 << 15),
} nvshmem_pe_index_t;

#endif
