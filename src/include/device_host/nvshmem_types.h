/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

#if !defined __CUDACC_RTC__
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#else
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#endif

#include "bootstrap_device_host/nvshmem_uniqueid.h"

#if defined(__cplusplus)
#define NVSHMEMI_TYPES_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define NVSHMEMI_TYPES_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define NVSHMEMX_INIT_ARGS_V2_RESERVED_BYTES 92
#define NVSHMEMX_INIT_ARGS_V1_RESERVED_BYTES 96
#define NVSHMEM_TEAM_CONFIG_V2_RESERVED_BYTES 48
#define NVSHMEM_TEAM_CONFIG_V1_RESERVED_BYTES 56
#define INIT_ARGS_SCALAR_INVALID -1
#define TEAM_CONFIG_SCALAR_INVALID -1
#define TEAM_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFFULL

typedef enum {
    NVSHMEM_SIGNAL_SET = 9,
    NVSHMEM_SIGNAL_ADD = 10,
} nvshmemx_signal_op_t;

typedef int nvshmemx_qp_handle_t;
typedef uint64_t nvshmemx_team_uniqueid_t;
typedef int32_t nvshmem_team_t;
typedef nvshmem_team_t nvshmemx_team_t;

typedef enum {
    NVSHMEM_TEAM_INVALID = -1,
    NVSHMEM_TEAM_WORLD = 0,
    NVSHMEM_TEAM_WORLD_INDEX = 0,
    NVSHMEM_TEAM_SHARED = 1,
    NVSHMEM_TEAM_SHARED_INDEX = 1,
    NVSHMEMX_TEAM_NODE = 2,
    NVSHMEM_TEAM_NODE_INDEX = 2,
    NVSHMEMX_TEAM_SAME_MYPE_NODE = 3,
    NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3,
    NVSHMEM_TEAM_SAME_GPU_INDEX = 4,
    NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5,
    NVSHMEM_TEAM_MC_SHARED = 6,
    NVSHMEM_TEAM_MC_SHARED_INDEX = 6,
    NVSHMEM_TEAMS_MIN = 7,
    NVSHMEM_TEAM_INDEX_MAX = INT_MAX
} nvshmem_team_id_t;

typedef struct {
    int major;
    int minor;
    int patch;
} nvshmemi_version_t;

typedef enum nvshmemx_smem_amount_t {
    NVSHMEMX_SMEM_RECOMMENDED = 0,
    NVSHMEMX_SMEM_MINIMUM = 1,
    NVSHMEMX_SMEM_BARRIERS_ONLY = 2,
    NVSHMEMX_SMEM_AMOUNT_MAX = INT_MAX
} nvshmemx_smem_amount_t;

typedef uint64_t nvshmemx_region_handle_t;

typedef enum {
    NVSHMEMX_REGION_HINT_NONE = 0,
    NVSHMEMX_REGION_HINT_BATCH_RMA = 1u << 0,
} nvshmemx_region_hint_t;

#define NVSHMEMX_REGION_ATTRS_RESERVED_BYTES 60
#define NVSHMEMX_REGION_ATTRS_INITIALIZER {NVSHMEMX_REGION_HINT_NONE, {0}}

typedef struct nvshmemx_region_attrs {
    uint32_t hints;
    char reserved[NVSHMEMX_REGION_ATTRS_RESERVED_BYTES];
} nvshmemx_region_attrs_t;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmemx_region_attrs_t) == 64,
                             "Region attributes must be 64 bytes.");

typedef struct {
    int version;
    nvshmemx_uniqueid_args_t uid_args;
    int cuda_device_id;
    char content[NVSHMEMX_INIT_ARGS_V2_RESERVED_BYTES];
} nvshmemx_init_args_v2;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmemx_init_args_v2) == 128,
                             "init_args_v2 must be 128 bytes.");

typedef struct {
    int version;
    nvshmemx_uniqueid_args_t uid_args;
    char content[NVSHMEMX_INIT_ARGS_V1_RESERVED_BYTES];
} nvshmemx_init_args_v1;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmemx_init_args_v1) == 128,
                             "init_args_v1 must be 128 bytes.");

typedef nvshmemx_init_args_v2 nvshmemx_init_args_t;

typedef struct {
    int version;
    void *mpi_comm;
    nvshmemx_init_args_t args;
} nvshmemx_init_attr_v2;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmemx_init_attr_v2) == 144,
                             "init_attr_v2 must be 144 bytes.");

typedef struct {
    int version;
    void *mpi_comm;
    nvshmemx_init_args_t args;
} nvshmemx_init_attr_v1;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmemx_init_attr_v1) == 144,
                             "init_attr_v1 must be 144 bytes.");

typedef nvshmemx_init_attr_v2 nvshmemx_init_attr_t;

typedef struct {
    int version;
    int num_contexts;
    nvshmemx_team_uniqueid_t uniqueid;
    char padding[NVSHMEM_TEAM_CONFIG_V2_RESERVED_BYTES];
} nvshmem_team_config_v2;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmem_team_config_v2) == 64,
                             "team_config_v2 must be 64 bytes.");

typedef struct {
    int version;
    int num_contexts;
    char padding[NVSHMEM_TEAM_CONFIG_V1_RESERVED_BYTES];
} nvshmem_team_config_v1;
NVSHMEMI_TYPES_STATIC_ASSERT(sizeof(nvshmem_team_config_v1) == 64,
                             "team_config_v1 must be 64 bytes.");

typedef nvshmem_team_config_v2 nvshmem_team_config_t;

#define NVSHMEM_TEAM_UNIQUID_INITIALIZER TEAM_ULSCALAR_INVALID

#define NVSHMEM_INIT_ARGS_V2_IDENTIFIER (2 << 16) + sizeof(nvshmemx_init_args_t)
#define NVSHMEMX_INIT_ARGS_V2_INITIALIZER           \
    {NVSHMEM_INIT_ARGS_V2_IDENTIFIER, /* version */ \
     NVSHMEMX_UNIQUEID_ARGS_INITIALIZER,            \
     INIT_ARGS_SCALAR_INVALID,                      \
     {0}}

#define NVSHMEMX_INIT_ARGS_INITIALIZER                       \
    {(1 << 16) + sizeof(nvshmemx_init_args_t), /* version */ \
     NVSHMEMX_UNIQUEID_ARGS_INITIALIZER,                     \
     {0}}

#define NVSHMEM_INIT_ATTR_V2_IDENTIFIER (2 << 16) + sizeof(nvshmemx_init_attr_t)
#define NVSHMEMX_INIT_ATTR_INITIALIZER               \
    {NVSHMEM_INIT_ATTR_V2_IDENTIFIER, /* version */  \
     NULL,                            /* mpi_comm */ \
     NVSHMEMX_INIT_ARGS_V2_INITIALIZER}

#define NVSHMEM_INIT_ATTR_V1_IDENTIFIER (1 << 16) + sizeof(nvshmemx_init_attr_t)
#define NVSHMEMX_INIT_ATTR_V1_INITIALIZER            \
    {NVSHMEM_INIT_ATTR_V1_IDENTIFIER, /* version */  \
     NULL,                            /* mpi_comm */ \
     NVSHMEMX_INIT_ARGS_INITIALIZER}

#define NVSHMEM_TEAM_CONFIG_VERSION_2_IDENTIFIER (2 << 16) + sizeof(nvshmem_team_config_t)
#define NVSHMEM_TEAM_CONFIG_INITIALIZER                           \
    {NVSHMEM_TEAM_CONFIG_VERSION_2_IDENTIFIER, /* version */      \
     TEAM_CONFIG_SCALAR_INVALID,               /* num_contexts */ \
     NVSHMEM_TEAM_UNIQUID_INITIALIZER,         /* uniqueid */     \
     {0}}
#define NVSHMEM_TEAM_CONFIG_MASK_NUM_CONTEXTS 0x0000000000000001
#define NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID 0x0000000000000002

#undef NVSHMEMI_TYPES_STATIC_ASSERT

#endif /* NVSHMEM_TYPES_H */
