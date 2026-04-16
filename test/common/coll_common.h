/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef COLL_COMMON_H
#define COLL_COMMON_H

#include "cuda.h"

#define COLL_CHECK_ERRS_D()                                                     \
    do {                                                                        \
        cudaMemcpyFromSymbol(&errs, errs_d, sizeof(unsigned long long int), 0); \
        if (errs) {                                                             \
            printf("Validation errors found\n");                                \
            goto out;                                                           \
        }                                                                       \
    } while (0)

#define COLL_CHECK_ERRS()                        \
    do {                                         \
        if (errs) {                              \
            printf("Validation errors found\n"); \
            goto out;                            \
        }                                        \
    } while (0)

/* Helper functions for tile variants */

#define NVSHMEMTEST_TILE_REPT_TYPES(NVSHMEMI_FN_TEMPLATE, OP, SC, SC_SUFFIX, SC_PREFIX) \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, half, half)                      \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, bfloat16, __nv_bfloat16)         \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, float, float)

#define NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(NVSHMEMI_FN_TEMPLATE, OP)    \
    NVSHMEMTEST_TILE_REPT_TYPES(NVSHMEMI_FN_TEMPLATE, OP, thread, , )       \
    NVSHMEMTEST_TILE_REPT_TYPES(NVSHMEMI_FN_TEMPLATE, OP, warp, _warp, x)   \
    NVSHMEMTEST_TILE_REPT_TYPES(NVSHMEMI_FN_TEMPLATE, OP, block, _block, x) \
    NVSHMEMTEST_TILE_REPT_TYPES(NVSHMEMI_FN_TEMPLATE, OP, warpgroup, _warpgroup, x)

#define NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES_AND_OPS(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(NVSHMEMI_FN_TEMPLATE, sum)        \
    NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(NVSHMEMI_FN_TEMPLATE, min)        \
    NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(NVSHMEMI_FN_TEMPLATE, max)

#define NVSHMEMTEST_TILE_REPT_SCOPES_AND_VLEN(NVSHMEMI_FN_TEMPLATE, OP, TYPENAME, TYPE)    \
    NVSHMEMTEST_TILE_REPT_VLEN(NVSHMEMI_FN_TEMPLATE, OP, thread, , , TYPENAME, TYPE)       \
    NVSHMEMTEST_TILE_REPT_VLEN(NVSHMEMI_FN_TEMPLATE, OP, warp, _warp, x, TYPENAME, TYPE)   \
    NVSHMEMTEST_TILE_REPT_VLEN(NVSHMEMI_FN_TEMPLATE, OP, block, _block, x, TYPENAME, TYPE) \
    NVSHMEMTEST_TILE_REPT_VLEN(NVSHMEMI_FN_TEMPLATE, OP, warpgroup, _warpgroup, x, TYPENAME, TYPE)

#define NVSHMEMTEST_TILE_REPT_VLEN(NVSHMEMI_FN_TEMPLATE, OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, \
                                   TYPE)                                                         \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, 4)                        \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, 2)                        \
    NVSHMEMI_FN_TEMPLATE(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, 1)

#endif /* COLL_COMMON_H */
