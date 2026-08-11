/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_REDUCE_COMMON_CPU_H
#define NVSHMEMTEST_REDUCE_COMMON_CPU_H

#include <cuda_runtime.h>
#include <iostream>
#include "stdio.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_tensor.h"
#include "utils.h"
#include "coll_common.h"
#include "cuda/std/tuple"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#define NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                              SC_PREFIX, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int, opname)                     \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long, opname)          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t, opname)                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMTEST_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC,       \
                                                               SC_SUFFIX, SC_PREFIX, opname)   \
    NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                          SC_PREFIX, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char, opname)                 \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short, opname)                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, half, half, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, bfloat16, __nv_bfloat16, opname)            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float, opname)                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMTEST_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                            SC_PREFIX, opname)                   \
    NVSHMEMTEST_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                           SC_PREFIX, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexf, double complex, opname)             \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexd, float complex, opname)
#else
#define NVSHMEMTEST_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                            SC_PREFIX, opname)                   \
    NVSHMEMTEST_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                           SC_PREFIX, opname)
#endif

#define NVSHMEMTEST_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                              SC_PREFIX)                           \
    NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                          SC_PREFIX, and)                          \
    NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                          SC_PREFIX, or)                           \
                                                                                                   \
    NVSHMEMTEST_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                           SC_PREFIX, min)                         \
    NVSHMEMTEST_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                           SC_PREFIX, max)                         \
                                                                                                   \
    NVSHMEMTEST_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,       \
                                                        SC_PREFIX, sum)

__device__ unsigned long long int errs_d;

#define reduce_and(op1, op2) ((op1) & (op2))
#define reduce_or(op1, op2) ((op1) | (op2))
#define reduce_xor(op1, op2) ((op1) ^ (op2))
#define reduce_max(op1, op2) ((op1) > (op2) ? (op1) : (op2))
#define reduce_min(op1, op2) ((op1) < (op2)) ? (op1) : (op2)
#define reduce_sum(op1, op2) ((op1) + (op2))
#define reduce_prod(op1, op2) ((op1) * (op2))

__host__ __device__ bool nvshmtest_are_strings_same(const char *str1, const char *str2) {
    while (*str1 != '\0') {
        if ((*str2 == '\0') || (*str1 != *str2)) {
            return 0;
        }
        str1++;
        str2++;
    }
    if (*str2 == '\0') {
        return 1;
    } else {
        return 0;
    }
}

#define DECL_INIT_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)              \
    __device__ void init_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(nvshmem_team_t team, \
                                                                    TYPE *source, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_INIT_REDUCE_DATA)
#undef DECL_INIT_REDUCE_DATA

#define DECL_INIT_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                                           \
    __global__ void init_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                                size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_INIT_REDUCE_DATA_KERNEL)
#undef DECL_INIT_REDUCE_DATA_KERNEL

#define DECL_VALIDATE_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    __device__ void validate_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(        \
        nvshmem_team_t team, TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_VALIDATE_REDUCE_DATA)
#undef DECL_VALIDATE_REDUCE_DATA

#define DECL_VALIDATE_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                             \
    __global__ void validate_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team, \
                                                                    TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_VALIDATE_REDUCE_DATA_KERNEL)
#undef DECL_VALIDATE_REDUCE_DATA_KERNEL

#define DECL_RESET_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)              \
    __device__ void reset_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(nvshmem_team_t team, \
                                                                     TYPE *dest, size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(DECL_RESET_REDUCE_DATA)
#undef DECL_RESET_REDUCE_DATA

#define DECL_RESET_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                                         \
    __global__ void reset_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                                 size_t nelems);
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(DECL_RESET_REDUCE_DATA_KERNEL)
#undef DECL_RESET_REDUCE_DATA_KERNEL

#define INIT_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                             \
    __device__ void init_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(nvshmem_team_t team,           \
                                                                    TYPE *source, size_t nelems) { \
        int my_pe = nvshmem_team_my_pe(team);                                                      \
        int n_pes = nvshmem_team_n_pes(team);                                                      \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                 \
        int groupSize = nvshmtest_##SC##_size();                                                   \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                       \
            if (nvshmtest_are_strings_same(#OP, "prod"))                                           \
                source[i] = assign<TYPE>((my_pe == n_pes - 1) ? n_pes : 1);                        \
            else if (nvshmtest_are_strings_same(#TYPENAME, "half"))                                \
                source[i] = assign<TYPE>((my_pe >= 2048) ? 0 : 1);                                 \
            else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                            \
                source[i] = assign<TYPE>((my_pe >= 256) ? 0 : 1);                                  \
            else                                                                                   \
                source[i] = assign<TYPE>(my_pe + i);                                               \
        }                                                                                          \
        nvshmtest_##SC##_sync();                                                                   \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(INIT_REDUCE_DATA)
#undef INIT_REDUCE_DATA

#define INIT_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                                                \
    __global__ void init_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                                size_t nelems) {                   \
        init_##TYPENAME##_##OP##_reduce_data_block(team, source, nelems);                          \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(INIT_REDUCE_DATA_KERNEL)
#undef INIT_REDUCE_DATA_KERNEL

#define VALIDATE_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                       \
    __device__ void validate_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(                         \
        nvshmem_team_t team, TYPE *dest, size_t nelems) {                                        \
        int n_pes = nvshmem_team_n_pes(team);                                                    \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
                                                                                                 \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                     \
            TYPE expected;                                                                       \
            if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                                 \
                nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                               \
                expected = assign<TYPE>(1);                                                      \
            else                                                                                 \
                expected = assign<TYPE>(i);                                                      \
                                                                                                 \
            if (nvshmtest_are_strings_same(#OP, "prod"))                                         \
                expected = assign<TYPE>(n_pes);                                                  \
            else {                                                                               \
                for (size_t j = 1; j < n_pes; j++) {                                             \
                    if (nvshmtest_are_strings_same(#TYPENAME, "half"))                           \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 2048) ? 0 : 1)); \
                    else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                  \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 256) ? 0 : 1));  \
                    else                                                                         \
                        expected = reduce_##OP(expected, assign<TYPE>(j + i));                   \
                }                                                                                \
            }                                                                                    \
            if (dest[i] != expected) {                                                           \
                printf(NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC), to_printable(dest[i]),     \
                       to_printable(expected), i, nelems, team);                                 \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(VALIDATE_REDUCE_DATA)
#undef VALIDATE_REDUCE_DATA

#define VALIDATE_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                                          \
    __global__ void validate_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team,         \
                                                                    TYPE *dest, size_t nelems) { \
        validate_##TYPENAME##_##OP##_reduce_data(team, dest, nelems);                            \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(VALIDATE_REDUCE_DATA_KERNEL)
#undef VALIDATE_REDUCE_DATA_KERNEL

#define RESET_REDUCE_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)                           \
    __device__ void reset_##TYPENAME##_##OP##_reduce_data##SC_SUFFIX(nvshmem_team_t team,         \
                                                                     TYPE *dest, size_t nelems) { \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                      \
            dest[i] = assign<TYPE>(0);                                                            \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(RESET_REDUCE_DATA)
#undef RESET_REDUCE_DATA

#define RESET_REDUCE_DATA_KERNEL(TYPENAME, TYPE, OP)                                              \
    __global__ void reset_##TYPENAME##_##OP##_reduce_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                                 size_t nelems) {                 \
        reset_##TYPENAME##_##OP##_reduce_data(team, dest, nelems);                                \
    }
NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(RESET_REDUCE_DATA_KERNEL)
#undef RESET_REDUCE_DATA_KERNEL

/****** Tile variants *********/
#define DECL_INIT_TILE_DATA_KERNEL(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)          \
    __global__ void init_##TYPENAME##_tile_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                       size_t nelems);
NVSHMEMTEST_TILE_REPT_TYPES(DECL_INIT_TILE_DATA_KERNEL, , , , )
#undef DECL_INIT_TILE_DATA_KERNEL

#define INIT_TILE_DATA_KERNEL(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)               \
    __global__ void init_##TYPENAME##_tile_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                       size_t nelems) {                   \
        int my_pe = nvshmem_team_my_pe(team);                                             \
        int n_pes = nvshmem_team_n_pes(team);                                             \
        int myIdx = threadIdx.x + blockIdx.x * blockDim.x;                                \
        int groupSize = blockDim.x * gridDim.x;                                           \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                              \
            if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                          \
                nvshmtest_are_strings_same(#TYPENAME, "cutlass_half_t"))                  \
                source[i] = assign<TYPE>((my_pe >= 2048) ? 0 : 1);                        \
            else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16") ||                 \
                     nvshmtest_are_strings_same(#TYPENAME, "cutlass_bfloat16_t"))         \
                source[i] = assign<TYPE>((my_pe >= 256) ? 0 : 1);                         \
            else                                                                          \
                source[i] = assign<TYPE>(my_pe + i);                                      \
        }                                                                                 \
    }

NVSHMEMTEST_TILE_REPT_TYPES(INIT_TILE_DATA_KERNEL, , , , )

#if CUTLASS_ENABLED == 1
INIT_TILE_DATA_KERNEL(, , , , cutlass_half_t, cutlass::half_t)
INIT_TILE_DATA_KERNEL(, , , , cutlass_bfloat16_t, cutlass::bfloat16_t)
#endif
#undef INIT_TILE_DATA_KERNEL

#define DECL_VALIDATE_ALLREDUCE_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    template <typename src_tensor_t, typename dst_tensor_t>                             \
    __device__ void validate_##TYPENAME##_##OP##_allreduce_tile_data##SC_SUFFIX(        \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dest);

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES_AND_OPS(DECL_VALIDATE_ALLREDUCE_TILE_DATA)
#undef DECL_VALIDATE_ALLREDUCE_TILE_DATA

#define VALIDATE_ALLREDUCE_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)               \
    template <typename src_tensor_t, typename dst_tensor_t, int major_dim, int minor_dim>        \
    __device__ void validate_##TYPENAME##_##OP##_allreduce_tile_data##SC_SUFFIX(                 \
        nvshmem_team_t team, size_t tile_start_elem_idx, src_tensor_t src, dst_tensor_t dest,    \
        size_t tensor_size_major, size_t tensor_size_minor, size_t boundary_major,               \
        size_t boundary_minor) {                                                                 \
        size_t flat_elem_idx;                                                                    \
        size_t idx_in_tile;                                                                      \
        int npes = nvshmem_team_n_pes(team);                                                     \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        size_t tot_elem_in_tile =                                                                \
            cuda::std::get<0>(dest.shape()) * cuda::std::get<1>(dest.shape());                   \
        TYPE expected = assign<TYPE>(0);                                                         \
        TYPE temp = assign<TYPE>(0);                                                             \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {                        \
            idx_in_tile = ((i % cuda::std::get<major_dim>(dest.shape())) *                       \
                           cuda::std::get<major_dim>(dest.stride())) +                           \
                          ((i / cuda::std::get<major_dim>(dest.shape())) *                       \
                           cuda::std::get<minor_dim>(dest.stride()));                            \
            flat_elem_idx = (tile_start_elem_idx + idx_in_tile);                                 \
            if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                                 \
                nvshmtest_are_strings_same(#TYPENAME, "cutlass_half_t") ||                       \
                nvshmtest_are_strings_same(#TYPENAME, "cutlass_bfloat16_t") ||                   \
                nvshmtest_are_strings_same(#TYPENAME, "bfloat16")) {                             \
                expected = assign<TYPE>(1);                                                      \
                for (size_t j = 1; j < npes; j++) {                                              \
                    if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                         \
                        nvshmtest_are_strings_same(#TYPENAME, "cutlass_half_t"))                 \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 2048) ? 0 : 1)); \
                    else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16") ||                \
                             nvshmtest_are_strings_same(#TYPENAME, "cutlass_bfloat16_t"))        \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 256) ? 0 : 1));  \
                }                                                                                \
            } else {                                                                             \
                if (nvshmtest_are_strings_same(#OP, "sum")) {                                    \
                    expected = assign<TYPE>((npes * flat_elem_idx) + (npes * (npes - 1) / 2));   \
                } else if (nvshmtest_are_strings_same(#OP, "min")) {                             \
                    expected = assign<TYPE>(flat_elem_idx);                                      \
                } else if (nvshmtest_are_strings_same(#OP, "max")) {                             \
                    expected = assign<TYPE>(flat_elem_idx + npes - 1);                           \
                }                                                                                \
            }                                                                                    \
                                                                                                 \
            if (((flat_elem_idx % tensor_size_major) < boundary_major) &&                        \
                ((flat_elem_idx / tensor_size_major) < boundary_minor) &&                        \
                (*(dest.data() + idx_in_tile) != expected)) {                                    \
                printf(NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC),                            \
                       to_printable(*(dest.data() + idx_in_tile)), to_printable(expected),       \
                       flat_elem_idx, tot_elem_in_tile, team);                                   \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES_AND_OPS(VALIDATE_ALLREDUCE_TILE_DATA)
#if CUTLASS_ENABLED == 1
VALIDATE_ALLREDUCE_TILE_DATA(sum, block, _block, x, cutlass_half_t, cutlass::half_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, thread, , x, cutlass_half_t, cutlass::half_t)
VALIDATE_ALLREDUCE_TILE_DATA(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t)
#endif
#undef VALIDATE_ALLREDUCE_TILE_DATA

#define VALIDATE_ALLREDUCE_TILE_1D_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)            \
    template <typename src_tensor_t, typename dst_tensor_t, int major_dim, int minor_dim>        \
    __device__ void validate_##TYPENAME##_##OP##_allreduce_tile_1D_data##SC_SUFFIX(              \
        nvshmem_team_t team, size_t tile_start_elem_idx, src_tensor_t src, dst_tensor_t dest,    \
        size_t tensor_size_major, size_t boundary_major) {                                       \
        size_t flat_elem_idx;                                                                    \
        size_t idx_in_tile;                                                                      \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int npes = nvshmem_team_n_pes(team);                                                     \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        size_t tot_elem_in_tile = cuda::std::get<0>(dest.shape());                               \
        TYPE expected = assign<TYPE>(0);                                                         \
        TYPE temp = assign<TYPE>(0);                                                             \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {                        \
            idx_in_tile = ((i) * cuda::std::get<major_dim>(dest.stride()));                      \
            flat_elem_idx = (tile_start_elem_idx + idx_in_tile);                                 \
            if (nvshmtest_are_strings_same(#TYPENAME, "half") ||                                 \
                nvshmtest_are_strings_same(#TYPENAME, "bfloat16")) {                             \
                expected = assign<TYPE>(1);                                                      \
                for (size_t j = 1; j < npes; j++) {                                              \
                    if (nvshmtest_are_strings_same(#TYPENAME, "half"))                           \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 2048) ? 0 : 1)); \
                    else if (nvshmtest_are_strings_same(#TYPENAME, "bfloat16"))                  \
                        expected = reduce_##OP(expected, assign<TYPE>((my_pe >= 256) ? 0 : 1));  \
                }                                                                                \
            } else {                                                                             \
                if (nvshmtest_are_strings_same(#OP, "sum")) {                                    \
                    expected = assign<TYPE>((npes * flat_elem_idx) + (npes * (npes - 1) / 2));   \
                } else if (nvshmtest_are_strings_same(#OP, "min")) {                             \
                    expected = assign<TYPE>(flat_elem_idx);                                      \
                } else if (nvshmtest_are_strings_same(#OP, "max")) {                             \
                    expected = assign<TYPE>(flat_elem_idx + npes - 1);                           \
                }                                                                                \
            }                                                                                    \
            assert(flat_elem_idx < tensor_size_major);                                           \
            if (((flat_elem_idx) < boundary_major) &&                                            \
                (*(dest.data() + idx_in_tile) != expected)) {                                    \
                printf(NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC),                            \
                       to_printable(*(dest.data() + idx_in_tile)), to_printable(expected),       \
                       flat_elem_idx, tot_elem_in_tile, team);                                   \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES_AND_OPS(VALIDATE_ALLREDUCE_TILE_1D_DATA)
#undef VALIDATE_ALLREDUCE_TILE_1D_DATA

#define DECL_RESET_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)              \
    template <typename dst_tensor_t>                                                    \
    __device__ void reset_##TYPENAME##_tile_data##SC_SUFFIX(size_t tile_start_elem_idx, \
                                                            dst_tensor_t dest);
NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(DECL_RESET_TILE_DATA, )
#undef DECL_RESET_TILE_DATA

#define RESET_ALLREDUCE_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)         \
    template <typename dst_tensor_t, int major_dim, int minor_dim>                      \
    __device__ void reset_##TYPENAME##_tile_data##SC_SUFFIX(size_t tile_start_elem_idx, \
                                                            dst_tensor_t dest) {        \
        size_t idx_in_tile;                                                             \
        int myIdx = nvshmtest_thread_id_in_##SC();                                      \
        int groupSize = nvshmtest_##SC##_size();                                        \
        size_t tot_elem_in_tile =                                                       \
            cuda::std::get<0>(dest.shape()) * cuda::std::get<1>(dest.shape());          \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {               \
            idx_in_tile = ((i % cuda::std::get<major_dim>(dest.shape())) *              \
                           cuda::std::get<major_dim>(dest.stride())) +                  \
                          ((i / cuda::std::get<major_dim>(dest.shape())) *              \
                           cuda::std::get<minor_dim>(dest.stride()));                   \
            *(dest.data() + idx_in_tile) = assign<TYPE>(0);                             \
        }                                                                               \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(RESET_ALLREDUCE_TILE_DATA, )
#if CUTLASS_ENABLED == 1
RESET_ALLREDUCE_TILE_DATA(sum, block, _block, x, cutlass_half_t, cutlass::half_t)
RESET_ALLREDUCE_TILE_DATA(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
RESET_ALLREDUCE_TILE_DATA(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t)
RESET_ALLREDUCE_TILE_DATA(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
RESET_ALLREDUCE_TILE_DATA(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t)
RESET_ALLREDUCE_TILE_DATA(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t)
RESET_ALLREDUCE_TILE_DATA(sum, thread, , x, cutlass_half_t, cutlass::half_t)
RESET_ALLREDUCE_TILE_DATA(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t)
#endif
#undef RESET_ALLREDUCE_TILE_DATA

#endif /* NVSHMEMTEST_REDUCE_COMMON_CPU_H */
