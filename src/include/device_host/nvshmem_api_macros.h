/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_API_MACROS_H
#define NVSHMEM_API_MACROS_H

#if !defined __CUDACC_RTC__
#include <stddef.h>
#include <stdint.h>
#else
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#endif

#include <cuda_runtime.h>
#ifdef NVSHMEM_COMPLEX_SUPPORT
#include <complex.h>
#endif
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "non_abi/nvshmem_build_options.h"

#if !defined(__cplusplus)
typedef __nv_bfloat16_raw __nv_bfloat16;
typedef __half_raw half;
#endif

#define NVSHMEMI_DEVICE_PREFIX __device__

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(bfloat16, __nv_bfloat16)                  \
    NVSHMEMI_FN_TEMPLATE(half, half)                               \
    NVSHMEMI_FN_TEMPLATE(float, float)                             \
    NVSHMEMI_FN_TEMPLATE(double, double)                           \
    NVSHMEMI_FN_TEMPLATE(char, char)                               \
    NVSHMEMI_FN_TEMPLATE(short, short)                             \
    NVSHMEMI_FN_TEMPLATE(schar, signed char)                       \
    NVSHMEMI_FN_TEMPLATE(int, int)                                 \
    NVSHMEMI_FN_TEMPLATE(long, long)                               \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                      \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char)                     \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                   \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                       \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                     \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)            \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t)                             \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t)                           \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                           \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                           \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t)                           \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t)                         \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                         \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                         \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                             \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC) \
    NVSHMEMI_FN_TEMPLATE(SC, bfloat16, __nv_bfloat16)                             \
    NVSHMEMI_FN_TEMPLATE(SC, half, half)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, float, float)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, double, double)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, char, char)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, short, short)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, int, int)                                            \
    NVSHMEMI_FN_TEMPLATE(SC, long, long)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long)                                 \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char)                                \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long)                                \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long)                       \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, bfloat16, __nv_bfloat16)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, half, half)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long)                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long)             \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(NVSHMEMI_FN_TEMPLATE)             \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )     \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x) \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#define NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(8)                           \
    NVSHMEMI_FN_TEMPLATE(16)                          \
    NVSHMEMI_FN_TEMPLATE(32)                          \
    NVSHMEMI_FN_TEMPLATE(64)                          \
    NVSHMEMI_FN_TEMPLATE(128)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(8, int8_t)                             \
    NVSHMEMI_FN_TEMPLATE(16, int16_t)                           \
    NVSHMEMI_FN_TEMPLATE(32, int32_t)                           \
    NVSHMEMI_FN_TEMPLATE(64, int64_t)                           \
    NVSHMEMI_FN_TEMPLATE(128, int4)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SCOPE, SC_SUFFIX, SC_PREFIX) \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 8)                                       \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 16)                                      \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 32)                                      \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 64)                                      \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 128)

#define NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(short, short)                     \
    NVSHMEMI_FN_TEMPLATE(int, int)                         \
    NVSHMEMI_FN_TEMPLATE(long, long)                       \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)              \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)           \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)               \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)             \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)    \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                   \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                   \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                 \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                 \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                     \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short, opname)                     \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int, opname)                         \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long, opname)              \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t, opname)                               \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)      \
    NVSHMEMI_FN_TEMPLATE(char, char, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(schar, signed char, opname)                          \
    NVSHMEMI_FN_TEMPLATE(short, short, opname)                                \
    NVSHMEMI_FN_TEMPLATE(int, int, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(long, long, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(longlong, long long, opname)                         \
    NVSHMEMI_FN_TEMPLATE(bfloat16, __nv_bfloat16, opname)                     \
    NVSHMEMI_FN_TEMPLATE(half, half, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(float, float, opname)                                \
    NVSHMEMI_FN_TEMPLATE(double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)  \
    NVSHMEMI_FN_TEMPLATE(complexf, double complex, opname)                 \
    NVSHMEMI_FN_TEMPLATE(complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)
#endif

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t, opname)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)      \
    NVSHMEMI_FN_TEMPLATE(SC, char, char, opname)                                             \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char, opname)                                     \
    NVSHMEMI_FN_TEMPLATE(SC, short, short, opname)                                           \
    NVSHMEMI_FN_TEMPLATE(SC, int, int, opname)                                               \
    NVSHMEMI_FN_TEMPLATE(SC, long, long, opname)                                             \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, bfloat16, __nv_bfloat16, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, half, half, opname)                                             \
    NVSHMEMI_FN_TEMPLATE(SC, float, float, opname)                                           \
    NVSHMEMI_FN_TEMPLATE(SC, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)  \
    NVSHMEMI_FN_TEMPLATE(SC, complexf, double complex, opname)                            \
    NVSHMEMI_FN_TEMPLATE(SC, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)
#endif

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                           SC_PREFIX, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char, opname)                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short, opname)              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int, opname)                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long, opname)                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long, opname)       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                            SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,      \
                                                       SC_PREFIX, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int, opname)                             \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long, opname)                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, bfloat16, __nv_bfloat16, opname)              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, half, half, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                        SC_PREFIX, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexf, double complex, opname)          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                        SC_PREFIX, opname)
#endif

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, and)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, or)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, xor)  \
                                                                       \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, min) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, max) \
                                                                       \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, prod)   \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                           SC_PREFIX)                           \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, and)                          \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, or)                           \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, xor)                          \
                                                                                                \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                        SC_PREFIX, min)                         \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                        SC_PREFIX, max)                         \
                                                                                                \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,       \
                                                     SC_PREFIX, prod)                           \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,       \
                                                     SC_PREFIX, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)             \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )     \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x) \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#endif /* NVSHMEM_API_MACROS_H */
