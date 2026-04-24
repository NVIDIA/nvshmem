/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMX_ERROR_H_
#define _NVSHMEMX_ERROR_H_
#if !defined __CUDACC_RTC__
#include <errno.h>  // for errno
#include <limits.h>
#include <stdio.h>   // for stderr, stdout, NULL
#include <string.h>  // IWYU pragma: keep for strerror
#include <stdlib.h>  // IWYU pragma: keep for exit
#else
#define fprintf(...)
#define exit(...)
#include <cuda/std/climits>
#include <cuda/std/cstdlib>
#endif

/* The !! idiom is used to convert non-boolean types to booleans.
 * Doing so in this case allows us to ensure that __builtin_expect
 * will be given a clean boolean value for the comparison. */
#define nvshmemxi_error_unlikely(x) __builtin_expect(!!(x), 0)

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
extern "C" {
#endif

/* Single source of truth for nvshmemx_status values.
 * This list is used to generate both the enum and the string table. */
#define NVSHMEMX_STATUS_LIST(X)            \
    X(NVSHMEMX_SUCCESS)                    \
    X(NVSHMEMX_ERROR_INVALID_VALUE)        \
    X(NVSHMEMX_ERROR_OUT_OF_MEMORY)        \
    X(NVSHMEMX_ERROR_NOT_SUPPORTED)        \
    X(NVSHMEMX_ERROR_SYMMETRY)             \
    X(NVSHMEMX_ERROR_GPU_NOT_SELECTED)     \
    X(NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED) \
    X(NVSHMEMX_ERROR_INTERNAL)

#define NVSHMEMX_GENERATE_ENUM(e) e,
#define NVSHMEMX_GENERATE_STRING(e) #e,

enum nvshmemx_status {
    NVSHMEMX_STATUS_LIST(NVSHMEMX_GENERATE_ENUM)
    NVSHMEMX_ERROR_SENTINEL = INT_MAX
};

#if !defined __CUDACC_RTC__
/* Human-readable string for nvshmemx_status values.
 * Safe to call from host code only (not available under __CUDACC_RTC__). */
static const char *const nvshmemx_status_strings[] = {NVSHMEMX_STATUS_LIST(NVSHMEMX_GENERATE_STRING)};

static inline const char *nvshmemx_status_string(int status) {
    if (status >= 0 && status < (int)(sizeof(nvshmemx_status_strings) / sizeof(nvshmemx_status_strings[0]))) {
        return nvshmemx_status_strings[status];
    }
    return "NVSHMEMX_ERROR_<unknown>";
}
#else
static inline const char *nvshmemx_status_string(int status) {
    (void)status;
    return "";
}
#endif

#undef NVSHMEMX_GENERATE_ENUM
#undef NVSHMEMX_GENERATE_STRING

#define NVSHMEMI_ERROR_EXIT(...)                                         \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
        exit(-1);                                                        \
    } while (0)

#define NVSHMEMI_ERROR_PRINT(...)                                        \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    } while (0)

#define NVSHMEMI_WARN_PRINT(...)      \
    do {                              \
        fprintf(stdout, "WARN: ");    \
        fprintf(stdout, __VA_ARGS__); \
        fprintf(stdout, "\n");        \
    } while (0)

#define NVSHMEMI_ERROR_JMP(status, err, label, ...)                                      \
    do {                                                                                 \
        fprintf(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, (err),     \
                nvshmemx_status_string((err)));                                          \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fprintf(stderr, "\n");                                                           \
        status = err;                                                                    \
        goto label;                                                                      \
    } while (0)

#define NVSHMEMI_NULL_ERROR_JMP(var, status, err, label, ...)          \
    do {                                                               \
        if (nvshmemxi_error_unlikely(var == NULL)) {                   \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                     \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)

#define NVSHMEMI_EQ_ERROR_JMP(status, expected, err, label, ...)                             \
    do {                                                                                     \
        if (nvshmemxi_error_unlikely(status == expected)) {                                  \
            fprintf(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, (status), \
                    nvshmemx_status_string((status)));                                       \
            fprintf(stderr, __VA_ARGS__);                                                    \
            fprintf(stderr, "\n");                                                           \
            status = err;                                                                    \
            goto label;                                                                      \
        }                                                                                    \
    } while (0)

#define NVSHMEMI_NE_ERROR_JMP(status, expected, err, label, ...)                                \
    do {                                                                                        \
        if (nvshmemxi_error_unlikely(status != expected)) {                                     \
            fprintf(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, (status),    \
                    nvshmemx_status_string((status)));                                          \
            fprintf(stderr, __VA_ARGS__);                                                       \
            fprintf(stderr, "\n");                                                              \
            status = err;                                                                       \
            goto label;                                                                         \
        }                                                                                       \
    } while (0)

#define NVSHMEMI_NZ_ERROR_JMP(status, err, label, ...)                                       \
    do {                                                                                     \
        if (nvshmemxi_error_unlikely(status != 0)) {                                          \
            fprintf(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, (status), \
                    nvshmemx_status_string((status)));                                       \
            fprintf(stderr, __VA_ARGS__);                                                    \
            fprintf(stderr, "\n");                                                           \
            status = err;                                                                    \
            goto label;                                                                      \
        }                                                                                    \
    } while (0)

#define NVSHMEMI_CHECK_ERROR_JMP(statement, status, err, label, ...) \
    do {                                                             \
        if (nvshmemxi_error_unlikely(statement)) {                   \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);          \
            fprintf(stderr, __VA_ARGS__);                            \
            fprintf(stderr, "\n");                                   \
            status = err;                                            \
            goto label;                                              \
        }                                                            \
    } while (0)

#define NVSHMEMI_NZ_EXIT(status, ...)                                                          \
    do {                                                                                       \
        if (nvshmemxi_error_unlikely(status != 0)) {                                           \
            fprintf(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    status, strerror(errno));                                                  \
            fprintf(stderr, __VA_ARGS__);                                                      \
            fprintf(stderr, "\n");                                                             \
            exit(-1);                                                                          \
        }                                                                                      \
    } while (0)

#define NVSHMEMI_NZ_SYSCHECK_EXIT(sys_status, ...)                                             \
    do {                                                                                       \
        if (nvshmemxi_error_unlikely((sys_status) != 0)) {                                     \
            fprintf(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    (sys_status), strerror(sys_status));                                       \
            fprintf(stderr, __VA_ARGS__);                                                      \
            fprintf(stderr, "\n");                                                             \
            exit(-1);                                                                          \
        }                                                                                      \
    } while (0)

#define NVSHMEMI_ERROR_RET(status, err, ...)                                      \
    do {                                                                          \
        fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, err);  \
        fprintf(stderr, __VA_ARGS__);                                             \
        fprintf(stderr, "\n");                                                    \
        status = err;                                                             \
        return status;                                                            \
    } while (0)

#define NVSHMEMI_NULL_ERROR_RET(var, status, err, ...)                  \
    do {                                                               \
        if (nvshmemxi_error_unlikely(var == NULL)) {                   \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                     \
            status = err;                                              \
            return status;                                             \
        }                                                              \
    } while (0)

#define NVSHMEMI_NZ_ERROR_RET(status, err, ...)                                         \
    do {                                                                                \
        if (nvshmemxi_error_unlikely(status != 0)) {                                    \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
            status = err;                                                               \
            return status;                                                              \
        }                                                                               \
    } while (0)

#define NVSHMEMI_NE_ERROR_RET(status, expected, err, ...)                                \
    do {                                                                                \
        if (nvshmemxi_error_unlikely(status != expected)) {                             \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
            status = err;                                                               \
            return status;                                                              \
        }                                                                               \
    } while (0)

#if defined __cplusplus || defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_BUILD_LTOIR_LIBRARY
}
#endif

#endif
