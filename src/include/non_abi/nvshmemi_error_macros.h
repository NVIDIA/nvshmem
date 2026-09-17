/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_ERROR_MACROS_H
#define NVSHMEMI_ERROR_MACROS_H
#if !defined __CUDACC_RTC__
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NVSHMEMI_ERROR_FPRINTF(...) fprintf(__VA_ARGS__)
#define NVSHMEMI_ERROR_EXIT_PROCESS(status) exit(status)
#define NVSHMEMI_ERROR_STRERROR(status) strerror(status)
#else
#define NVSHMEMI_ERROR_FPRINTF(...)
#define NVSHMEMI_ERROR_EXIT_PROCESS(status)
#define NVSHMEMI_ERROR_STRERROR(status) ""
#endif

#include "device_host/nvshmemx_status.h"

/* The !! idiom is used to convert non-boolean types to booleans.
 * Doing so in this case allows us to ensure that __builtin_expect
 * will be given a clean boolean value for the comparison. */
#define nvshmemxi_error_unlikely(x) __builtin_expect(!!(x), 0)

#define NVSHMEMI_ERROR_EXIT(...)                                                        \
    do {                                                                                \
        NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                    \
        NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                           \
        NVSHMEMI_ERROR_EXIT_PROCESS(-1);                                                \
    } while (0)

#define NVSHMEMI_ERROR_PRINT(...)                                                       \
    do {                                                                                \
        NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                    \
        NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                           \
    } while (0)

#define NVSHMEMI_WARN_PRINT(...)                     \
    do {                                             \
        NVSHMEMI_ERROR_FPRINTF(stdout, "WARN: ");    \
        NVSHMEMI_ERROR_FPRINTF(stdout, __VA_ARGS__); \
        NVSHMEMI_ERROR_FPRINTF(stdout, "\n");        \
    } while (0)

#define NVSHMEMI_ERROR_JMP(status, err, label, ...)                                                \
    do {                                                                                           \
        NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, (err), \
                               nvshmemx_status_string((err)));                                     \
        NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                               \
        NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                                      \
        status = err;                                                                              \
        goto label;                                                                                \
    } while (0)

#define NVSHMEMI_NULL_ERROR_JMP(var, status, err, label, ...)                         \
    do {                                                                              \
        if (nvshmemxi_error_unlikely(var == NULL)) {                                  \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                              \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                     \
            status = err;                                                             \
            goto label;                                                               \
        }                                                                             \
    } while (0)

#define NVSHMEMI_EQ_ERROR_JMP(status, expected, err, label, ...)                                \
    do {                                                                                        \
        if (nvshmemxi_error_unlikely(status == expected)) {                                     \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, \
                                   (status), nvshmemx_status_string((status)));                 \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                        \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                               \
            status = err;                                                                       \
            goto label;                                                                         \
        }                                                                                       \
    } while (0)

#define NVSHMEMI_NE_ERROR_JMP(status, expected, err, label, ...)                                \
    do {                                                                                        \
        if (nvshmemxi_error_unlikely(status != expected)) {                                     \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, \
                                   (status), nvshmemx_status_string((status)));                 \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                        \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                               \
            status = err;                                                                       \
            goto label;                                                                         \
        }                                                                                       \
    } while (0)

#define NVSHMEMI_NZ_ERROR_JMP(status, err, label, ...)                                          \
    do {                                                                                        \
        if (nvshmemxi_error_unlikely(status != 0)) {                                            \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: error status: %d (%s) ", __FILE__, __LINE__, \
                                   (status), nvshmemx_status_string((status)));                 \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                        \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                               \
            status = err;                                                                       \
            goto label;                                                                         \
        }                                                                                       \
    } while (0)

#define NVSHMEMI_CHECK_ERROR_JMP(statement, status, err, label, ...)       \
    do {                                                                   \
        if (nvshmemxi_error_unlikely(statement)) {                         \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: ", __FILE__, __LINE__); \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                   \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                          \
            status = err;                                                  \
            goto label;                                                    \
        }                                                                  \
    } while (0)

#define NVSHMEMI_NZ_EXIT(status, ...)                                                           \
    do {                                                                                        \
        if (nvshmemxi_error_unlikely(status != 0)) {                                            \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: non-zero status: %d: %s, exiting... ",       \
                                   __FILE__, __LINE__, status, NVSHMEMI_ERROR_STRERROR(errno)); \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                        \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                               \
            NVSHMEMI_ERROR_EXIT_PROCESS(-1);                                                    \
        }                                                                                       \
    } while (0)

#define NVSHMEMI_NZ_SYSCHECK_EXIT(sys_status, ...)                                        \
    do {                                                                                  \
        if (nvshmemxi_error_unlikely((sys_status) != 0)) {                                \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", \
                                   __FILE__, __LINE__, (sys_status),                      \
                                   NVSHMEMI_ERROR_STRERROR(sys_status));                  \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                  \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                         \
            NVSHMEMI_ERROR_EXIT_PROCESS(-1);                                              \
        }                                                                                 \
    } while (0)

#define NVSHMEMI_ERROR_RET(status, err, ...)                                                    \
    do {                                                                                        \
        NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, err); \
        NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                            \
        NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                                   \
        status = err;                                                                           \
        return status;                                                                          \
    } while (0)

#define NVSHMEMI_NULL_ERROR_RET(var, status, err, ...)                                \
    do {                                                                              \
        if (nvshmemxi_error_unlikely(var == NULL)) {                                  \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                              \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                     \
            status = err;                                                             \
            return status;                                                            \
        }                                                                             \
    } while (0)

#define NVSHMEMI_NZ_ERROR_RET(status, err, ...)                                               \
    do {                                                                                      \
        if (nvshmemxi_error_unlikely(status != 0)) {                                          \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, \
                                   status);                                                   \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                      \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                             \
            status = err;                                                                     \
            return status;                                                                    \
        }                                                                                     \
    } while (0)

#define NVSHMEMI_NE_ERROR_RET(status, expected, err, ...)                                     \
    do {                                                                                      \
        if (nvshmemxi_error_unlikely(status != expected)) {                                   \
            NVSHMEMI_ERROR_FPRINTF(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, \
                                   status);                                                   \
            NVSHMEMI_ERROR_FPRINTF(stderr, __VA_ARGS__);                                      \
            NVSHMEMI_ERROR_FPRINTF(stderr, "\n");                                             \
            status = err;                                                                     \
            return status;                                                                    \
        }                                                                                     \
    } while (0)

#endif
