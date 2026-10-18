/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_PLUGIN_LOG_H
#define NVSHMEMI_PLUGIN_LOG_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * These symbols are supplied by libnvshmem_host. Keep them weak so that
 * standalone and third-party plugin tests retain their legacy output path.
 */
void nvshmem_debug_log(int level, unsigned long flags, const char *filefunc, int line,
                       const char *fmt, ...) __attribute__((weak));
void nvshmem_debug_printf(const char *fmt, ...) __attribute__((weak));
int nvshmem_debug_file_active(void) __attribute__((weak));

#if defined(__cplusplus)
}
#endif

enum {
    NVSHMEMI_PLUGIN_LOG_INFO = 3,
    NVSHMEMI_PLUGIN_LOG_TRACE = 5,
    NVSHMEMI_PLUGIN_LOG_TRANSPORT = 16,
    NVSHMEMI_PLUGIN_LOG_BOOTSTRAP = 64
};

static inline char *nvshmemi_plugin_format(const char *fmt, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    int length = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);
    if (length < 0) {
        return NULL;
    }

    char *message = (char *)malloc((size_t)length + 1);
    if (message == NULL) {
        return NULL;
    }
    (void)vsnprintf(message, (size_t)length + 1, fmt, args);
    return message;
}

static inline void nvshmemi_plugin_log(int level, unsigned long flags, FILE *fallback,
                                       const char *filefunc, int line, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char *message = nvshmemi_plugin_format(fmt, args);
    va_end(args);

    if (message == NULL) {
        return;
    }
    if (nvshmem_debug_log != NULL && nvshmem_debug_file_active != NULL &&
        nvshmem_debug_file_active()) {
        nvshmem_debug_log(level, flags, filefunc, line, "%s", message);
    } else {
        fprintf(fallback, "%s:%d: %s\n", filefunc, line, message);
    }
    free(message);
}

static inline void nvshmemi_plugin_printf(FILE *fallback, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char *message = nvshmemi_plugin_format(fmt, args);
    va_end(args);

    if (message == NULL) {
        return;
    }
    if (nvshmem_debug_printf != NULL) {
        nvshmem_debug_printf("%s", message);
    } else {
        fprintf(fallback, "%s", message);
        fflush(fallback);
    }
    free(message);
}

static inline void nvshmemi_plugin_fprintf(FILE *fallback, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char *message = nvshmemi_plugin_format(fmt, args);
    va_end(args);

    if (message == NULL) {
        return;
    }
    if (nvshmem_debug_printf != NULL && nvshmem_debug_file_active != NULL &&
        nvshmem_debug_file_active()) {
        nvshmem_debug_printf("%s", message);
    } else {
        fprintf(fallback, "%s", message);
        fflush(fallback);
    }
    free(message);
}

static inline void nvshmemi_plugin_warn_at(FILE *fallback, const char *file, const char *function,
                                           int line, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char *message = nvshmemi_plugin_format(fmt, args);
    va_end(args);

    if (message == NULL) {
        return;
    }
    if (nvshmem_debug_printf != NULL && nvshmem_debug_file_active != NULL &&
        nvshmem_debug_file_active()) {
        nvshmem_debug_printf("%s:%s:%d: %s\n", file, function, line, message);
    } else {
        fprintf(fallback, "%s:%s:%d: %s\n", file, function, line, message);
        fflush(fallback);
    }
    free(message);
}

static inline void nvshmemi_plugin_warn(FILE *fallback, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char *message = nvshmemi_plugin_format(fmt, args);
    va_end(args);

    if (message == NULL) {
        return;
    }
    if (nvshmem_debug_printf != NULL) {
        nvshmem_debug_printf("WARN: %s\n", message);
    } else {
        fprintf(fallback, "WARN: %s\n", message);
        fflush(fallback);
    }
    free(message);
}

#endif
