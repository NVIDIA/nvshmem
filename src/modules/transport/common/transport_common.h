/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_COMMON_H
#define _TRANSPORT_COMMON_H

#define __STDC_FORMAT_MACROS 1

#include <errno.h>    // for errno
#include <stdio.h>    // for fprintf, stderr
#include <string.h>   // for memcpy, strerror
#include <strings.h>  // for strncasecmp
#include <cuda_fp16.h>
#include <unordered_map>
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_opt...
#include "internal/host_transport/transport.h"           // for nvshmem_tran...

#define MAXPATHSIZE 1024
#define MAX_TRANSPORT_EP_COUNT 1

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define TRANSPORT_LOG_NONE 0
#define TRANSPORT_LOG_VERSION 1
#define TRANSPORT_LOG_WARN 2
#define TRANSPORT_LOG_INFO 3
#define TRANSPORT_LOG_ABORT 4
#define TRANSPORT_LOG_TRACE 5

#if defined(NVSHMEM_x86_64)
#define MEM_BARRIER() asm volatile("mfence" ::: "memory")
#define STORE_BARRIER() asm volatile("sfence" ::: "memory")
#define LOAD_BARRIER() asm volatile("lfence" ::: "memory")
#elif defined(NVSHMEM_PPC64LE)
#define MEM_BARRIER() asm volatile("sync" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#elif defined(NVSHMEM_AARCH64)
#define MEM_BARRIER() asm volatile("dmb sy" ::: "memory")
#define STORE_BARRIER() asm volatile("dmb st" ::: "memory")
#define LOAD_BARRIER() MEM_BARRIER()
#else
#define MEM_BARRIER() asm volatile("" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#endif

#define INFO(LOG_LEVEL, fmt, ...)                                                  \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_INFO) {                                     \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

#define TRACE(LOG_LEVEL, fmt, ...)                                                 \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_TRACE) {                                    \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

static inline int nvshmemt_errno_from_status(int status) {
    if (status < 0) {
        return -status;
    }
    if (status > 0) {
        return status;
    }
    return 0;
}

static inline const char *nvshmemt_strerror_from_errno(int err) {
    return (err != 0) ? strerror(err) : "errno unavailable";
}

static inline const char *nvshmemt_strerror_from_status(int status) {
    return nvshmemt_strerror_from_errno(nvshmemt_errno_from_status(status));
}

#define NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, err, label, ...)                                \
    do {                                                                                    \
        if (unlikely((status) != 0)) {                                                      \
            fprintf(stderr, "%s:%d: non-zero status: %d (%s) ", __FILE__, __LINE__, status, \
                    nvshmemt_strerror_from_status(status));                                 \
            fprintf(stderr, __VA_ARGS__);                                                   \
            fprintf(stderr, "\n");                                                          \
            status = err;                                                                   \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define NVSHMEMT_ERRNO_NULL_ERROR_JMP(var, status, err, label, ...)                                \
    do {                                                                                           \
        if (unlikely((var) == NULL)) {                                                             \
            int saved_errno = errno;                                                               \
            fprintf(stderr, "%s:%d: NULL value (errno: %d, %s) ", __FILE__, __LINE__, saved_errno, \
                    nvshmemt_strerror_from_errno(saved_errno));                                    \
            fprintf(stderr, __VA_ARGS__);                                                          \
            fprintf(stderr, "\n");                                                                 \
            status = err;                                                                          \
            goto label;                                                                            \
        }                                                                                          \
    } while (0)

#define NVSHMEMT_ERRNO_NZ_ERROR_RET(status, err, ...)                                       \
    do {                                                                                    \
        if (unlikely((status) != 0)) {                                                      \
            fprintf(stderr, "%s:%d: non-zero status: %d (%s) ", __FILE__, __LINE__, status, \
                    nvshmemt_strerror_from_status(status));                                 \
            fprintf(stderr, __VA_ARGS__);                                                   \
            fprintf(stderr, "\n");                                                          \
            status = err;                                                                   \
            return status;                                                                  \
        }                                                                                   \
    } while (0)

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) \
    do {                                                   \
        void **cast = (void **)&funcptr;                   \
        void *tmp = dlvsym(handle, symbol, version);       \
        *cast = tmp;                                       \
    } while (0)

static inline int nvshmemt_common_get_log_level(struct nvshmemi_options_s *options) {
    if (!options->DEBUG_provided && !options->DEBUG_SUBSYS_provided) {
        return TRANSPORT_LOG_NONE;
    } else if (strncasecmp(options->DEBUG, "VERSION", 8) == 0) {
        return TRANSPORT_LOG_VERSION;
    } else if (strncasecmp(options->DEBUG, "WARN", 5) == 0) {
        return TRANSPORT_LOG_WARN;
    } else if (strncasecmp(options->DEBUG, "INFO", 5) == 0) {
        return TRANSPORT_LOG_INFO;
    } else if (strncasecmp(options->DEBUG, "ABORT", 6) == 0) {
        return TRANSPORT_LOG_ABORT;
    } else if (strncasecmp(options->DEBUG, "TRACE", 6) == 0) {
        return TRANSPORT_LOG_TRACE;
    }

    return TRANSPORT_LOG_INFO;
}

struct transport_mem_handle_info_cache;  // IWYU pragma: keep

int nvshmemt_put_signal(struct nvshmem_transport *tcurr, int pe, rma_verb_t write_verb,
                        std::vector<rma_memdesc_t> &write_remote,
                        std::vector<rma_memdesc_t> &write_local,
                        std::vector<rma_bytesdesc_t> &write_bytesdesc, amo_verb_t sig_verb,
                        amo_memdesc_t *sig_target, amo_bytesdesc_t sig_bytesdesc, int is_proxy);

struct nvshmemt_hca_info {
    char name[64];
    int port;
    int count;
    int found;
};

typedef int (*pci_path_cb)(int dev, char **pcipath, struct nvshmem_transport *transport);

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level);

int nvshmemt_mem_handle_cache_init(nvshmem_transport_t t,
                                   struct transport_mem_handle_info_cache **cache);
int nvshmemt_mem_handle_cache_add(nvshmem_transport_t t,
                                  struct transport_mem_handle_info_cache *cache, void *addr,
                                  void *mem_handle_info);
void *nvshmemt_mem_handle_cache_get(nvshmem_transport_t t,
                                    struct transport_mem_handle_info_cache *cache, void *addr);
void *nvshmemt_mem_handle_cache_get_by_idx(struct transport_mem_handle_info_cache *cache,
                                           size_t idx);
size_t nvshmemt_mem_handle_cache_get_size(struct transport_mem_handle_info_cache *cache);
int nvshmemt_mem_handle_cache_remove(nvshmem_transport_t t,
                                     struct transport_mem_handle_info_cache *cache, void *addr);
int nvshmemt_mem_handle_cache_fini(struct transport_mem_handle_info_cache *cache);

bool check_egm(void *addr, std::unordered_map<void *, size_t> *egm_map);

/* C++11-safe helpers: reinterpret integer bits as half/float/double, add, return result bits.
 * Used by transport AMO handlers to implement software float atomic add.
 * Primary template is unreachable (only uint16_t, uint32_t and uint64_t are valid). */
template <typename T>
static inline T nvshmemt_float_atomic_add(T /*old_bits*/, uint64_t /*add_bits*/) {
    return T{};
}
template <>
inline uint16_t nvshmemt_float_atomic_add<uint16_t>(uint16_t old_bits, uint64_t add_bits) {
    uint16_t add_trunc = static_cast<uint16_t>(add_bits);
    __half_raw h_old_raw = {old_bits};
    __half_raw h_add_raw = {add_trunc};
    __half h_new = __hadd(__half(h_old_raw), __half(h_add_raw));
    return static_cast<__half_raw>(h_new).x;
}
template <>
inline uint32_t nvshmemt_float_atomic_add<uint32_t>(uint32_t old_bits, uint64_t add_bits) {
    float f_old, f_add, f_new;
    uint32_t add_trunc = static_cast<uint32_t>(add_bits);
    memcpy(&f_old, &old_bits, sizeof(float));
    memcpy(&f_add, &add_trunc, sizeof(float));
    f_new = f_old + f_add;
    uint32_t result;
    memcpy(&result, &f_new, sizeof(float));
    return result;
}
template <>
inline uint64_t nvshmemt_float_atomic_add<uint64_t>(uint64_t old_bits, uint64_t add_bits) {
    double d_old, d_add, d_new;
    memcpy(&d_old, &old_bits, sizeof(double));
    memcpy(&d_add, &add_bits, sizeof(double));
    d_new = d_old + d_add;
    uint64_t result;
    memcpy(&result, &d_new, sizeof(double));
    return result;
}

extern "C" {
int nvshmemt_init(nvshmem_transport_t *transport, struct nvshmemi_cuda_fn_table *table,
                  int api_version);
}

#endif
