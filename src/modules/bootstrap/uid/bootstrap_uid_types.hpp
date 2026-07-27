/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BOOTSTRAP_UID_HPP
#define BOOTSTRAP_UID_HPP

#include <pthread.h>  // for PTHREAD_MUTEX_INITIALIZER
#include <stdint.h>   // for uint64_t, uint32_t
#include <stdio.h>    // for fclose, fopen, fread
#include <stdlib.h>   // for malloc
#include <algorithm>  // for fill
#include <limits.h>   // for SIZE_MAX
#include <cstring>    // for NULL, size_t
#include <list>       // for list
#include <type_traits>
#include <vector>                            // for vector
#include "bootstrap_uid_remap.h"             // for bootstrap_uid_socket_a...
#include "bootstrap_util.h"                  // for BOOTSTRAP_ERROR_PRINT
#include "ncclSocket/ncclsocket_socket.hpp"  // for MAX_IF_NAME_SIZE

template <typename T>
inline bootstrap_result_t bootstrap_calloc_debug(T** ptr, size_t nelem, const char* filefunc,
                                                 int line) {
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>,
                  "BOOTSTRAP_CALLOC requires POD storage");

    if (ptr == NULL) {
        BOOTSTRAP_ERROR_PRINT("%s:%d allocation called with null output pointer", filefunc, line);
        return BOOTSTRAP_INVALID_ARGUMENT;
    }

    *ptr = NULL;
    if (nelem > 0 && nelem > SIZE_MAX / sizeof(T)) {
        BOOTSTRAP_ERROR_PRINT("%s:%d allocation size overflow: %zu elements of %zu bytes", filefunc,
                              line, nelem, sizeof(T));
        return BOOTSTRAP_INTERNAL_ERROR;
    }

    const size_t allocation_size = nelem * sizeof(T);
    if (allocation_size == 0) {
        return BOOTSTRAP_SUCCESS;
    }

    void* p = malloc(allocation_size);
    if (p == NULL) {
        BOOTSTRAP_ERROR_PRINT("%s:%d unable to malloc %zu bytes", filefunc, line, allocation_size);
        return BOOTSTRAP_INTERNAL_ERROR;
    }
    // BOOTSTRAP_DEBUG_PRINT("%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T),
    // p);
    unsigned char* begin = static_cast<unsigned char*>(p);
    std::fill_n(begin, allocation_size, 0);
    *ptr = (T*)p;
    return BOOTSTRAP_SUCCESS;
}

#define BOOTSTRAP_CALLOC(...) bootstrap_calloc_debug(__VA_ARGS__, __FILE__, __LINE__)

/* Socket Root thread fn arguments */
struct bootstrap_root_args {
    bootstrap_uid_socket_t* listen_sock;
    uint64_t magic;
    int version;
    int nranks;
};

/* Socket External PEs address connection info */
struct bootstrap_ext_info {
    int rank;
    int nranks;
    bootstrap_uid_socket_address_t ext_address_listen_root;
    bootstrap_uid_socket_address_t ext_address_listen;
};

/* Internal UID implementation */
struct bootstrap_uid_handle {
    int version;
    bootstrap_uid_socket_address_t addr;
    uint64_t magic;
};

/* Socket Net state: name, addr, status, lock */
struct bootstrap_netstate {
    char bootstrap_netifname[MAX_IF_NAME_SIZE + 1];     /* Socket Interface Name */
    bootstrap_uid_socket_address_t bootstrap_netifaddr; /* Socket Interface Address */
    int bootstrap_netinitdone = 0;                      /* Socket Interface Init Status */
    pthread_mutex_t bootstrap_netlock = PTHREAD_MUTEX_INITIALIZER; /* Socket Interface Lock */
    pthread_t bootstrap_root; /* Socket Root Thread for phoning root to non-root peers */
};

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

class unexpected_connection {
   public:
    unexpected_connection(int peer, int tag, bootstrap_uid_socket_t& socket);
    ~unexpected_connection();

    unexpected_connection(const unexpected_connection&) = delete;
    unexpected_connection& operator=(const unexpected_connection&) = delete;
    unexpected_connection(unexpected_connection&&) = delete;
    unexpected_connection& operator=(unexpected_connection&&) = delete;

    int peer() const { return peer_; }
    int tag() const { return tag_; }
    bootstrap_uid_socket_t release();

   private:
    int peer_;
    int tag_;
    bootstrap_uid_socket_t socket_;
};

struct bootstrap_state {
    bootstrap_uid_socket_t listen_sock;
    bootstrap_uid_socket_t ring_recv_socket;
    bootstrap_uid_socket_t ring_send_socket;
    std::vector<bootstrap_uid_socket_address_t> peer_comm_addresses;
    std::list<unexpected_connection> unexpected_connections;
    int rank;
    int nranks;
    uint64_t magic;
    volatile uint32_t* abort_flag;
};

/* get any bytes of random data from /dev/urandom, return 0 if it succeeds; else
 * return -1 */
inline int bootstrap_get_random_data(void* buffer, size_t bytes) {
    int ret = 0;
    if (bytes > 0) {
        const size_t one = 1UL;
        FILE* fp = fopen("/dev/urandom", "r");
        if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) {
            ret = -1;
        }

        if (fp) {
            fclose(fp);
        }
    }

    return ret;
}

#endif /* BOOTSTRAP_UID_HPP */
