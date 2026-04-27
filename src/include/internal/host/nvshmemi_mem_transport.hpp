/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_MEM_TRANSPORT_HPP
#define NVSHMEMI_MEM_TRANSPORT_HPP

#include <cstdlib>
#include <climits>
#include <memory>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <vector>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvmlwrap.h"

static_assert(sizeof(nvshmem_mem_handle_t) % sizeof(uint64_t) == 0,
              "nvshmem_mem_handle_t size is not a multiple of 8B");

/**
 * This is a singleton class managing memory kind specific business logic for p2p transport
 */
class nvshmemi_mem_p2p_transport final {
   public:
    ~nvshmemi_mem_p2p_transport();
    nvshmemi_mem_p2p_transport(const nvshmemi_mem_p2p_transport &obj) = delete;
    static nvshmemi_mem_p2p_transport *get_instance(int mype, int npes) {
        if (p2p_objref_ == nullptr) {
            p2p_objref_ = new nvshmemi_mem_p2p_transport(mype, npes);
            return p2p_objref_;
        } else {
            return p2p_objref_;
        }
    }

    static void destroy_instance(void) {
        if (p2p_objref_ != nullptr) {
            delete p2p_objref_;
            p2p_objref_ = nullptr;
        }
    }

    void print_mem_handle(int pe_id, int transport_idx, nvshmemi_symmetric_heap &obj);

    struct nvml_function_table *get_nvml_ftable(void) { return &nvml_ftable_; }
    CUmemAllocationHandleType get_mem_handle_type(void) const { return nvshmemi_mem_handle_type_; }
    bool is_mnnvl_fabric(void) const { return nvshmemi_has_mnnvl_fabric_; }
    bool is_initialized(void) const { return !errored_on_initialization_; }
    int create_proc_map(nvshmemi_symmetric_heap &obj);
    std::map<pid_t, int> get_proc_map(void) const { return proc_map_; }
    int get_num_p2p_connected_pes(nvshmemi_symmetric_heap &obj);
    bool is_nvl_connected_pe(int pe) {
        /* Check if the peer GPU is connected via the MNNVL fabric */
        return nvshmemi_nvl_connected_pes_.at(pe) != 0;
    }
    bool is_nvls_connected_pe(int pe) const noexcept {
        /* Check if the peer GPU is connected via the MNNVL fabric
         * and within same multicast domain
         */
        return nvshmemi_nvls_connected_pes_.at(pe) != 0;
    }
    bool is_handle_accessible_pe(int pe) {
        /* Check if the peer GPU is accessible with handles */
        return nvshmemi_handle_accessible_pes_.at(pe) != 0;
    }

    // This function allows to p2p connected PE list is all P2P PEs could not be mapped to VA
    void update_p2p_connected_pes(const std::vector<uint8_t> &updated_connected_pes) {
        nvshmemi_nvl_connected_pes_.clear();
        nvshmemi_nvl_connected_pes_ = updated_connected_pes;
    }

    size_t get_nvls_connected_pes_count(void) const {
        return std::count(nvshmemi_nvls_connected_pes_.begin(),
                          nvshmemi_nvls_connected_pes_.end(), uint8_t{1});
    }
    const std::vector<uint8_t> &get_nvls_connected_pes(void) const {
        return nvshmemi_nvls_connected_pes_;
    }

   private:
    explicit nvshmemi_mem_p2p_transport(int mype, int npes);
    static void *nvml_handle_;
    static struct nvml_function_table nvml_ftable_;
    static nvshmemi_mem_p2p_transport *p2p_objref_;  // singleton instance
    std::map<pid_t, int> proc_map_;

    // Static wrapper function for atexit that matches the expected signature
    static void nvshmemi_nvml_ftable_fini_wrapper(void);

    bool nvshmemi_has_mnnvl_fabric_ = false;
    std::vector<uint8_t> nvshmemi_nvl_connected_pes_;

    // list of PEs that can be accessed with handles
    std::vector<uint8_t> nvshmemi_handle_accessible_pes_;

    // this is a bitmap to track the PEs that are within the same multicast domain
    std::vector<uint8_t> nvshmemi_nvls_connected_pes_;
    bool errored_on_initialization_ = true;
    CUmemAllocationHandleType nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
};

/**
 * This is a singleton class managing memory kind specific business logic for remote transport
 */
class nvshmemi_mem_remote_transport final {
   public:
    ~nvshmemi_mem_remote_transport() {
        if (remote_objref_ != nullptr) remote_objref_ = nullptr;
    }
    nvshmemi_mem_remote_transport(const nvshmemi_mem_remote_transport &obj) = delete;
    nvshmemi_mem_remote_transport(nvshmemi_mem_remote_transport &&obj) = delete;
    static nvshmemi_mem_remote_transport *get_instance(void) noexcept {
        if (remote_objref_ == nullptr) {
            remote_objref_ = new nvshmemi_mem_remote_transport();
            return remote_objref_;
        } else {
            return remote_objref_;
        }
    }

    static void destroy_instance(void) {
        if (remote_objref_ != nullptr) {
            delete remote_objref_;
            remote_objref_ = nullptr;
        }
    }

    int gather_mem_handles(nvshmemi_symmetric_heap &obj, uint64_t heap_offset, size_t size,
                           bool ext_allocation = false);
    /* On-demand registration and release of memory */
    int register_mem_handle(nvshmem_mem_handle_t *local_handles, int transport_idx, void *buf,
                            size_t size, nvshmem_transport_t current);
    int release_mem_handles(nvshmem_mem_handle_t *handles, nvshmemi_symmetric_heap &obj);

    int is_mem_handle_null(nvshmem_mem_handle_t *handle) {
        NVSHMEMU_FOR_EACH(i, (sizeof(nvshmem_mem_handle_t) / sizeof(uint64_t))) {
            if (*((uint64_t *)handle + i) != (uint64_t)0) return 0;
        }

        return 1;
    }

   private:
    explicit nvshmemi_mem_remote_transport(void) noexcept {};
    static nvshmemi_mem_remote_transport *remote_objref_;  // singleton instance
};

#endif /* MEM_TRANSPORT_HPP */
