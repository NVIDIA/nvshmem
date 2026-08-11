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
#include <array>
#include <stdint.h>
#include <vector>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/nvmlwrap.h"

class nvshmemi_transport_view;

enum class nvshmemi_clique_discovery_mode : uint8_t { LEGACY_NVML, CUDA_CLIQUE_API };
inline constexpr size_t nvshmemi_num_cuda_clique_types = 4;

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

    void print_mem_handle(nvshmem_mem_handle_t *handle, int mype);

    struct nvml_function_table *get_nvml_ftable(void) { return &nvml_ftable_; }
    CUmemAllocationHandleType get_mem_handle_type(void) const { return nvshmemi_mem_handle_type_; }
    bool is_mnnvl_fabric(void) const { return nvshmemi_has_mnnvl_fabric_; }
    bool is_initialized(void) const { return !errored_on_initialization_; }
    bool has_cuda_clique_info(void) const {
        return clique_discovery_mode_ == nvshmemi_clique_discovery_mode::CUDA_CLIQUE_API;
    }
    const std::vector<uint8_t> &get_uc_ptr_connected_pes(void) const {
        return cuda_clique_connected_pes_.at(static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_POINTER));
    }
    int create_proc_map(int npes, const nvshmemi_transport_view &transports);
    std::map<pid_t, int> get_proc_map(void) const { return proc_map_; }
    int get_num_uc_ptr_connected_pes(int npes_node);
    bool is_uc_ptr_connected_pe(int pe) const {
        /* Check if the peer GPU is reachable through a unicast pointer. */
        return cuda_clique_connected_pes_.at(static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_POINTER))
                   .at(pe) != 0;
    }
    bool is_mc_ptr_connected_pe(int pe) const noexcept {
        /* Check if the peer GPU is in the multicast-pointer domain. */
        return cuda_clique_connected_pes_.at(static_cast<size_t>(CU_CLIQUE_TYPE_MULTICAST_POINTER))
                   .at(pe) != 0;
    }
    bool is_uc_le_connected_pe(int pe) const {
        /* Check if the peer GPU is reachable through a unicast logical endpoint. */
        return cuda_clique_connected_pes_
                   .at(static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT))
                   .at(pe) != 0;
    }
    bool is_mc_le_connected_pe(int pe) const noexcept {
        return cuda_clique_connected_pes_
                   .at(static_cast<size_t>(CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT))
                   .at(pe) != 0;
    }

    // This function allows to modify the P2P-connected PE list if not all P2P PEs could be mapped
    // to VA
    void update_uc_ptr_connected_pes(const std::vector<uint8_t> &updated_connected_pes) {
        cuda_clique_connected_pes_.at(static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_POINTER)) =
            updated_connected_pes;
    }

    const std::vector<uint8_t> &get_mc_ptr_connected_pes(void) const {
        return cuda_clique_connected_pes_.at(static_cast<size_t>(CU_CLIQUE_TYPE_MULTICAST_POINTER));
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
    nvshmemi_clique_discovery_mode clique_discovery_mode_ =
        nvshmemi_clique_discovery_mode::LEGACY_NVML;
    // Connectivity bitmaps indexed by CUcliqueType.
    std::array<std::vector<uint8_t>, nvshmemi_num_cuda_clique_types> cuda_clique_connected_pes_;
    bool errored_on_initialization_ = true;
    CUmemAllocationHandleType nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
};

/**
 * This is a singleton class managing memory kind specific business logic for remote transport
 */
class nvshmemi_mem_remote_transport final {
   public:
    ~nvshmemi_mem_remote_transport() {
        if (remote_objref_ != nullptr) {
            remote_objref_ = nullptr;
        }
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

    int gather_mem_handles(const nvshmemi_transport_view &transports,
                           nvshmem_mem_handle_t *handle_data, uint64_t heap_offset, size_t size);
    /* On-demand registration and release of memory */
    int register_mem_handle(nvshmem_mem_handle_t *local_handles, int transport_idx, void *buf,
                            size_t size, const nvshmemi_transport_view &transports);
    int release_mem_handles(nvshmem_mem_handle_t *handles,
                            const nvshmemi_transport_view &transports);

    int is_mem_handle_null(nvshmem_mem_handle_t *handle) {
        NVSHMEMU_FOR_EACH(i, (sizeof(nvshmem_mem_handle_t) / sizeof(uint64_t))) {
            if (*((uint64_t *)handle + i) != (uint64_t)0) {
                return 0;
            }
        }

        return 1;
    }

   private:
    explicit nvshmemi_mem_remote_transport(void) noexcept {};
    static nvshmemi_mem_remote_transport *remote_objref_;  // singleton instance
};

#endif /* MEM_TRANSPORT_HPP */
