/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_HEAP_REGISTRATION_HPP
#define NVSHMEMI_HEAP_REGISTRATION_HPP

#include <cstddef>
#include <optional>
#include <sys/types.h>
#include <vector>
#include "internal/host/nvshmemi_transport_view.hpp"
#include "internal/host/nvshmemi_types.h"

class nvshmemi_handle_table;
class nvshmemi_mem_p2p_transport;
class nvshmemi_mem_remote_transport;
enum class nvshmemi_allocation_kind { INTERNAL, EXTERNAL };

enum class nvshmemi_heap_registration_policy {
    DYNAMIC_VMM,
    STATIC_VIDMEM_FULL_HEAP,
    STATIC_SYSMEM_FULL_HEAP
};

struct nvshmemi_heap_registration_geometry {
    void *heap_base;
    void *global_heap_base;
    size_t logical_heap_size;
    size_t mem_granularity;
    size_t log2_mem_granularity;
};

struct nvshmemi_heap_registration_config {
    nvshmemi_heap_registration_config(
        int mype, int npes, int npes_node, nvshmemi_heap_registration_geometry geometry,
        CUmemAllocationHandleType effective_handle_type, nvshmemi_transport_view transports,
        nvshmemi_mem_remote_transport &remote_transport, nvshmemi_mem_p2p_transport &p2p_transport,
        std::vector<void *> &peer_heap_base_p2p, nvshmemi_heap_registration_policy policy) noexcept
        : mype(mype),
          npes(npes),
          npes_node(npes_node),
          geometry(geometry),
          effective_handle_type(effective_handle_type),
          transports(transports),
          remote_transport(remote_transport),
          p2p_transport(p2p_transport),
          peer_heap_base_p2p(peer_heap_base_p2p),
          policy(policy) {}

    int mype;
    int npes;
    int npes_node;
    nvshmemi_heap_registration_geometry geometry;
    CUmemAllocationHandleType effective_handle_type;
    nvshmemi_transport_view transports;
    nvshmemi_mem_remote_transport &remote_transport;
    nvshmemi_mem_p2p_transport &p2p_transport;
    std::vector<void *> &peer_heap_base_p2p;
    nvshmemi_heap_registration_policy policy;
};

/** Mandatory transport registration backend for the symmetric heap. */
class nvshmemi_heap_registration {
   public:
    nvshmemi_heap_registration(nvshmemi_handle_table *table,
                               nvshmemi_heap_registration_config cfg) noexcept;

    int setup();
    int teardown();
    /** Register a VMM chunk; allocation kind distinguishes internal and user-provided memory. */
    int register_vmm_chunk(CUmemGenericAllocationHandle handle, off_t mc_offset, size_t size,
                           nvshmemi_allocation_kind alloc_kind,
                           std::optional<size_t> mmap_allocated_range);
    /**
     * Release published remote handles and lookup entries for a user-provided VMM range.
     * Chunks whose registration failed before publication are ignored.
     */
    int unregister_vmm_chunk(off_t mc_offset, size_t size);

    int npes() const { return npes_; }
    const nvshmemi_transport_view &transports() const { return transports_; }

   private:
    nvshmemi_mem_remote_transport &remote_transport();
    nvshmemi_mem_p2p_transport &p2p_transport();

    int map_dynamic_p2p_chunk(CUmemGenericAllocationHandle handle, void *buf, size_t size);
    int map_posix_p2p_chunk(CUmemGenericAllocationHandle handle, void *buf, size_t size);
    int map_fabric_p2p_chunk(CUmemGenericAllocationHandle handle, void *buf, size_t size);
    int map_static_vidmem_p2p_chunk(void *buf);
    int map_static_sysmem_p2p_chunk();
    int map_fabric_p2p_memory(int pe_id, CUmemFabricHandle handle, void *buf, size_t size);
    int map_imported_p2p_memory(int pe_id, CUmemGenericAllocationHandle handle, void *buf,
                                size_t size);
    int map_static_vidmem_p2p_memory(int pe_id, const cudaIpcMemHandle_t &handle);
    int map_static_sysmem_p2p_memory(int pe_id);
    /** Register a buffer range for non-mapping transports and gather its handles. */
    int register_remote_chunk(void *buf, size_t size, nvshmemi_allocation_kind alloc_kind);
    /** Update address-to-handle lookup after registration. */
    void update_handle_index(void *buf, size_t size, nvshmemi_allocation_kind alloc_kind);
    /** Plan peer virtual addresses and initialize the PE-to-process map. */
    int plan_vmm_peer_bases();
    /** Register the statically allocated heap for P2P and remote transports. */
    int register_static_heap();
    bool has_p2p_mapping() const;
    bool is_node_local_pe(int pe_id) const;
    int node_local_index(int pe_id) const;

    nvshmemi_handle_table *table_;
    nvshmemi_heap_registration_policy policy_;
    nvshmemi_mem_remote_transport &remote_transport_;
    nvshmemi_mem_p2p_transport &p2p_transport_;
    std::vector<void *> &peer_heap_base_p2p_;
    nvshmemi_heap_registration_geometry geometry_;
    CUmemAllocationHandleType effective_handle_type_;

    nvshmemi_transport_view transports_;

    /* PE topology */
    int mype_;
    int npes_;
    int npes_node_;

    bool gather_mem_handles_done_ = false;
};

#endif /* NVSHMEMI_HEAP_REGISTRATION_HPP */
