/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_HANDLE_TABLE_HPP
#define NVSHMEMI_HANDLE_TABLE_HPP

#include <cassert>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "internal/host_transport/nvshmemi_transport_defines.h"

/** Maps an address granule to its handle set. */
struct nvshmemi_mem_handle_index_entry {
    size_t handle_idx; /* which element in mem_handles_ covers this address */
    void *start_addr;  /* base address of the registered chunk */
    size_t size;       /* size of the registered chunk */
};

class nvshmemi_mem_handle_registry_base {
   public:
    explicit nvshmemi_mem_handle_registry_base(int num_transports);
    virtual ~nvshmemi_mem_handle_registry_base() = default;

    void push_mem_handles(std::vector<nvshmem_mem_handle_t> handles);
    nvshmem_mem_handle_t *get_mem_handle(size_t handle_idx, int pe, int transport_idx);
    size_t num_handle_sets() const { return mem_handles_.size(); }

   protected:
    int num_transports_;
    std::vector<std::vector<nvshmem_mem_handle_t>> mem_handles_;
};

/** Dense address-granule index for internal heap allocations. */
class nvshmemi_dense_mem_handle_registry : public nvshmemi_mem_handle_registry_base {
   public:
    using nvshmemi_mem_handle_registry_base::nvshmemi_mem_handle_registry_base;

    void append_index(size_t count, size_t handle_idx, void *start_addr, size_t size);
    const nvshmemi_mem_handle_index_entry &get_index(size_t addr_idx) const;
    size_t index_size() const { return index_.size(); }
    bool has_index(size_t addr_idx) const { return addr_idx < index_.size(); }

   private:
    std::vector<nvshmemi_mem_handle_index_entry> index_;
};

/** Sparse address-granule index for user-provided mmap allocations. */
class nvshmemi_sparse_mem_handle_registry : public nvshmemi_mem_handle_registry_base {
   public:
    using nvshmemi_mem_handle_registry_base::nvshmemi_mem_handle_registry_base;

    void set_index(size_t addr_idx, size_t handle_idx, void *start_addr, size_t size);
    const nvshmemi_mem_handle_index_entry &get_index(size_t addr_idx) const;
    void clear_index(size_t addr_idx);
    bool has_index(size_t addr_idx) const { return index_.find(addr_idx) != index_.end(); }

   private:
    std::unordered_map<size_t, nvshmemi_mem_handle_index_entry> index_;
};

/** Handle table for proxy lookups. */
class nvshmemi_handle_table {
   public:
    nvshmemi_handle_table(void *heap_base, size_t heap_size, size_t log2_granularity,
                          size_t granularity, int num_transports, int npes);

    /**
     * Given an address, size, PE index, and transport index, retrieve the corresponding transport
     * handle.
     */
    nvshmem_mem_handle_t *get_mem_handle(void *addr, size_t *len, int pe, int transport_idx);

    /** Return the address offset within its registered handle range. */
    size_t get_addr_offset(void *addr);

    bool empty_handle_cache() const { return handle_cache_ == 0; }
    void inc_handle_cache() { handle_cache_++; }

    void set_mmap_allocated_range(size_t range) { mmap_allocated_range_ = range; }

    /* Registration */
    nvshmemi_dense_mem_handle_registry &internal_reg() { return internal_; }
    nvshmemi_sparse_mem_handle_registry &mmap_reg() { return mmap_; }

    void push_p2p_mem_handles(std::vector<nvshmem_mem_handle_t> handles);
    const nvshmem_mem_handle_t &get_p2p_mem_handle(size_t set_idx, int pe,
                                                   int transport_idx) const {
        assert(set_idx < p2p_mem_handles_.size());
        size_t sub_idx = static_cast<size_t>(pe) * num_transports_ + transport_idx;
        assert(sub_idx < p2p_mem_handles_[set_idx].size());
        return p2p_mem_handles_[set_idx][sub_idx];
    }
    size_t num_p2p_handle_sets() const { return p2p_mem_handles_.size(); }

   private:
    /* Heap geometry */
    void *heap_base_;
    size_t heap_size_;
    size_t log2_granularity_;
    size_t granularity_;
    int num_transports_;
    int npes_;

    size_t mmap_allocated_range_ = 0;
    int handle_cache_ = 0;

    /* Internal allocations are indexed in heap order. */
    nvshmemi_dense_mem_handle_registry internal_;
    /* mmap allocations are indexed by address. */
    nvshmemi_sparse_mem_handle_registry mmap_;

    /* P2P handles */
    std::vector<std::vector<nvshmem_mem_handle_t>> p2p_mem_handles_;
};

#endif /* NVSHMEMI_HANDLE_TABLE_HPP */
