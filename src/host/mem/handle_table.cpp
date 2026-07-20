/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include "internal/host/nvshmemi_handle_table.hpp"

nvshmemi_mem_handle_registry_base::nvshmemi_mem_handle_registry_base(int num_transports)
    : num_transports_(num_transports) {}

void nvshmemi_mem_handle_registry_base::push_mem_handles(
    std::vector<nvshmem_mem_handle_t> handles) {
    mem_handles_.push_back(std::move(handles));
}

nvshmem_mem_handle_t *nvshmemi_mem_handle_registry_base::get_mem_handle(size_t handle_idx, int pe,
                                                                        int transport_idx) {
    assert(handle_idx < mem_handles_.size());
    size_t sub_idx = static_cast<size_t>(pe) * num_transports_ + transport_idx;
    assert(sub_idx < mem_handles_[handle_idx].size());
    return &mem_handles_[handle_idx][sub_idx];
}

void nvshmemi_dense_mem_handle_registry::append_index(size_t count, size_t handle_idx,
                                                      void *start_addr, size_t size) {
    index_.resize(index_.size() + count, {handle_idx, start_addr, size});
}

const nvshmemi_mem_handle_index_entry &nvshmemi_dense_mem_handle_registry::get_index(
    size_t addr_idx) const {
    return index_[addr_idx];
}

void nvshmemi_sparse_mem_handle_registry::set_index(size_t addr_idx, size_t handle_idx,
                                                    void *start_addr, size_t size) {
    index_[addr_idx] = {handle_idx, start_addr, size};
}

const nvshmemi_mem_handle_index_entry &nvshmemi_sparse_mem_handle_registry::get_index(
    size_t addr_idx) const {
    return index_.at(addr_idx);
}

void nvshmemi_sparse_mem_handle_registry::clear_index(size_t addr_idx) { index_.erase(addr_idx); }

nvshmemi_handle_table::nvshmemi_handle_table(void *heap_base, size_t heap_size,
                                             size_t log2_granularity, size_t granularity,
                                             int num_transports, int npes)
    : heap_base_(heap_base),
      heap_size_(heap_size),
      log2_granularity_(log2_granularity),
      granularity_(granularity),
      num_transports_(num_transports),
      npes_(npes),
      internal_(num_transports),
      mmap_(num_transports) {}

nvshmem_mem_handle_t *nvshmemi_handle_table::get_mem_handle(void *addr, size_t *len, int pe,
                                                            int transport_idx) {
    const uintptr_t addr_value = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t heap_base_value = reinterpret_cast<uintptr_t>(heap_base_);

    if (pe < 0 || pe >= npes_ || transport_idx < 0 || transport_idx >= num_transports_) {
        return nullptr;
    }

    if (addr_value < heap_base_value || addr_value - heap_base_value >= heap_size_) return nullptr;

    const size_t offset = static_cast<size_t>(addr_value - heap_base_value);
    const size_t addr_idx = offset >> log2_granularity_;

    const nvshmemi_mem_handle_index_entry *entry;
    nvshmemi_mem_handle_registry_base *reg;

    /* If the address is within the mmap range, use the mmap handles. */
    if (offset >= heap_size_ - mmap_allocated_range_) {
        if (!mmap_.has_index(addr_idx)) return nullptr;
        entry = &mmap_.get_index(addr_idx);
        reg = &mmap_;
    } else {
        if (!internal_.has_index(addr_idx)) return nullptr;
        entry = &internal_.get_index(addr_idx);
        reg = &internal_;
    }

    const uintptr_t entry_start = reinterpret_cast<uintptr_t>(entry->start_addr);
    if (entry_start > addr_value || addr_value - entry_start >= entry->size) return nullptr;
    const size_t entry_offset = static_cast<size_t>(addr_value - entry_start);

    /* Return the remaining space within the registered chunk. */
    if (len) {
        *len = entry->size - entry_offset;
    }
    return reg->get_mem_handle(entry->handle_idx, pe, transport_idx);
}

size_t nvshmemi_handle_table::get_addr_offset(void *addr) {
    const uintptr_t addr_value = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t heap_base_value = reinterpret_cast<uintptr_t>(heap_base_);
    assert(addr_value >= heap_base_value && addr_value - heap_base_value < heap_size_);
    const size_t offset = static_cast<size_t>(addr_value - heap_base_value);
    const size_t addr_idx = offset >> log2_granularity_;

    const nvshmemi_mem_handle_index_entry &entry = (offset >= heap_size_ - mmap_allocated_range_)
                                                       ? mmap_.get_index(addr_idx)
                                                       : internal_.get_index(addr_idx);

    const uintptr_t entry_start = reinterpret_cast<uintptr_t>(entry.start_addr);
    assert(addr_value >= entry_start && addr_value - entry_start < entry.size);
    return static_cast<size_t>(addr_value - entry_start);
}

void nvshmemi_handle_table::push_p2p_mem_handles(std::vector<nvshmem_mem_handle_t> handles) {
    p2p_mem_handles_.push_back(std::move(handles));
}
