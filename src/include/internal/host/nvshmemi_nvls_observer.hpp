/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_NVLS_OBSERVER_HPP
#define NVSHMEMI_NVLS_OBSERVER_HPP

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <vector>
#include "internal/host/nvshmemi_heap_observer.hpp"
#include "internal/host/nvshmemi_types.h"
#include "device_host/nvshmem_types.h"

class nvshmemi_symmetric_heap_vidmem_dynamic_vmm;
struct nvshmemi_team_dec;

/** Heap observer that owns NVLS multicast bind/unbind operations. */
class nvshmemi_nvls_observer : public nvshmemi_heap_observer {
   public:
    explicit nvshmemi_nvls_observer(nvshmemi_symmetric_heap_vidmem_dynamic_vmm *heap,
                                    bool is_platform_nvls)
        : heap_(heap), is_platform_nvls_(is_platform_nvls) {}
    ~nvshmemi_nvls_observer() = default;

    int on_chunk_mapped(nvshmem_mem_handle_t *handle, off_t mc_offset, off_t mmap_offset,
                        size_t size) override;
    int on_chunk_unmapped(off_t mc_offset, size_t size) override;
    int on_heap_teardown() override { return 0; }

    /* Per-team multicast groups. */
    int nvls_create_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_bind_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_map_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unmap_heap_memory_by_size(nvshmemi_team_t *team, off_t mc_offset, uint64_t size);
    void nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unmap_heap_memory(off_t mc_offset, uint64_t size);
    void nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unbind_heap_memory_by_size(off_t mc_offset, size_t size);
    int nvls_broadcast_heap_handle_by_team(char *buffer, size_t length, nvshmemi_team_t *team);

   private:
    int nvls_broadcast_heap_handle_ipc(char *shareable_handle, int root, nvshmemi_team_t *team);
    int nvls_broadcast_heap_handle_fabric(char *buffer, size_t length, int root,
                                          nvshmemi_team_t *team);
    int nvls_create_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size);
    int nvls_bind_heap_memory_by_size(nvshmemi_team_t *team, nvshmem_mem_handle_t *mem_handle,
                                      off_t mc_offset, off_t mmap_offset, size_t mmap_size);
    int nvls_unbind_heap_memory_by_size(nvshmemi_team_t *team, off_t mc_offset, size_t size);
    int nvls_map_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size, off_t mmap_offset,
                                     off_t mc_offset);
    int nvls_create_heap_memory(uint64_t mem_size);
    int nvls_bind_heap_memory(nvshmem_mem_handle_t *mem_handle, off_t mc_offset, off_t mmap_offset,
                              size_t mmap_size);
    int nvls_map_heap_memory(uint64_t mem_size, off_t mmap_offset, off_t mc_offset);

    struct mapped_chunk {
        CUmemGenericAllocationHandle handle;
        off_t mc_offset;
        off_t mmap_offset;
        size_t mmap_size;
    };

    nvshmemi_symmetric_heap_vidmem_dynamic_vmm *heap_;
    bool is_platform_nvls_;
    std::vector<mapped_chunk> mapped_chunks_;
};

#endif /* NVSHMEMI_NVLS_OBSERVER_HPP */
