/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_HEAP_OBSERVER_HPP
#define NVSHMEMI_HEAP_OBSERVER_HPP

#include <cstddef>
#include <sys/types.h>
#include "internal/host_transport/nvshmemi_transport_defines.h"

/** Observer interface for symmetric heap lifecycle events. */
class nvshmemi_heap_observer {
   public:
    virtual ~nvshmemi_heap_observer() = default;

    /** Called after a chunk is mapped and before transport registration. */
    virtual int on_chunk_mapped(nvshmem_mem_handle_t *handle, off_t mc_offset, off_t mmap_offset,
                                size_t size) = 0;

    /** Called before a chunk is unmapped. */
    virtual int on_chunk_unmapped(off_t mc_offset, size_t size) = 0;

    /** Called before heap teardown releases mappings or handles. */
    virtual int on_heap_teardown() = 0;
};

#endif /* NVSHMEMI_HEAP_OBSERVER_HPP */
