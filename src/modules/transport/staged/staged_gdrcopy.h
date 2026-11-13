/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_STAGED_GDRCOPY_H
#define NVSHMEM_STAGED_GDRCOPY_H

#include <stddef.h>

#include "staged_config.h"

namespace nvshmemi {

/* GDRCOPY copy policy: the CPU copies to and from GPU memory through CPU mappings of it. Mapped GPU
 * memory (registered user buffers and a VIDMEM bounce) is tracked in s->memory.mem_handle_infos so
 * copies find it by address. Without GDRCopy support, every function fails. */

int staged_gdrcopy_init(transport_staged_state_t* s, struct nvshmemi_cuda_fn_table* cuda_syms);

/* Unmaps everything still mapped and shuts down the mapping backend. */
int staged_gdrcopy_fini(transport_staged_state_t* s);

/* mapping_buf is the address to pin and may alias buf (same physical memory, different VA). */
int staged_gdrcopy_map(transport_staged_state_t* s, void* buf, size_t length, void* mapping_buf,
                       staged_mem_handle_info_t** info_out);

int staged_gdrcopy_unmap(transport_staged_state_t* s, staged_mem_handle_info_t* info);

int staged_gdrcopy_copy(transport_staged_state_t* s, void* dst, bool dst_on_gpu, const void* src,
                        bool src_on_gpu, size_t bytes, const char* label);

}  // namespace nvshmemi

#endif /* NVSHMEM_STAGED_GDRCOPY_H */
