/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_GDR_INTERNAL_H
#define _TRANSPORT_GDR_INTERNAL_H

#include "transport_gdr_common.h"

struct nvshmemi_cuda_fn_table;

struct nvshmemt_gpu_cpu_backend_info {
    uintptr_t gpu_va;
    size_t mapped_size;
    nvshmemt_gpu_cpu_mapping_type type;
};

struct nvshmemt_gpu_cpu_backend_ops {
    int (*close)(void *context);
    int (*pin)(void *context, uintptr_t gpu_va, size_t size, uint32_t flags, uintptr_t *handle);
    int (*unpin)(void *context, uintptr_t handle);
    int (*get_info)(void *context, uintptr_t handle, nvshmemt_gpu_cpu_backend_info *info);
    int (*map)(void *context, uintptr_t handle, void **cpu_ptr_base, size_t size, uint32_t flags);
    int (*unmap)(void *context, uintptr_t handle, void *cpu_ptr_base, size_t size);
    int (*copy_to)(void *context, uintptr_t handle, void *map_d_ptr, const void *h_ptr,
                   size_t size);
    int (*copy_from)(void *context, uintptr_t handle, void *h_ptr, const void *map_d_ptr,
                     size_t size);
    bool (*has_capability)(void *context, nvshmemt_gpu_cpu_mapping_capability capability);
};

struct nvshmemt_gpu_cpu_backend {
    const char *name;
    void *context;
    void *library_handle;
    nvshmemt_gpu_cpu_backend_ops ops;
};

bool nvshmemt_gdrcopy_backend_init(nvshmemt_gpu_cpu_backend *backend, int log_level);
bool nvshmemt_dmabuf_backend_init(nvshmemt_gpu_cpu_backend *backend,
                                  struct nvshmemi_cuda_fn_table *cuda_syms, int log_level);
void nvshmemt_gpu_cpu_backend_reset(nvshmemt_gpu_cpu_backend *backend);

#endif
