/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_GDR_COMMON_H
#define _TRANSPORT_GDR_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "transport_gdr_abi.h"

struct nvshmemi_cuda_fn_table;

/*
 * Temporary compatibility surface for transports that have not yet moved to
 * the backend-neutral mapping API below. Remove once all users are migrated.
 */
struct gdrcopy_function_table {
    gdr_t (*open)();
    int (*close)(gdr_t g);
    int (*pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token,
                      uint32_t va_space, gdr_mh_t *handle);
    int (*unpin_buffer)(gdr_t g, gdr_mh_t handle);
    int (*get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
    int (*map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);
    int (*unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);
    int (*copy_from_mapping)(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
    int (*copy_to_mapping)(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
    void (*runtime_get_version)(int *major, int *minor);
    int (*driver_get_version)(gdr_t g, int *major, int *minor);
    int (*pin_buffer_v2)(gdr_t g, unsigned long addr, size_t size, uint32_t flags,
                         gdr_mh_t *handle);
    int (*map_v2)(gdr_t g, gdr_mh_t handle, void **va, size_t size, int flags);
    int (*get_attribute)(gdr_t g, int attr, int *value);
};

bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle, int log_level);
void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle);

enum nvshmemt_gpu_cpu_mapping_flags : uint32_t {
    NVSHMEMT_GPU_CPU_MAPPING_FLAG_NONE = 0,
    NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE = 1U,
};

enum nvshmemt_gpu_cpu_mapping_capability {
    NVSHMEMT_GPU_CPU_MAPPING_CAP_FORCE_PCIE = 1,
};

enum nvshmemt_gpu_cpu_mapping_type {
    NVSHMEMT_GPU_CPU_MAPPING_TYPE_UNKNOWN = 0,
    NVSHMEMT_GPU_CPU_MAPPING_TYPE_WC,
    NVSHMEMT_GPU_CPU_MAPPING_TYPE_CACHING,
    NVSHMEMT_GPU_CPU_MAPPING_TYPE_DEVICE,
};

/* Caller-owned state. Backend implementation details remain private to common transport code. */
struct nvshmemt_gpu_cpu_mapping_state {
    void *impl = nullptr;
};

/* Caller-owned mapping record. backend_handle is opaque outside the common implementation. */
struct nvshmemt_gpu_cpu_mapping {
    uintptr_t backend_handle = 0;
    void *gpu_ptr = nullptr;
    void *cpu_ptr_base = nullptr;
    void *cpu_ptr = nullptr;
    size_t size = 0;
    nvshmemt_gpu_cpu_mapping_type type = NVSHMEMT_GPU_CPU_MAPPING_TYPE_UNKNOWN;
    bool pinned = false;
    bool mapped = false;
};

bool nvshmemt_gpu_cpu_mapping_init(nvshmemt_gpu_cpu_mapping_state *state,
                                   struct nvshmemi_cuda_fn_table *cuda_syms, int log_level,
                                   bool force_internal_dmabuf);
void nvshmemt_gpu_cpu_mapping_fini(nvshmemt_gpu_cpu_mapping_state *state);

bool nvshmemt_gpu_cpu_mapping_is_available(const nvshmemt_gpu_cpu_mapping_state *state);
const char *nvshmemt_gpu_cpu_mapping_backend_name(const nvshmemt_gpu_cpu_mapping_state *state);
bool nvshmemt_gpu_cpu_mapping_has_capability(const nvshmemt_gpu_cpu_mapping_state *state,
                                             nvshmemt_gpu_cpu_mapping_capability capability);

int nvshmemt_gpu_cpu_map(nvshmemt_gpu_cpu_mapping_state *state, void *gpu_ptr, size_t size,
                         uint32_t flags, nvshmemt_gpu_cpu_mapping *mapping, size_t pin_size = 0);
int nvshmemt_gpu_cpu_unmap(nvshmemt_gpu_cpu_mapping_state *state,
                           nvshmemt_gpu_cpu_mapping *mapping);
int nvshmemt_gpu_cpu_copy_to(nvshmemt_gpu_cpu_mapping_state *state,
                             const nvshmemt_gpu_cpu_mapping *mapping, void *map_d_ptr,
                             const void *h_ptr, size_t size);
int nvshmemt_gpu_cpu_copy_from(nvshmemt_gpu_cpu_mapping_state *state,
                               const nvshmemt_gpu_cpu_mapping *mapping, void *h_ptr,
                               const void *map_d_ptr, size_t size);

#endif
