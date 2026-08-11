/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_GDR_COMMON_H
#define _TRANSPORT_GDR_COMMON_H

#include "transport_gdr_abi.h"

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
    /* GDRCopy 2.5+ v2 APIs. These may be NULL at runtime if the loaded
     * libgdrapi.so.2 predates 2.5; callers must check before invoking. */
    int (*pin_buffer_v2)(gdr_t g, unsigned long addr, size_t size, uint32_t flags,
                         gdr_mh_t *handle);
    int (*map_v2)(gdr_t g, gdr_mh_t handle, void **va, size_t size, int flags);
    int (*get_attribute)(gdr_t g, int attr, int *value);
};

bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle, int log_level);
void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle);

#endif
