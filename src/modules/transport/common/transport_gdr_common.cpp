/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transport_gdr_common.h"

#include <dlfcn.h>
#include <string.h>

#include "transport_common.h"

namespace {

static bool nvshmemt_gdr_is_at_least(int major, int minor, int required_major, int required_minor) {
    return major > required_major || (major == required_major && minor >= required_minor);
}

static void nvshmemt_gdrcopy_reset(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                   void **gdrcopy_handle) {
    if (gdr_desc && *gdr_desc && gdrcopy_ftable && gdrcopy_ftable->close) {
        (void)gdrcopy_ftable->close(*gdr_desc);
    }
    if (gdr_desc) *gdr_desc = nullptr;
    if (gdrcopy_handle && *gdrcopy_handle) {
        dlclose(*gdrcopy_handle);
        *gdrcopy_handle = nullptr;
    }
    if (gdrcopy_ftable) memset(gdrcopy_ftable, 0, sizeof(*gdrcopy_ftable));
}

static bool nvshmemt_gdrcopy_library_symbols_available(
    const struct gdrcopy_function_table *gdrcopy_ftable) {
    return gdrcopy_ftable->open && gdrcopy_ftable->close && gdrcopy_ftable->pin_buffer &&
           gdrcopy_ftable->unpin_buffer && gdrcopy_ftable->get_info && gdrcopy_ftable->map &&
           gdrcopy_ftable->unmap && gdrcopy_ftable->copy_from_mapping &&
           gdrcopy_ftable->copy_to_mapping && gdrcopy_ftable->runtime_get_version;
}

static bool nvshmemt_gdrcopy_library_init(struct gdrcopy_function_table *gdrcopy_ftable,
                                          gdr_t *gdr_desc, void **gdrcopy_handle, int log_level) {
    *gdrcopy_handle = dlopen("libgdrapi.so.2", RTLD_LAZY);
    if (!*gdrcopy_handle) {
        INFO(log_level, "GDRCopy library not found.");
        return false;
    }

    void *local_gdrcopy_handle = *gdrcopy_handle;
    LOAD_SYM(local_gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable->runtime_get_version);
    LOAD_SYM(local_gdrcopy_handle, "gdr_open", gdrcopy_ftable->open);
    LOAD_SYM(local_gdrcopy_handle, "gdr_close", gdrcopy_ftable->close);
    LOAD_SYM(local_gdrcopy_handle, "gdr_pin_buffer", gdrcopy_ftable->pin_buffer);
    LOAD_SYM(local_gdrcopy_handle, "gdr_unpin_buffer", gdrcopy_ftable->unpin_buffer);
    LOAD_SYM(local_gdrcopy_handle, "gdr_map", gdrcopy_ftable->map);
    LOAD_SYM(local_gdrcopy_handle, "gdr_unmap", gdrcopy_ftable->unmap);
    LOAD_SYM(local_gdrcopy_handle, "gdr_get_info", gdrcopy_ftable->get_info);
    LOAD_SYM(local_gdrcopy_handle, "gdr_copy_from_mapping", gdrcopy_ftable->copy_from_mapping);
    LOAD_SYM(local_gdrcopy_handle, "gdr_copy_to_mapping", gdrcopy_ftable->copy_to_mapping);
    LOAD_SYM(local_gdrcopy_handle, "gdr_driver_get_version", gdrcopy_ftable->driver_get_version);

    int (*get_info_v2)(gdr_t, gdr_mh_t, gdr_info_t *) = nullptr;
    LOAD_SYM(local_gdrcopy_handle, "gdr_get_info_v2", get_info_v2);
    LOAD_SYM(local_gdrcopy_handle, "gdr_pin_buffer_v2", gdrcopy_ftable->pin_buffer_v2);
    LOAD_SYM(local_gdrcopy_handle, "gdr_map_v2", gdrcopy_ftable->map_v2);
    LOAD_SYM(local_gdrcopy_handle, "gdr_get_attribute", gdrcopy_ftable->get_attribute);

    if (!nvshmemt_gdrcopy_library_symbols_available(gdrcopy_ftable)) {
        INFO(log_level, "GDRCopy library is missing required symbols.");
        nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
        return false;
    }

    int runtime_major = 0;
    int runtime_minor = 0;
    gdrcopy_ftable->runtime_get_version(&runtime_major, &runtime_minor);
    if (runtime_major != NVSHMEMT_GDR_API_MAJOR_VERSION) {
        INFO(log_level,
             "GDRCopy library major version %d is incompatible with the supported API major "
             "version %d.",
             runtime_major, NVSHMEMT_GDR_API_MAJOR_VERSION);
        nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
        return false;
    }
    INFO(log_level, "GDRCopy library version: (%d, %d)", runtime_major, runtime_minor);

    if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) && get_info_v2) {
        gdrcopy_ftable->get_info = get_info_v2;
    }
    if (gdrcopy_ftable->pin_buffer_v2 && gdrcopy_ftable->map_v2 && gdrcopy_ftable->get_attribute) {
        INFO(log_level,
             "GDRCopy v2 symbols found (gdr_pin_buffer_v2, gdr_map_v2, gdr_get_attribute).");
    }

    *gdr_desc = gdrcopy_ftable->open();
    if (!*gdr_desc) {
        INFO(log_level, "GDRCopy open call failed.");
        nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
        return false;
    }

    bool using_dmabuf = false;
    if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) &&
        gdrcopy_ftable->get_attribute) {
        int value = 0;
        if (gdrcopy_ftable->get_attribute(*gdr_desc, NVSHMEMT_GDR_ATTR_USING_DMA_BUF_MMAP,
                                          &value) == 0 &&
            value) {
            using_dmabuf = true;
            INFO(log_level, "GDRCopy enabled using library %d.%d DMA-BUF mmap backend.",
                 runtime_major, runtime_minor);
        }
    }

    if (!using_dmabuf) {
        if (!gdrcopy_ftable->driver_get_version) {
            INFO(log_level, "GDRCopy driver-version API is unavailable.");
            nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
            return false;
        }

        int driver_major = 0;
        int driver_minor = 0;
        int status = gdrcopy_ftable->driver_get_version(*gdr_desc, &driver_major, &driver_minor);
        if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) && status != 0) {
            INFO(log_level, "GDRCopy driver-version query failed (%d).", status);
            nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
            return false;
        }
        INFO(log_level, "GDR driver version: (%d, %d)", driver_major, driver_minor);
    }

    return true;
}

}  // namespace

bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle, int log_level) {
    if (!gdrcopy_ftable || !gdr_desc || !gdrcopy_handle) return false;

    *gdr_desc = nullptr;
    *gdrcopy_handle = nullptr;
    memset(gdrcopy_ftable, 0, sizeof(*gdrcopy_ftable));
    return nvshmemt_gdrcopy_library_init(gdrcopy_ftable, gdr_desc, gdrcopy_handle, log_level);
}

void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle) {
    if (!gdrcopy_ftable) return;
    nvshmemt_gdrcopy_reset(gdrcopy_ftable, gdr_desc, gdrcopy_handle);
}
