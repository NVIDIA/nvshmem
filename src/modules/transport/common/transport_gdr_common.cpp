/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transport_gdr_common.h"

#include <dlfcn.h>
#include <errno.h>

#include <new>

#include "transport_common.h"
#include "transport_gdr_abi.h"
#include "transport_gdr_internal.h"

namespace {

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

struct nvshmemt_gdrcopy_backend_context {
    gdrcopy_function_table ftable;
    gdr_t descriptor;
};

struct nvshmemt_gpu_cpu_mapping_impl {
    nvshmemt_gpu_cpu_backend backend;
    int log_level;
};

static nvshmemt_gpu_cpu_mapping_impl *nvshmemt_gpu_cpu_impl(
    const nvshmemt_gpu_cpu_mapping_state *state) {
    return state ? static_cast<nvshmemt_gpu_cpu_mapping_impl *>(state->impl) : nullptr;
}

static bool nvshmemt_gdr_is_at_least(int major, int minor, int required_major, int required_minor) {
    return major > required_major || (major == required_major && minor >= required_minor);
}

static gdr_mh_t nvshmemt_gdrcopy_handle(uintptr_t handle) {
    gdr_mh_t result{};
    result.h = static_cast<unsigned long>(handle);
    return result;
}

static nvshmemt_gpu_cpu_mapping_type nvshmemt_gdrcopy_mapping_type(const gdr_info_t &info) {
    switch (info.mapping_type) {
        case 1:
            return NVSHMEMT_GPU_CPU_MAPPING_TYPE_WC;
        case 2:
            return NVSHMEMT_GPU_CPU_MAPPING_TYPE_CACHING;
        case 3:
            return NVSHMEMT_GPU_CPU_MAPPING_TYPE_DEVICE;
        default:
            return info.wc_mapping ? NVSHMEMT_GPU_CPU_MAPPING_TYPE_WC
                                   : NVSHMEMT_GPU_CPU_MAPPING_TYPE_UNKNOWN;
    }
}

static int nvshmemt_gdrcopy_close(void *opaque_context) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }
    int status = 0;
    if (context->descriptor && context->ftable.close) {
        status = context->ftable.close(context->descriptor);
    }
    delete context;
    return status;
}

static int nvshmemt_gdrcopy_pin(void *opaque_context, uintptr_t gpu_va, size_t size, uint32_t flags,
                                uintptr_t *handle) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context || !handle || flags & ~NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        return EINVAL;
    }

    gdr_mh_t gdr_handle{};
    int status;
    if (flags & NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        if (!context->ftable.pin_buffer_v2) {
            return ENOTSUP;
        }
        status =
            context->ftable.pin_buffer_v2(context->descriptor, static_cast<unsigned long>(gpu_va),
                                          size, NVSHMEMT_GDR_PIN_FLAG_FORCE_PCIE, &gdr_handle);
    } else {
        status = context->ftable.pin_buffer(context->descriptor, static_cast<unsigned long>(gpu_va),
                                            size, 0, 0, &gdr_handle);
    }
    if (status == 0) {
        *handle = static_cast<uintptr_t>(gdr_handle.h);
    }
    return status;
}

static int nvshmemt_gdrcopy_unpin(void *opaque_context, uintptr_t handle) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }
    return context->ftable.unpin_buffer(context->descriptor, nvshmemt_gdrcopy_handle(handle));
}

static int nvshmemt_gdrcopy_get_info(void *opaque_context, uintptr_t handle,
                                     nvshmemt_gpu_cpu_backend_info *backend_info) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context || !backend_info) {
        return EINVAL;
    }

    gdr_info_t info{};
    int status =
        context->ftable.get_info(context->descriptor, nvshmemt_gdrcopy_handle(handle), &info);
    if (status != 0) {
        return status;
    }

    backend_info->gpu_va = static_cast<uintptr_t>(info.va);
    backend_info->mapped_size = static_cast<size_t>(info.mapped_size);
    backend_info->type = nvshmemt_gdrcopy_mapping_type(info);
    return 0;
}

static int nvshmemt_gdrcopy_map(void *opaque_context, uintptr_t handle, void **cpu_ptr_base,
                                size_t size, uint32_t flags) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context || !cpu_ptr_base || flags & ~NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        return EINVAL;
    }
    if (flags & NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        if (!context->ftable.map_v2) {
            return ENOTSUP;
        }
        return context->ftable.map_v2(context->descriptor, nvshmemt_gdrcopy_handle(handle),
                                      cpu_ptr_base, size, NVSHMEMT_GDR_MAP_FLAG_DEFAULT);
    }
    return context->ftable.map(context->descriptor, nvshmemt_gdrcopy_handle(handle), cpu_ptr_base,
                               size);
}

static int nvshmemt_gdrcopy_unmap(void *opaque_context, uintptr_t handle, void *cpu_ptr_base,
                                  size_t size) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }
    return context->ftable.unmap(context->descriptor, nvshmemt_gdrcopy_handle(handle), cpu_ptr_base,
                                 size);
}

static int nvshmemt_gdrcopy_copy_to(void *opaque_context, uintptr_t handle, void *map_d_ptr,
                                    const void *h_ptr, size_t size) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }
    return context->ftable.copy_to_mapping(nvshmemt_gdrcopy_handle(handle), map_d_ptr, h_ptr, size);
}

static int nvshmemt_gdrcopy_copy_from(void *opaque_context, uintptr_t handle, void *h_ptr,
                                      const void *map_d_ptr, size_t size) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }
    return context->ftable.copy_from_mapping(nvshmemt_gdrcopy_handle(handle), h_ptr, map_d_ptr,
                                             size);
}

static bool nvshmemt_gdrcopy_has_capability(void *opaque_context,
                                            nvshmemt_gpu_cpu_mapping_capability capability) {
    auto *context = static_cast<nvshmemt_gdrcopy_backend_context *>(opaque_context);
    if (!context || capability != NVSHMEMT_GPU_CPU_MAPPING_CAP_FORCE_PCIE ||
        !context->ftable.pin_buffer_v2 || !context->ftable.map_v2 ||
        !context->ftable.get_attribute) {
        return false;
    }

    int supported = 0;
    return context->ftable.get_attribute(context->descriptor,
                                         NVSHMEMT_GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE,
                                         &supported) == 0 &&
           supported != 0;
}

static bool nvshmemt_gdrcopy_library_symbols_available(const gdrcopy_function_table &ftable) {
    return ftable.open && ftable.close && ftable.pin_buffer && ftable.unpin_buffer &&
           ftable.get_info && ftable.map && ftable.unmap && ftable.copy_from_mapping &&
           ftable.copy_to_mapping && ftable.runtime_get_version;
}

static void nvshmemt_gdrcopy_library_cleanup(nvshmemt_gdrcopy_backend_context *context,
                                             void *library_handle) {
    if (context) {
        if (context->descriptor && context->ftable.close) {
            (void)context->ftable.close(context->descriptor);
        }
        delete context;
    }
    if (library_handle) {
        dlclose(library_handle);
    }
}

}  // namespace

void nvshmemt_gpu_cpu_backend_reset(nvshmemt_gpu_cpu_backend *backend) {
    if (!backend) {
        return;
    }
    if (backend->context && backend->ops.close) {
        (void)backend->ops.close(backend->context);
    }
    if (backend->library_handle) {
        dlclose(backend->library_handle);
    }
    *backend = {};
}

bool nvshmemt_gdrcopy_backend_init(nvshmemt_gpu_cpu_backend *backend, int log_level) {
    if (!backend) {
        return false;
    }
    *backend = {};

    void *library_handle = dlopen("libgdrapi.so.2", RTLD_LAZY);
    if (!library_handle) {
        INFO(log_level, "GDRCopy library not found.");
        return false;
    }

    auto *context = new (std::nothrow) nvshmemt_gdrcopy_backend_context{};
    if (!context) {
        dlclose(library_handle);
        return false;
    }

    auto &ftable = context->ftable;
    LOAD_SYM(library_handle, "gdr_runtime_get_version", ftable.runtime_get_version);
    LOAD_SYM(library_handle, "gdr_open", ftable.open);
    LOAD_SYM(library_handle, "gdr_close", ftable.close);
    LOAD_SYM(library_handle, "gdr_pin_buffer", ftable.pin_buffer);
    LOAD_SYM(library_handle, "gdr_unpin_buffer", ftable.unpin_buffer);
    LOAD_SYM(library_handle, "gdr_map", ftable.map);
    LOAD_SYM(library_handle, "gdr_unmap", ftable.unmap);
    LOAD_SYM(library_handle, "gdr_get_info", ftable.get_info);
    LOAD_SYM(library_handle, "gdr_copy_from_mapping", ftable.copy_from_mapping);
    LOAD_SYM(library_handle, "gdr_copy_to_mapping", ftable.copy_to_mapping);
    LOAD_SYM(library_handle, "gdr_driver_get_version", ftable.driver_get_version);

    int (*get_info_v2)(gdr_t, gdr_mh_t, gdr_info_t *) = nullptr;
    LOAD_SYM(library_handle, "gdr_get_info_v2", get_info_v2);
    LOAD_SYM(library_handle, "gdr_pin_buffer_v2", ftable.pin_buffer_v2);
    LOAD_SYM(library_handle, "gdr_map_v2", ftable.map_v2);
    LOAD_SYM(library_handle, "gdr_get_attribute", ftable.get_attribute);

    if (!nvshmemt_gdrcopy_library_symbols_available(ftable)) {
        INFO(log_level, "GDRCopy library is missing required symbols.");
        nvshmemt_gdrcopy_library_cleanup(context, library_handle);
        return false;
    }

    int runtime_major = 0;
    int runtime_minor = 0;
    ftable.runtime_get_version(&runtime_major, &runtime_minor);
    if (runtime_major != NVSHMEMT_GDR_API_MAJOR_VERSION) {
        INFO(log_level,
             "GDRCopy library major version %d is incompatible with the supported API major "
             "version %d.",
             runtime_major, NVSHMEMT_GDR_API_MAJOR_VERSION);
        nvshmemt_gdrcopy_library_cleanup(context, library_handle);
        return false;
    }
    INFO(log_level, "GDRCopy library version: (%d, %d)", runtime_major, runtime_minor);

    if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) && get_info_v2) {
        ftable.get_info = get_info_v2;
    }
    if (ftable.pin_buffer_v2 && ftable.map_v2 && ftable.get_attribute) {
        INFO(log_level,
             "GDRCopy v2 symbols found (gdr_pin_buffer_v2, gdr_map_v2, gdr_get_attribute).");
    }

    context->descriptor = ftable.open();
    if (!context->descriptor) {
        INFO(log_level, "GDRCopy open call failed.");
        nvshmemt_gdrcopy_library_cleanup(context, library_handle);
        return false;
    }

    bool using_dmabuf = false;
    if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) && ftable.get_attribute) {
        int value = 0;
        if (ftable.get_attribute(context->descriptor, NVSHMEMT_GDR_ATTR_USING_DMA_BUF_MMAP,
                                 &value) == 0 &&
            value) {
            using_dmabuf = true;
            INFO(log_level, "GPU CPU mapping enabled using GDRCopy %d.%d DMA-BUF mmap backend.",
                 runtime_major, runtime_minor);
        }
    }

    if (!using_dmabuf) {
        if (!ftable.driver_get_version) {
            INFO(log_level, "GDRCopy driver-version API is unavailable.");
            nvshmemt_gdrcopy_library_cleanup(context, library_handle);
            return false;
        }
        int driver_major = 0;
        int driver_minor = 0;
        int status = ftable.driver_get_version(context->descriptor, &driver_major, &driver_minor);
        if (nvshmemt_gdr_is_at_least(runtime_major, runtime_minor, 2, 6) && status != 0) {
            INFO(log_level, "GDRCopy driver-version query failed (%d).", status);
            nvshmemt_gdrcopy_library_cleanup(context, library_handle);
            return false;
        }
        INFO(log_level, "GDR driver version: (%d, %d)", driver_major, driver_minor);
    }

    backend->name = "GDRCopy library";
    backend->context = context;
    backend->library_handle = library_handle;
    backend->ops.close = nvshmemt_gdrcopy_close;
    backend->ops.pin = nvshmemt_gdrcopy_pin;
    backend->ops.unpin = nvshmemt_gdrcopy_unpin;
    backend->ops.get_info = nvshmemt_gdrcopy_get_info;
    backend->ops.map = nvshmemt_gdrcopy_map;
    backend->ops.unmap = nvshmemt_gdrcopy_unmap;
    backend->ops.copy_to = nvshmemt_gdrcopy_copy_to;
    backend->ops.copy_from = nvshmemt_gdrcopy_copy_from;
    backend->ops.has_capability = nvshmemt_gdrcopy_has_capability;
    return true;
}

bool nvshmemt_gpu_cpu_mapping_init(nvshmemt_gpu_cpu_mapping_state *state,
                                   struct nvshmemi_cuda_fn_table *cuda_syms, int log_level,
                                   bool force_internal_dmabuf) {
    if (!state) {
        return false;
    }
    nvshmemt_gpu_cpu_mapping_fini(state);

    auto *impl = new (std::nothrow) nvshmemt_gpu_cpu_mapping_impl{};
    if (!impl) {
        return false;
    }
    impl->log_level = log_level;

    if (force_internal_dmabuf) {
        INFO(log_level, "NVSHMEM_GDRCOPY_USE_INTERNAL_DMABUF is set; skipping libgdrapi.so.2.");
    } else if (nvshmemt_gdrcopy_backend_init(&impl->backend, log_level)) {
        state->impl = impl;
        return true;
    }

    nvshmemt_gpu_cpu_backend_reset(&impl->backend);
    if (nvshmemt_dmabuf_backend_init(&impl->backend, cuda_syms, log_level)) {
        state->impl = impl;
        return true;
    }

    INFO(log_level, "GPU CPU mapping disabled: no usable GDRCopy or CUDA DMA-BUF backend.");
    delete impl;
    return false;
}

void nvshmemt_gpu_cpu_mapping_fini(nvshmemt_gpu_cpu_mapping_state *state) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    if (!impl) {
        return;
    }
    nvshmemt_gpu_cpu_backend_reset(&impl->backend);
    delete impl;
    state->impl = nullptr;
}

bool nvshmemt_gpu_cpu_mapping_is_available(const nvshmemt_gpu_cpu_mapping_state *state) {
    return nvshmemt_gpu_cpu_impl(state) != nullptr;
}

const char *nvshmemt_gpu_cpu_mapping_backend_name(const nvshmemt_gpu_cpu_mapping_state *state) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    return impl && impl->backend.name ? impl->backend.name : "none";
}

bool nvshmemt_gpu_cpu_mapping_has_capability(const nvshmemt_gpu_cpu_mapping_state *state,
                                             nvshmemt_gpu_cpu_mapping_capability capability) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    return impl && impl->backend.ops.has_capability &&
           impl->backend.ops.has_capability(impl->backend.context, capability);
}

int nvshmemt_gpu_cpu_map(nvshmemt_gpu_cpu_mapping_state *state, void *gpu_ptr, size_t size,
                         uint32_t flags, nvshmemt_gpu_cpu_mapping *mapping, size_t pin_size) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    if (!impl || !gpu_ptr || size == 0 || !mapping || mapping->pinned || mapping->mapped ||
        flags & ~NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        return EINVAL;
    }
    if ((flags & NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) &&
        !nvshmemt_gpu_cpu_mapping_has_capability(state, NVSHMEMT_GPU_CPU_MAPPING_CAP_FORCE_PCIE)) {
        return ENOTSUP;
    }

    auto &backend = impl->backend;
    nvshmemt_gpu_cpu_mapping result{};
    nvshmemt_gpu_cpu_backend_info info{};
    const uintptr_t requested_va = reinterpret_cast<uintptr_t>(gpu_ptr);
    uintptr_t offset = 0;
    result.gpu_ptr = gpu_ptr;
    result.size = size;
    if (pin_size == 0) {
        pin_size = size;
    }
    if (pin_size < size) {
        return EINVAL;
    }

    int status = backend.ops.pin(backend.context, reinterpret_cast<uintptr_t>(gpu_ptr), pin_size,
                                 flags, &result.backend_handle);
    if (status != 0) {
        return status;
    }
    result.pinned = true;

    status =
        backend.ops.map(backend.context, result.backend_handle, &result.cpu_ptr_base, size, flags);
    if (status != 0) {
        goto out;
    }
    result.mapped = true;

    status = backend.ops.get_info(backend.context, result.backend_handle, &info);
    if (status != 0) {
        goto out;
    }

    if (requested_va < info.gpu_va) {
        status = EINVAL;
        goto out;
    }
    offset = requested_va - info.gpu_va;
    if (offset > info.mapped_size || size > info.mapped_size - offset ||
        reinterpret_cast<uintptr_t>(result.cpu_ptr_base) > UINTPTR_MAX - offset) {
        status = EINVAL;
        goto out;
    }

    result.cpu_ptr =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(result.cpu_ptr_base) + offset);
    result.type = info.type;
    *mapping = result;
    return 0;

out:
    if (result.mapped) {
        (void)backend.ops.unmap(backend.context, result.backend_handle, result.cpu_ptr_base, size);
    }
    if (result.pinned) {
        (void)backend.ops.unpin(backend.context, result.backend_handle);
    }
    return status;
}

int nvshmemt_gpu_cpu_unmap(nvshmemt_gpu_cpu_mapping_state *state,
                           nvshmemt_gpu_cpu_mapping *mapping) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    if (!impl || !mapping) {
        return EINVAL;
    }

    int first_error = 0;
    if (mapping->mapped) {
        int status = impl->backend.ops.unmap(impl->backend.context, mapping->backend_handle,
                                             mapping->cpu_ptr_base, mapping->size);
        if (status == 0) {
            mapping->mapped = false;
            mapping->cpu_ptr_base = nullptr;
            mapping->cpu_ptr = nullptr;
        } else {
            first_error = status;
        }
    }
    if (mapping->pinned) {
        /* Preserve a live handle when unpin fails so a later cleanup can retry it. */
        int status = impl->backend.ops.unpin(impl->backend.context, mapping->backend_handle);
        if (status == 0) {
            mapping->pinned = false;
            mapping->backend_handle = 0;
            if (first_error != 0) {
                mapping->mapped = false;
                mapping->cpu_ptr_base = nullptr;
                mapping->cpu_ptr = nullptr;
            }
        } else if (first_error == 0) {
            first_error = status;
        }
    }
    if (!mapping->mapped && !mapping->pinned) {
        *mapping = {};
    }
    return first_error;
}

int nvshmemt_gpu_cpu_copy_to(nvshmemt_gpu_cpu_mapping_state *state,
                             const nvshmemt_gpu_cpu_mapping *mapping, void *map_d_ptr,
                             const void *h_ptr, size_t size) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    if (!impl || !mapping || !mapping->mapped) {
        return EINVAL;
    }
    return impl->backend.ops.copy_to(impl->backend.context, mapping->backend_handle, map_d_ptr,
                                     h_ptr, size);
}

int nvshmemt_gpu_cpu_copy_from(nvshmemt_gpu_cpu_mapping_state *state,
                               const nvshmemt_gpu_cpu_mapping *mapping, void *h_ptr,
                               const void *map_d_ptr, size_t size) {
    auto *impl = nvshmemt_gpu_cpu_impl(state);
    if (!impl || !mapping || !mapping->mapped) {
        return EINVAL;
    }
    return impl->backend.ops.copy_from(impl->backend.context, mapping->backend_handle, h_ptr,
                                       map_d_ptr, size);
}
