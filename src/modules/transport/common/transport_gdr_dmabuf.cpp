/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transport_gdr_internal.h"

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <new>

#include "internal/host_transport/cudawrap.h"
#include "transport_common.h"

namespace {

/* CUDA 13.3 added this value. Keep the numeric definition local so NVSHMEM can still build with
 * older toolkit headers while using an R610-or-newer driver at runtime. */
static constexpr CUdevice_attribute NVSHMEMT_CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED =
    static_cast<CUdevice_attribute>(152);

struct nvshmemt_gdr_dmabuf_mapping;

struct nvshmemt_gdr_dmabuf_context {
    struct nvshmemi_cuda_fn_table *cuda_syms;
    size_t page_size;
    int log_level;
    std::mutex mappings_mutex;
    nvshmemt_gdr_dmabuf_mapping *mappings;
};

struct nvshmemt_gdr_dmabuf_mapping {
    nvshmemt_gdr_dmabuf_context *context;
    nvshmemt_gdr_dmabuf_mapping *next;
    int fd;
    uintptr_t gpu_va;
    size_t page_offset;
    size_t mapped_size;
    size_t page_size;
    void *cpu_ptr_base;
    size_t cpu_map_len;
    nvshmemt_gpu_cpu_mapping_type type;
    bool mapped;
};

static bool nvshmemt_gdr_round_up(size_t value, size_t alignment, size_t *rounded) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) {
        return false;
    }
    *rounded = ((value + alignment - 1) / alignment) * alignment;
    return true;
}

static nvshmemt_gdr_dmabuf_mapping *nvshmemt_gdr_mapping_from_handle(uintptr_t handle) {
    return reinterpret_cast<nvshmemt_gdr_dmabuf_mapping *>(handle);
}

static bool nvshmemt_gdr_dmabuf_remove_mapping(nvshmemt_gdr_dmabuf_context *context,
                                               nvshmemt_gdr_dmabuf_mapping *mapping) {
    std::lock_guard<std::mutex> lock(context->mappings_mutex);
    auto **current = &context->mappings;
    while (*current && *current != mapping) {
        current = &(*current)->next;
    }
    if (*current == mapping) {
        *current = mapping->next;
        mapping->next = nullptr;
        mapping->context = nullptr;
        return true;
    }
    return false;
}

static int nvshmemt_gdr_dmabuf_release_mapping(nvshmemt_gdr_dmabuf_mapping *mapping) {
    int status = 0;
    if (mapping->mapped && mapping->cpu_ptr_base) {
        if (munmap(mapping->cpu_ptr_base, mapping->cpu_map_len) != 0) {
            status = errno;
        } else {
            mapping->cpu_ptr_base = nullptr;
            mapping->cpu_map_len = 0;
            mapping->mapped = false;
        }
    }
    if (mapping->fd >= 0) {
        /* Linux consumes descriptor ownership even when close reports a late error. */
        int fd = mapping->fd;
        mapping->fd = -1;
        if (::close(fd) != 0 && status == 0) {
            status = errno;
        }
    }
    return status;
}

static int nvshmemt_gdr_dmabuf_destroy_mapping(nvshmemt_gdr_dmabuf_mapping *mapping) {
    int status = nvshmemt_gdr_dmabuf_release_mapping(mapping);
    delete mapping;
    return status;
}

static int nvshmemt_gdr_cuda_error(struct nvshmemi_cuda_fn_table *cuda_syms, CUresult status,
                                   const char *operation, int log_level) {
    const char *error_name = "UNKNOWN";
    const char *error_description = "Unknown";
    nvshmemi_cu_get_error_info(cuda_syms, status, &error_name, &error_description);
    INFO(log_level, "Internal CUDA DMA-BUF backend: %s failed: %s (%s)", operation, error_name,
         error_description);
    return status == CUDA_SUCCESS ? EIO : static_cast<int>(status);
}

static bool nvshmemt_gdr_dmabuf_device_supported(struct nvshmemi_cuda_fn_table *cuda_syms,
                                                 CUdevice device, int log_level) {
    int export_supported = 0;
    CUresult status = CUPFN(
        cuda_syms,
        cuDeviceGetAttribute(&export_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device));
    if (status != CUDA_SUCCESS || !export_supported) {
        INFO(log_level,
             "Internal CUDA DMA-BUF backend unavailable: CUDA device lacks DMA-BUF export "
             "support.");
        return false;
    }

    int mmap_supported = 0;
    status = CUPFN(cuda_syms, cuDeviceGetAttribute(
                                  &mmap_supported,
                                  NVSHMEMT_CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED, device));
    if (status != CUDA_SUCCESS || !mmap_supported) {
        INFO(log_level,
             "Internal CUDA DMA-BUF backend unavailable: CUDA device lacks DMA-BUF mmap "
             "support.");
        return false;
    }
    return true;
}

static bool nvshmemt_gdr_dmabuf_supported(struct nvshmemi_cuda_fn_table *cuda_syms, int log_level) {
    if (!cuda_syms || !CUPFN(cuda_syms, cuDriverGetVersion) || !CUPFN(cuda_syms, cuCtxGetDevice) ||
        !CUPFN(cuda_syms, cuDeviceGetAttribute) || !CUPFN(cuda_syms, cuPointerGetAttribute) ||
        !CUPFN(cuda_syms, cuDeviceGet) || !CUPFN(cuda_syms, cuCtxGetCurrent) ||
        !CUPFN(cuda_syms, cuCtxSetCurrent) || !CUPFN(cuda_syms, cuDevicePrimaryCtxRetain) ||
        !CUPFN(cuda_syms, cuDevicePrimaryCtxRelease) ||
        !CUPFN(cuda_syms, cuMemGetHandleForAddressRange)) {
        INFO(log_level, "Internal CUDA DMA-BUF backend unavailable: required CUDA APIs missing.");
        return false;
    }

    int driver_version = 0;
    if (CUPFN(cuda_syms, cuDriverGetVersion(&driver_version)) != CUDA_SUCCESS ||
        driver_version < 13030) {
        INFO(log_level,
             "Internal CUDA DMA-BUF backend unavailable: CUDA driver %d is older than 13.3/R610.",
             driver_version);
        return false;
    }

    CUcontext current_context = nullptr;
    if (CUPFN(cuda_syms, cuCtxGetCurrent(&current_context)) == CUDA_SUCCESS && current_context) {
        CUdevice device;
        if (CUPFN(cuda_syms, cuCtxGetDevice(&device)) != CUDA_SUCCESS ||
            !nvshmemt_gdr_dmabuf_device_supported(cuda_syms, device, log_level)) {
            return false;
        }
    } else {
        INFO(log_level,
             "Internal CUDA DMA-BUF backend: deferring GPU capability check until first mapping.");
    }

    return true;
}

static int nvshmemt_gdr_dmabuf_close(void *opaque_context) {
    auto *context = static_cast<nvshmemt_gdr_dmabuf_context *>(opaque_context);
    if (!context) {
        return EINVAL;
    }

    nvshmemt_gdr_dmabuf_mapping *mapping = nullptr;
    {
        std::lock_guard<std::mutex> lock(context->mappings_mutex);
        mapping = context->mappings;
        context->mappings = nullptr;
        for (auto *current = mapping; current; current = current->next) {
            current->context = nullptr;
        }
    }

    int status = 0;
    while (mapping) {
        auto *next = mapping->next;
        mapping->next = nullptr;
        int mapping_status = nvshmemt_gdr_dmabuf_destroy_mapping(mapping);
        if (status == 0) {
            status = mapping_status;
        }
        mapping = next;
    }

    delete context;
    return status;
}

static int nvshmemt_gdr_dmabuf_pin(void *opaque_context, uintptr_t gpu_va, size_t size,
                                   uint32_t flags, uintptr_t *handle) {
    auto *context = static_cast<nvshmemt_gdr_dmabuf_context *>(opaque_context);
    if (!context || !handle || size == 0 || flags & ~NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        return EINVAL;
    }
    auto *cuda_syms = context->cuda_syms;
    *handle = 0;

    uintptr_t aligned_gpu_va = gpu_va - (gpu_va % context->page_size);
    size_t page_offset = gpu_va - aligned_gpu_va;
    size_t mapped_size;
    if (size > SIZE_MAX - page_offset ||
        !nvshmemt_gdr_round_up(size + page_offset, context->page_size, &mapped_size)) {
        return EINVAL;
    }

    int device_ordinal = -1;
    CUresult cuda_status =
        CUPFN(cuda_syms, cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                               static_cast<CUdeviceptr>(gpu_va)));
    if (cuda_status != CUDA_SUCCESS) {
        return nvshmemt_gdr_cuda_error(cuda_syms, cuda_status,
                                       "cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)",
                                       context->log_level);
    }

    CUdevice device;
    cuda_status = CUPFN(cuda_syms, cuDeviceGet(&device, device_ordinal));
    if (cuda_status != CUDA_SUCCESS) {
        return nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuDeviceGet", context->log_level);
    }
    if (!nvshmemt_gdr_dmabuf_device_supported(cuda_syms, device, context->log_level)) {
        return ENOTSUP;
    }

    int coherent = 0;
    cuda_status = CUPFN(
        cuda_syms,
        cuDeviceGetAttribute(
            &coherent, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device));
    if (cuda_status != CUDA_SUCCESS) {
        coherent = 0;
    }

    CUcontext previous_context = nullptr;
    CUcontext retained_context = nullptr;
    bool switched_context = false;
    bool retained_primary_context = false;
    bool use_primary_context = false;
    bool handle_created = false;
    unsigned long long dmabuf_flags = 0;
    int fd = -1;
    int status = 0;
    auto *mapping = static_cast<nvshmemt_gdr_dmabuf_mapping *>(nullptr);

    cuda_status = CUPFN(cuda_syms, cuCtxGetCurrent(&previous_context));
    if (cuda_status != CUDA_SUCCESS) {
        status =
            nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuCtxGetCurrent", context->log_level);
        goto out;
    }

    use_primary_context = previous_context == nullptr;
    if (!use_primary_context) {
        CUdevice current_device;
        cuda_status = CUPFN(cuda_syms, cuCtxGetDevice(&current_device));
        if (cuda_status != CUDA_SUCCESS) {
            status = nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuCtxGetDevice",
                                             context->log_level);
            goto out;
        }
        use_primary_context = current_device != device;
    }

    if (use_primary_context) {
        cuda_status = CUPFN(cuda_syms, cuDevicePrimaryCtxRetain(&retained_context, device));
        if (cuda_status != CUDA_SUCCESS) {
            status = nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuDevicePrimaryCtxRetain",
                                             context->log_level);
            goto out;
        }
        retained_primary_context = true;

        cuda_status = CUPFN(cuda_syms, cuCtxSetCurrent(retained_context));
        if (cuda_status != CUDA_SUCCESS) {
            status = nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuCtxSetCurrent",
                                             context->log_level);
            goto out;
        }
        switched_context = true;
    }

    if (flags & NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE) {
        dmabuf_flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
    }
    cuda_status = CUPFN(cuda_syms, cuMemGetHandleForAddressRange(
                                       &fd, static_cast<CUdeviceptr>(aligned_gpu_va), mapped_size,
                                       CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, dmabuf_flags));
    if (cuda_status != CUDA_SUCCESS) {
        status = nvshmemt_gdr_cuda_error(cuda_syms, cuda_status, "cuMemGetHandleForAddressRange",
                                         context->log_level);
        goto out;
    }

    mapping = new (std::nothrow) nvshmemt_gdr_dmabuf_mapping{};
    if (!mapping) {
        status = ENOMEM;
        goto out;
    }

    mapping->context = context;
    mapping->fd = fd;
    mapping->gpu_va = aligned_gpu_va;
    mapping->page_offset = page_offset;
    mapping->mapped_size = mapped_size;
    mapping->page_size = context->page_size;
    mapping->type = (coherent && !(flags & NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE))
                        ? NVSHMEMT_GPU_CPU_MAPPING_TYPE_CACHING
                        : NVSHMEMT_GPU_CPU_MAPPING_TYPE_WC;
    {
        std::lock_guard<std::mutex> lock(context->mappings_mutex);
        mapping->next = context->mappings;
        context->mappings = mapping;
    }
    *handle = reinterpret_cast<uintptr_t>(mapping);
    fd = -1;
    handle_created = true;

out:
    if (fd >= 0) {
        (void)::close(fd);
    }
    if (switched_context) {
        cuda_status = CUPFN(cuda_syms, cuCtxSetCurrent(previous_context));
        if (cuda_status != CUDA_SUCCESS) {
            int cleanup_status = nvshmemt_gdr_cuda_error(
                cuda_syms, cuda_status, "cuCtxSetCurrent(previous)", context->log_level);
            if (status == 0) {
                status = cleanup_status;
            }
        }
    }
    if (retained_primary_context) {
        cuda_status = CUPFN(cuda_syms, cuDevicePrimaryCtxRelease(device));
        if (cuda_status != CUDA_SUCCESS) {
            int cleanup_status = nvshmemt_gdr_cuda_error(
                cuda_syms, cuda_status, "cuDevicePrimaryCtxRelease", context->log_level);
            if (status == 0) {
                status = cleanup_status;
            }
        }
    }
    if (status != 0 && handle_created) {
        if (nvshmemt_gdr_dmabuf_remove_mapping(context, mapping)) {
            (void)nvshmemt_gdr_dmabuf_destroy_mapping(mapping);
        }
        *handle = 0;
    }
    return status;
}

static int nvshmemt_gdr_dmabuf_unpin(void *opaque_context, uintptr_t handle) {
    auto *context = static_cast<nvshmemt_gdr_dmabuf_context *>(opaque_context);
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!context || !mapping) {
        return EINVAL;
    }

    std::lock_guard<std::mutex> lock(context->mappings_mutex);
    auto **current = &context->mappings;
    while (*current && *current != mapping) {
        current = &(*current)->next;
    }
    if (*current != mapping) {
        return EINVAL;
    }

    int status = nvshmemt_gdr_dmabuf_release_mapping(mapping);
    if (status != 0) {
        return status;
    }

    *current = mapping->next;
    mapping->next = nullptr;
    mapping->context = nullptr;
    delete mapping;
    return 0;
}

static int nvshmemt_gdr_dmabuf_get_info(void *, uintptr_t handle,
                                        nvshmemt_gpu_cpu_backend_info *info) {
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!mapping || !info) {
        return EINVAL;
    }
    info->gpu_va = mapping->gpu_va;
    info->mapped_size = mapping->mapped_size;
    info->type = mapping->type;
    return 0;
}

static int nvshmemt_gdr_dmabuf_map(void *opaque_context, uintptr_t handle, void **va, size_t size,
                                   uint32_t flags) {
    auto *context = static_cast<nvshmemt_gdr_dmabuf_context *>(opaque_context);
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!context || !mapping || !va || flags & ~NVSHMEMT_GPU_CPU_MAPPING_FLAG_FORCE_PCIE ||
        mapping->fd < 0 || mapping->mapped || size == 0 ||
        mapping->page_offset > mapping->mapped_size ||
        size > mapping->mapped_size - mapping->page_offset) {
        return EINVAL;
    }

    size_t map_length;
    if (size > SIZE_MAX - mapping->page_offset ||
        !nvshmemt_gdr_round_up(size + mapping->page_offset, context->page_size, &map_length)) {
        return EINVAL;
    }

    void *cpu_ptr = mmap(nullptr, map_length, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (cpu_ptr == MAP_FAILED) {
        int status = errno;
        INFO(context->log_level, "Internal CUDA DMA-BUF backend: mmap(fd=%d, size=%zu) failed: %s",
             mapping->fd, map_length, strerror(status));
        return status;
    }

    mapping->cpu_ptr_base = cpu_ptr;
    mapping->cpu_map_len = map_length;
    mapping->mapped = true;
    *va = cpu_ptr;
    return 0;
}

static int nvshmemt_gdr_dmabuf_unmap(void *opaque_context, uintptr_t handle, void *va,
                                     size_t size) {
    auto *context = static_cast<nvshmemt_gdr_dmabuf_context *>(opaque_context);
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!context || !mapping || !mapping->mapped || va != mapping->cpu_ptr_base || size == 0 ||
        mapping->page_offset > mapping->mapped_size ||
        size > mapping->mapped_size - mapping->page_offset) {
        return EINVAL;
    }

    size_t map_length;
    if (size > SIZE_MAX - mapping->page_offset ||
        !nvshmemt_gdr_round_up(size + mapping->page_offset, context->page_size, &map_length) ||
        map_length != mapping->cpu_map_len) {
        return EINVAL;
    }

    if (munmap(mapping->cpu_ptr_base, map_length) != 0) {
        int status = errno;
        INFO(context->log_level,
             "Internal CUDA DMA-BUF backend: munmap(ptr=%p, size=%zu) failed: %s",
             mapping->cpu_ptr_base, map_length, strerror(status));
        return status;
    }

    mapping->cpu_ptr_base = nullptr;
    mapping->cpu_map_len = 0;
    mapping->mapped = false;
    return 0;
}

static int nvshmemt_gdr_dmabuf_copy_to(void *, uintptr_t handle, void *map_d_ptr, const void *h_ptr,
                                       size_t size) {
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!mapping || !mapping->mapped || (!map_d_ptr && size != 0) || (!h_ptr && size != 0)) {
        return EINVAL;
    }
    if (size != 0) {
        memcpy(map_d_ptr, h_ptr, size);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return 0;
}

static int nvshmemt_gdr_dmabuf_copy_from(void *, uintptr_t handle, void *h_ptr,
                                         const void *map_d_ptr, size_t size) {
    auto *mapping = nvshmemt_gdr_mapping_from_handle(handle);
    if (!mapping || !mapping->mapped || (!map_d_ptr && size != 0) || (!h_ptr && size != 0)) {
        return EINVAL;
    }
    if (size != 0) {
        memcpy(h_ptr, map_d_ptr, size);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return 0;
}

static bool nvshmemt_gdr_dmabuf_has_capability(void *,
                                               nvshmemt_gpu_cpu_mapping_capability capability) {
    return capability == NVSHMEMT_GPU_CPU_MAPPING_CAP_FORCE_PCIE;
}

}  // namespace

bool nvshmemt_dmabuf_backend_init(nvshmemt_gpu_cpu_backend *backend,
                                  struct nvshmemi_cuda_fn_table *cuda_syms, int log_level) {
    if (!backend || !nvshmemt_gdr_dmabuf_supported(cuda_syms, log_level)) {
        return false;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        INFO(log_level,
             "Internal CUDA DMA-BUF backend unavailable: cannot determine host page size.");
        return false;
    }

    auto *context = new (std::nothrow) nvshmemt_gdr_dmabuf_context{
        cuda_syms, static_cast<size_t>(page_size), log_level, {}, nullptr};
    if (!context) {
        return false;
    }

    *backend = {};
    backend->name = "internal CUDA DMA-BUF";
    backend->context = context;
    backend->ops.close = nvshmemt_gdr_dmabuf_close;
    backend->ops.pin = nvshmemt_gdr_dmabuf_pin;
    backend->ops.unpin = nvshmemt_gdr_dmabuf_unpin;
    backend->ops.get_info = nvshmemt_gdr_dmabuf_get_info;
    backend->ops.map = nvshmemt_gdr_dmabuf_map;
    backend->ops.unmap = nvshmemt_gdr_dmabuf_unmap;
    backend->ops.copy_from = nvshmemt_gdr_dmabuf_copy_from;
    backend->ops.copy_to = nvshmemt_gdr_dmabuf_copy_to;
    backend->ops.has_capability = nvshmemt_gdr_dmabuf_has_capability;

    INFO(log_level, "GPU CPU mapping enabled using internal CUDA DMA-BUF mmap backend.");
    return true;
}
