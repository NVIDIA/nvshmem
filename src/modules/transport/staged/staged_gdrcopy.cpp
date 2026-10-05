/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "staged_gdrcopy.h"

#include <stdint.h>
#include <algorithm>
#include <memory>
#include <mutex>

#include "non_abi/c/nvshmemi_error_macros.h"
#include "transport_common.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

namespace nvshmemi {

#ifdef NVSHMEM_USE_GDRCOPY

int staged_gdrcopy_init(transport_staged_state_t* s, struct nvshmemi_cuda_fn_table* cuda_syms) {
    if (s->options.DISABLE_GDRCOPY) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] NVSHMEM_STAGED_COPY_POLICY=GDRCOPY requires GPU CPU mapping, but "
            "NVSHMEM_DISABLE_GDRCOPY=1 is set");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (!nvshmemt_gpu_cpu_mapping_init(&s->memory.gpu_cpu_mapping_state, cuda_syms, s->log_level,
                                       s->options.GDRCOPY_USE_INTERNAL_DMABUF)) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] NVSHMEM_STAGED_COPY_POLICY=GDRCOPY requires a usable GPU CPU "
            "mapping backend; staged initialization failed");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    INFO(s->log_level, "[STAGED] copy policy GDRCOPY selected (%s backend)",
         nvshmemt_gpu_cpu_mapping_backend_name(&s->memory.gpu_cpu_mapping_state));
    return 0;
}

int staged_gdrcopy_fini(transport_staged_state_t* s) {
    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    int status = 0;
    for (const auto& [_, info] : s->memory.mem_handle_infos) {
        int unmap_status =
            nvshmemt_gpu_cpu_unmap(&s->memory.gpu_cpu_mapping_state, &info->cpu_mapping);
        if (unmap_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping final cleanup failed status=%d",
                                 unmap_status);
            status = NVSHMEMX_ERROR_INTERNAL;
        }
    }
    s->memory.mem_handle_infos.clear();
    nvshmemt_gpu_cpu_mapping_fini(&s->memory.gpu_cpu_mapping_state);
    return status;
}

int staged_gdrcopy_map(transport_staged_state_t* s, void* buf, size_t length, void* mapping_buf,
                       staged_mem_handle_info_t** info_out) {
    auto info = std::make_unique<staged_mem_handle_info_t>();
    info->ptr = buf;
    info->size = length;

    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    int status = nvshmemt_gpu_cpu_map(&s->memory.gpu_cpu_mapping_state, mapping_buf, length,
                                      NVSHMEMT_GPU_CPU_MAPPING_FLAG_NONE, &info->cpu_mapping);
    if (status != 0) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] CPU-mapping policy (%s) failed for %p len=%zu status=%d; "
            "memory registration failed",
            nvshmemt_gpu_cpu_mapping_backend_name(&s->memory.gpu_cpu_mapping_state), mapping_buf,
            length, status);
        return NVSHMEMX_ERROR_INTERNAL;
    }
    staged_mem_handle_info_t* stored_info = info.get();
    s->memory.mem_handle_infos.emplace(stored_info, std::move(info));
    if (info_out) {
        *info_out = stored_info;
    }
    return 0;
}

int staged_gdrcopy_unmap(transport_staged_state_t* s, staged_mem_handle_info_t* info) {
    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    auto it = s->memory.mem_handle_infos.find(info);
    if (it == s->memory.mem_handle_infos.end()) {
        return 0;
    }

    int status = nvshmemt_gpu_cpu_unmap(&s->memory.gpu_cpu_mapping_state, &it->second->cpu_mapping);
    if (status != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping cleanup failed status=%d", status);
        return NVSHMEMX_ERROR_INTERNAL;
    }
    s->memory.mem_handle_infos.erase(it);
    return 0;
}

/* Caller holds gpu_cpu_mapping_mutex for the lifetime of the returned pointer. */
static staged_mem_handle_info_t* staged_gdrcopy_find(transport_staged_state_t* s, const void* ptr,
                                                     size_t bytes) {
    if (!ptr) {
        return nullptr;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    auto it = std::find_if(s->memory.mem_handle_infos.begin(), s->memory.mem_handle_infos.end(),
                           [addr, bytes](const auto& entry) {
                               const auto& info = entry.second;
                               if (!info || !info->ptr) {
                                   return false;
                               }
                               const uintptr_t begin = reinterpret_cast<uintptr_t>(info->ptr);
                               if (addr < begin) {
                                   return false;
                               }
                               const size_t offset = static_cast<size_t>(addr - begin);
                               return offset <= info->size && bytes <= info->size - offset;
                           });
    return it == s->memory.mem_handle_infos.end() ? nullptr : it->second.get();
}

/* CPU address of mapped GPU memory, or nullptr. Caller holds gpu_cpu_mapping_mutex. */
static void* staged_gdrcopy_cpu_address(transport_staged_state_t* s, const void* gpu_ptr,
                                        size_t bytes, staged_mem_handle_info_t** info_out) {
    auto* info = staged_gdrcopy_find(s, gpu_ptr, bytes);
    if (!info || !info->cpu_mapping.mapped || !info->cpu_mapping.cpu_ptr) {
        return nullptr;
    }
    *info_out = info;
    return static_cast<char*>(info->cpu_mapping.cpu_ptr) +
           (static_cast<const char*>(gpu_ptr) - static_cast<char*>(info->ptr));
}

int staged_gdrcopy_copy(transport_staged_state_t* s, void* dst, bool dst_on_gpu, const void* src,
                        bool src_on_gpu, size_t bytes, const char* label) {
    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    staged_mem_handle_info_t* dst_info = nullptr;
    staged_mem_handle_info_t* src_info = nullptr;
    void* dst_cpu = dst_on_gpu ? staged_gdrcopy_cpu_address(s, dst, bytes, &dst_info) : dst;
    const void* src_cpu = src_on_gpu ? staged_gdrcopy_cpu_address(s, src, bytes, &src_info) : src;
    if (!dst_cpu || !src_cpu) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s requires a GPU CPU mapping for %p (bytes=%zu)", label,
                             dst_cpu ? src : dst, bytes);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    int status = 0;
    if (dst_on_gpu) {
        status = nvshmemt_gpu_cpu_copy_to(&s->memory.gpu_cpu_mapping_state, &dst_info->cpu_mapping,
                                          dst_cpu, src_cpu, bytes);
        if (status == 0) {
            /* Wait for the writes to reach GPU memory before the NIC or GPU reads them: drain the
             * write-combining buffers, then read back through the mapping. */
            STORE_BARRIER();
            const volatile char* last_byte = static_cast<char*>(dst_cpu) + bytes - 1;
            (void)*last_byte;
        }
    } else {
        status = nvshmemt_gpu_cpu_copy_from(&s->memory.gpu_cpu_mapping_state,
                                            &src_info->cpu_mapping, dst_cpu, src_cpu, bytes);
    }
    if (status != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED] %s GPU CPU mapping copy failed: %d", label, status);
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return 0;
}

#else

int staged_gdrcopy_init(transport_staged_state_t*, struct nvshmemi_cuda_fn_table*) {
    NVSHMEMI_ERROR_PRINT(
        "[STAGED] NVSHMEM_STAGED_COPY_POLICY=GDRCOPY requested, but NVSHMEM was "
        "built without GPU CPU mapping support");
    return NVSHMEMX_ERROR_INVALID_VALUE;
}

int staged_gdrcopy_fini(transport_staged_state_t*) { return 0; }

int staged_gdrcopy_map(transport_staged_state_t*, void*, size_t, void*,
                       staged_mem_handle_info_t**) {
    return NVSHMEMX_ERROR_INTERNAL;
}

int staged_gdrcopy_unmap(transport_staged_state_t*, staged_mem_handle_info_t*) { return 0; }

int staged_gdrcopy_copy(transport_staged_state_t*, void*, bool, const void*, bool, size_t,
                        const char*) {
    return NVSHMEMX_ERROR_INTERNAL;
}

#endif

}  // namespace nvshmemi
