/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "staged_gdrcopy.h"

#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

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
    staged_mem_handle_registry_t mem_handle_infos;
    {
        std::lock_guard<std::mutex> lk(s->memory.mem_handle_mutex);
        mem_handle_infos.swap(s->memory.mem_handle_infos);
    }

    std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
    int status = 0;
    for (const auto& [_, info] : mem_handle_infos) {
        int unmap_status =
            nvshmemt_gpu_cpu_unmap(&s->memory.gpu_cpu_mapping_state, &info->cpu_mapping);
        if (unmap_status != 0) {
            NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping final cleanup failed status=%d",
                                 unmap_status);
            status = NVSHMEMX_ERROR_INTERNAL;
        }
    }
    nvshmemt_gpu_cpu_mapping_fini(&s->memory.gpu_cpu_mapping_state);
    return status;
}

int staged_gdrcopy_map(transport_staged_state_t* s, void* buf, size_t length, void* mapping_buf,
                       staged_mem_handle_info_t** info_out) {
    auto info = std::make_shared<staged_mem_handle_info_t>();
    info->ptr = buf;
    info->size = length;

    int status = 0;
    {
        std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
        status = nvshmemt_gpu_cpu_map(&s->memory.gpu_cpu_mapping_state, mapping_buf, length,
                                      NVSHMEMT_GPU_CPU_MAPPING_FLAG_NONE, &info->cpu_mapping);
    }
    if (status != 0) {
        NVSHMEMI_ERROR_PRINT(
            "[STAGED] CPU-mapping policy (%s) failed for %p len=%zu status=%d; "
            "memory registration failed",
            nvshmemt_gpu_cpu_mapping_backend_name(&s->memory.gpu_cpu_mapping_state), mapping_buf,
            length, status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    staged_mem_handle_info_t* stored_info = info.get();
    {
        std::lock_guard<std::mutex> lk(s->memory.mem_handle_mutex);
        s->memory.mem_handle_infos.emplace(stored_info, std::move(info));
    }
    if (info_out) {
        *info_out = stored_info;
    }
    return 0;
}

int staged_gdrcopy_unmap(transport_staged_state_t* s, staged_mem_handle_info_t* key) {
    std::shared_ptr<staged_mem_handle_info_t> info;
    {
        std::lock_guard<std::mutex> lk(s->memory.mem_handle_mutex);
        auto it = s->memory.mem_handle_infos.find(key);
        if (it == s->memory.mem_handle_infos.end()) {
            return 0;
        }
        info = std::move(it->second);
        s->memory.mem_handle_infos.erase(it);
    }

    /* No new copy can find the mapping now; wait for copies that still hold a reference.
     * use_count() is a relaxed load, so the fence orders their mapping accesses before unmap. */
    while (info.use_count() > 1) {
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    int status = 0;
    {
        std::lock_guard<std::mutex> lk(s->memory.gpu_cpu_mapping_mutex);
        status = nvshmemt_gpu_cpu_unmap(&s->memory.gpu_cpu_mapping_state, &info->cpu_mapping);
    }
    if (status != 0) {
        NVSHMEMI_ERROR_PRINT("[STAGED] GPU CPU mapping cleanup failed status=%d", status);
        std::lock_guard<std::mutex> lk(s->memory.mem_handle_mutex);
        s->memory.mem_handle_infos.emplace(key, std::move(info));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return 0;
}

/* Caller holds mem_handle_mutex. */
static std::shared_ptr<staged_mem_handle_info_t> staged_gdrcopy_find(transport_staged_state_t* s,
                                                                     const void* ptr,
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
    return it == s->memory.mem_handle_infos.end() ? nullptr : it->second;
}

/* CPU address of gpu_ptr through its mapping, or nullptr if it is not mapped. */
static void* staged_gdrcopy_cpu_address(const staged_mem_handle_info_t* info, const void* gpu_ptr) {
    if (!info || !info->cpu_mapping.mapped || !info->cpu_mapping.cpu_ptr) {
        return nullptr;
    }
    return static_cast<char*>(info->cpu_mapping.cpu_ptr) +
           (static_cast<const char*>(gpu_ptr) - static_cast<char*>(info->ptr));
}

int staged_gdrcopy_copy(transport_staged_state_t* s, void* dst, bool dst_on_gpu, const void* src,
                        bool src_on_gpu, size_t bytes, const char* label) {
    std::shared_ptr<staged_mem_handle_info_t> dst_info;
    std::shared_ptr<staged_mem_handle_info_t> src_info;
    {
        std::lock_guard<std::mutex> lk(s->memory.mem_handle_mutex);
        if (dst_on_gpu) {
            dst_info = staged_gdrcopy_find(s, dst, bytes);
        }
        if (src_on_gpu) {
            src_info = staged_gdrcopy_find(s, src, bytes);
        }
    }
    void* dst_cpu = dst_on_gpu ? staged_gdrcopy_cpu_address(dst_info.get(), dst) : dst;
    const void* src_cpu = src_on_gpu ? staged_gdrcopy_cpu_address(src_info.get(), src) : src;
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
