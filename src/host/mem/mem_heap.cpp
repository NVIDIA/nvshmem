/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>                                                        // for assert
#include <cuda.h>                                                          // for CUDA_...
#include <cuda_runtime.h>                                                  // for cudaFree
#include <driver_types.h>                                                  // for cudaH...
#include <ext/alloc_traits.h>                                              // for __all...
#include <stdint.h>                                                        // for uintp...
#include <stdio.h>                                                         // for size_t
#include <stdlib.h>                                                        // for calloc
#include <string.h>                                                        // for memset
#include <unistd.h>                                                        // for pid_t
#include <mutex>                                                           // for std::lock_guard
#include <algorithm>                                                       // for max
#include <iosfwd>                                                          // for std
#include <map>                                                             // for map
#include <memory>                                                          // for alloc...
#include <tuple>                                                           // for tuple
#include <typeinfo>                                                        // for type_...
#include <utility>                                                         // for pair
#include <vector>                                                          // for vector
#include "device_host/nvshmem_types.h"                                     // for nvshm...
#include "device_host/nvshmem_common.cuh"                                  // for nvshm...
#include "host/nvshmem_api.h"                                              // for nvshm...
#include "host/nvshmemx_api.h"                                             // for nvshm...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHM...
#include "non_abi/nvshmem_build_options.h"                                 // IWYU pragma: keep
#include "device_host_transport/nvshmem_common_transport.h"                // for g_elem_t
#include "internal/host/debug.h"                                           // for INFO
#include "internal/host/nvshmem_internal.h"                                // for nvshm...
#include "internal/common/error_codes_internal.h"                          // for NVSHM...
#include "internal/host/custom_malloc.h"                                   // for mspace
#include "internal/host/nvshmem_nvtx.hpp"                                  // for nvtx_...
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshm...
#include "internal/host/nvshmemi_handle_table.hpp"                         // for nvshm...
#include "internal/host/nvshmemi_heap_registration.hpp"                    // for nvshm...
#include "internal/host/nvshmemi_mem_transport.hpp"                        // for nvshm...
#include "internal/host/nvshmemi_team.h"                                   // for nvshm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshm...
#include "internal/host/shared_memory.h"                                   // for share...
#include "internal/host/util.h"                                            // for nvshm...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshm...
#include "internal/host_transport/cudawrap.h"                              // for CUPFN
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshm...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for nvshm...
#include "internal/host_transport/transport.h"                             // for nvshm..
#include "internal/host/nvshmemi_nvls_rsc.hpp"
#include "internal/host/scope_guard.h"
#include "internal/host/nvshmemi_nvls_observer.hpp"

#ifdef NVSHMEM_USE_DLMALLOC
#include "dlmalloc.h"
#endif

static_assert(sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE,
              "sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE");

namespace {
std::mutex &get_cs_mutex() {
    static std::mutex instance;
    return instance;
}
}  // namespace

struct nvshmemi_make_heap_result {
    std::unique_ptr<nvshmemi_symmetric_heap> heap;
    nvshmemi_symmetric_heap_vidmem_dynamic_vmm *vmm_heap = nullptr;
    nvshmemi_nvls_observer *nvls_obs = nullptr;
    int status = NVSHMEMX_SUCCESS;
};

int nvshmemi_bootstrap_aggregate_status(int local_status, int npes) {
    int status = NVSHMEMX_SUCCESS;
    std::vector<int> peer_statuses(npes, NVSHMEMX_SUCCESS);

    status = nvshmemi_boot_handle.allgather(&local_status, peer_statuses.data(),
                                            sizeof(local_status), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "allgather of heap operation status failed\n");

    for (const auto peer_status : peer_statuses) {
        if (peer_status != NVSHMEMX_SUCCESS) {
            return peer_status;
        }
    }

    return NVSHMEMX_SUCCESS;
}
namespace {

void nvshmemi_release_uncommitted_vmm_chunk(CUmemGenericAllocationHandle cumem_handle,
                                            char *buf_start, size_t size, bool handle_created,
                                            bool memory_mapped) {
    int status = CUDA_SUCCESS;

    if (memory_mapped) {
        status = CUPFN(nvshmemi_cuda_syms, cuMemUnmap((CUdeviceptr)buf_start, size));
        if (status != CUDA_SUCCESS) {
            NVSHMEMI_WARN_PRINT("cuMemUnmap failed while rolling back VMM heap allocation");
        }
    }

    if (handle_created) {
        status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(cumem_handle));
        if (status != CUDA_SUCCESS) {
            NVSHMEMI_WARN_PRINT("cuMemRelease failed while rolling back VMM heap allocation");
        }
    }
}
}  // namespace

long nvshmem_error = 0;

#define LE_IPC_HANDLE_TYPE CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC
/**
 * By OpenSHMEM spec standard, coll sync are not needed
 * if size == 0 or if ptr is NULL
 */
#define NVSHMEMI_IS_NO_ACTION_BY_SIZE(size) ((size) == 0)
#define NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr) ((ptr) == NULL)

/**
 * Global static variables references, shared by dependent classes
 */
std::vector<nvshmemi_shared_memory_info_t> nvshmemi_symmetric_heap_sysmem_static_shm::infos_;
nvshmemi_mem_remote_transport *nvshmemi_mem_remote_transport::remote_objref_;
nvshmemi_mem_p2p_transport *nvshmemi_mem_p2p_transport::p2p_objref_;

/* Returns the single concrete handle type to use for cuMem export/import operations.
 * When mem_handle_type_ is a combined bitmask (FABRIC | POSIX_FILE_DESCRIPTOR), selects
 * FABRIC when the MNNVL fabric is active on this PE, otherwise POSIX_FILE_DESCRIPTOR. */
CUmemAllocationHandleType nvshmemi_symmetric_heap::get_effective_import_handle_type(void) const {
    if ((mem_handle_type_ & CU_MEM_HANDLE_TYPE_FABRIC) &&
        (mem_handle_type_ & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
        return (p2p_ref_ && p2p_ref_->is_mnnvl_fabric()) ? CU_MEM_HANDLE_TYPE_FABRIC
                                                         : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }
    return mem_handle_type_;
}

std::map<pid_t, int> nvshmemi_symmetric_heap::get_p2p_proc_map() const {
    return p2p_ref_ ? p2p_ref_->get_proc_map() : std::map<pid_t, int>();
}

void nvshmemi_symmetric_heap::set_heap_registration(
    std::unique_ptr<nvshmemi_heap_registration> registration) {
    assert(registration != nullptr);
    heap_registration_ = std::move(registration);
}

namespace {
nvshmemi_make_heap_result make_heap_failure(int status) {
    nvshmemi_make_heap_result result;
    result.status = status;
    return result;
}

nvshmemi_make_heap_result make_vmm_symmetric_heap(const nvshmemi_heap_config &cfg,
                                                  bool enable_nvls) {
    nvshmemi_make_heap_result result;
    int status = NVSHMEMX_SUCCESS;
    auto heap = nvshmemi_symmetric_heap_vidmem_dynamic_vmm::create_reserved(cfg, &status);
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("nvshmem reserve VMM heap failed, status: %d\n", status);
        return make_heap_failure(NVSHMEMX_ERROR_INTERNAL);
    }

    auto *vmm = heap.get();
    if (enable_nvls) {
        auto nvls_obs = std::make_unique<nvshmemi_nvls_observer>(vmm, enable_nvls);
        result.nvls_obs = nvls_obs.get();
        vmm->register_observer(std::move(nvls_obs));
    }

    result.vmm_heap = vmm;
    result.heap = std::move(heap);
    return result;
}

std::unique_ptr<nvshmemi_symmetric_heap> make_static_heap_object(const nvshmemi_heap_config &cfg,
                                                                 int heap_kind, int *status) {
    assert(status != nullptr);
    *status = NVSHMEMX_SUCCESS;

    if (heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
        return nvshmemi_symmetric_heap_sysmem_static_shm::create_reserved(cfg, status);
    }
    if (heap_kind == NVSHMEMI_HEAP_KIND_VIDMEM) {
        return nvshmemi_symmetric_heap_vidmem_static_pinned::create_reserved(cfg, status);
    }

    *status = NVSHMEMX_ERROR_INVALID_VALUE;
    return nullptr;
}

nvshmemi_make_heap_result make_static_symmetric_heap(const nvshmemi_heap_config &cfg,
                                                     int heap_kind) {
    int status = NVSHMEMX_SUCCESS;
    auto heap = make_static_heap_object(cfg, heap_kind, &status);
    if (status == NVSHMEMX_ERROR_INVALID_VALUE) {
        NVSHMEMI_ERROR_PRINT(
            "Requested Heap Kind: %d(0-VIDMEM,1-SYSMEM,>3-INVALID), with VMM: %s\n", heap_kind,
            "No");
        return make_heap_failure(NVSHMEMX_ERROR_INVALID_VALUE);
    }
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("nvshmem reserve static heap failed, status: %d\n", status);
        return make_heap_failure(NVSHMEMX_ERROR_INTERNAL);
    }

    nvshmemi_make_heap_result result;
    result.heap = std::move(heap);
    return result;
}

nvshmemi_make_heap_result make_symmetric_heap(const nvshmemi_heap_config &cfg, bool is_vmm,
                                              int heap_kind, bool enable_nvls) {
    if (is_vmm) {
        return make_vmm_symmetric_heap(cfg, enable_nvls);
    }
    return make_static_symmetric_heap(cfg, heap_kind);
}
}  // namespace

int nvshmemi_init_symmetric_heap(nvshmemi_state_t *state, bool is_vmm, int heap_kind) {
    int status = NVSHMEMX_SUCCESS;

    if (state->heap_obj != nullptr) {
        return status;
    }

    state->vmm_heap = nullptr;
    state->nvls_obs = nullptr;

    const nvshmemi_heap_config cfg{state->mype, state->npes, state->npes_node, state->device_id};
    auto result = make_symmetric_heap(cfg, is_vmm, heap_kind, state->is_platform_nvls);
    status = result.status;
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem symmetric heap creation failed \n");

    state->heap_obj = result.heap.release();
    state->vmm_heap = result.vmm_heap;
    state->nvls_obs = result.nvls_obs;

    // Heap constructors initialize the p2p transport singleton; publish it to state.
    state->p2p_transport = nvshmemi_mem_p2p_transport::get_instance(state->mype, state->npes);

out:
    return status;
}

static nvshmemi_heap_registration_policy get_heap_registration_policy(
    const nvshmemi_state_t *state) {
    if (state->vmm_heap != nullptr) {
        return nvshmemi_heap_registration_policy::DYNAMIC_VMM;
    }

    if (nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
        return nvshmemi_heap_registration_policy::STATIC_SYSMEM_FULL_HEAP;
    }

    assert(nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_VIDMEM);
    return nvshmemi_heap_registration_policy::STATIC_VIDMEM_FULL_HEAP;
}

static nvshmemi_transport_view make_transport_view(const nvshmemi_state_t &state) {
    return {state.num_initialized_transports, state.transports, state.transport_bitmap,
            state.transport_map};
}

static nvshmemi_heap_registration_geometry make_heap_registration_geometry(
    const nvshmemi_symmetric_heap &heap) {
    return {heap.get_base(), heap.get_global_base(), heap.get_logical_heap_size(),
            heap.get_mem_granularity(), heap.get_log2_cumem_granularity()};
}

static nvshmemi_heap_registration_config make_heap_registration_config(
    nvshmemi_state_t &state, nvshmemi_symmetric_heap &heap) {
    assert(state.p2p_transport != nullptr);
    return {state.mype,
            state.npes,
            state.npes_node,
            make_heap_registration_geometry(heap),
            heap.get_effective_import_handle_type(),
            make_transport_view(state),
            *nvshmemi_mem_remote_transport::get_instance(),
            *state.p2p_transport,
            heap.get_local_pe_bases(),
            get_heap_registration_policy(&state)};
}

static std::unique_ptr<nvshmemi_handle_table> make_heap_handle_table(
    const nvshmemi_state_t &state, const nvshmemi_symmetric_heap &heap) {
    return std::make_unique<nvshmemi_handle_table>(
        heap.get_base(), heap.get_logical_heap_size(), heap.get_log2_cumem_granularity(),
        heap.get_mem_granularity(), state.num_initialized_transports, state.npes);
}

/** Finalizes heap setup and transport registration. */
int nvshmemi_setup_transport(nvshmemi_state_t *state) {
    int status = 0;
    assert(state != nullptr);
    assert(state->heap_obj != nullptr);
    auto &heap = *state->heap_obj;

    status = heap.setup_symmetric_heap();
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL, "setup_symmetric_heap failed \n");

    auto table = make_heap_handle_table(*state, heap);
    auto *table_ptr = table.get();

    const auto registration_cfg = make_heap_registration_config(*state, heap);
    auto registration = std::make_unique<nvshmemi_heap_registration>(table_ptr, registration_cfg);

    status = registration->setup();
    if (status != NVSHMEMX_SUCCESS) {
        const int teardown_status = registration->teardown();
        if (teardown_status != NVSHMEMX_SUCCESS) {
            NVSHMEMI_WARN_PRINT("heap registration teardown failed after setup failure: %d\n",
                                teardown_status);
        }
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "heap registration setup failed\n");

    heap.set_handle_table(std::move(table));
    state->handle_table = table_ptr;
    heap.set_heap_registration(std::move(registration));

out:
    return status;
}

void nvshmemi_fini_symmetric_heap(nvshmemi_state_t *state) {
    // State holds non-owning heap aliases.
    state->nvls_obs = nullptr;
    state->handle_table = nullptr;
    NVSHMEMU_HOST_PTR_DELETE(state->heap_obj);
    state->heap_obj = nullptr;
    state->vmm_heap = nullptr;
}

/**
 * nvshmemi_symmetric_heap common functions
 */
template <typename T>
int nvshmemi_symmetric_heap::is_symmetric(T value) {
    int status = 0;
    /* TODO: need to handle multi-threaded scenarios */
    if (!nvshmemi_options.ENABLE_ERROR_CHECKS) {
        return 0;
    }

    std::vector<T> scratch(cfg_.npes);
    status =
        nvshmemi_boot_handle.allgather(&value, scratch.data(), sizeof(T), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather in symmetry check failed \n");

    for (const auto &t : scratch) {
        status = (t == value);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_SYMMETRY, out, "symmetry check failed \n");
    }

out:
    return status;
}

void *nvshmemi_symmetric_heap::heap_malloc(size_t size) {
    if (nvshmemi_options.ENABLE_ALIGNED_MALLOC) {
        return heap_allocate(size, 0, mem_granularity_, NVSHMEMX_ALIGN);
    }
    return heap_allocate(size, 0, 0, NVSHMEMX_MALLOC);
}

void *nvshmemi_symmetric_heap::heap_calloc(size_t size, size_t count) {
    return heap_allocate(size, count, 0, NVSHMEMX_CALLOC);
}

void *nvshmemi_symmetric_heap::heap_align(size_t size, size_t alignment) {
    return heap_allocate(size, 0, alignment, NVSHMEMX_ALIGN);
}

void *nvshmemi_symmetric_heap::heap_allocate(size_t size, size_t count, size_t alignment,
                                             int type) {
    int status = 0;
    void *ptr = NULL;

    status = is_symmetric(size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "symmetry check for size failed\n");

    ptr = allocate_symmetric_memory(size, count, alignment, type);
    /* Don't inspect the ptr as caller will decide if its okay to have it to be NULL or non-NULL */

    INFO(NVSHMEM_MEM, "[%d] type: %s allocated %zu bytes, %zu count, %zu alignment ptr: %p",
         cfg_.mype, typeid(decltype(*this)).name(), size, count, alignment, ptr);

out:
    return ptr;
}

void nvshmemi_symmetric_heap::heap_deallocate(void *ptr) {
    heap_mspace_->deallocate(ptr);
    INFO(NVSHMEM_MEM, "[%d] freeing buf: %p type: %s", cfg_.mype, ptr,
         typeid(decltype(*this)).name());
    nvshmemi_update_device_state();
    return;
}

void nvshmemi_symmetric_heap::set_heap_size_attr(size_t mem_granularity, size_t *heapextra,
                                                 size_t *alignbytes, size_t *logmem_granularity) {
    *alignbytes = NVSHMEMI_MALLOC_ALIGNMENT;
    assert((mem_granularity & (mem_granularity - 1)) == 0);
    *logmem_granularity = nvshmemu_compute_log2(mem_granularity);
    *heapextra = G_BUF_SIZE + nvshmemi_get_teams_mem_requirement() + G_COALESCING_BUF_SIZE +
                 4 * (*alignbytes) +
                 20 * (*alignbytes);  // alignbytes, providing capacity for 2 allocations for
                                      // the library and 10 allocations for the user
}

int nvshmemi_symmetric_heap::cleanup_mspace(void) {
    if (heap_mspace_ != nullptr) {
        NVSHMEMU_HOST_PTR_DELETE(heap_mspace_);
    }
    if (mmap_mspace_ != nullptr) {
        NVSHMEMU_HOST_PTR_DELETE(mmap_mspace_);
    }

    return 0;
}

int nvshmemi_symmetric_heap::allgather_peer_base() {
    int status = NVSHMEMX_SUCCESS;

    // Base virtual address of heap_base for all PEs (needed for REMOTE)
    peer_heap_base_remote_.assign(cfg_.npes, nullptr);

    status =
        nvshmemi_boot_handle.allgather((void *)&heap_base_, (void *)peer_heap_base_remote_.data(),
                                       sizeof(void *), &nvshmemi_boot_handle);

    // Base virtual address of heap_base for my PE (needed for P2P)
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of heap base for all PE failed \n");

    peer_heap_base_p2p_.assign(cfg_.npes, nullptr);
    peer_heap_base_p2p_[cfg_.mype] = heap_base_;
out:
    if (status) {
        peer_heap_base_p2p_.clear();
        peer_heap_base_remote_.clear();
    }

    return status;
}

nvshmemi_symmetric_heap::~nvshmemi_symmetric_heap() = default;

/**
 * nvshmemi_symmetric_heap kind allocate/free memory functions
 */
int nvshmemi_symmetric_heap_vidmem_static_pinned::allocate_heap_memory() {
    int status = 0;
    status = cudaMalloc(&heap_base_, heap_size_);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "cuMemAlloc failed \n");
    reserved_heap_size_ = heap_size_;
out:
    return status;
}

int nvshmemi_symmetric_heap_sysmem_static_shm::allocate_heap_memory() {
    int status = 0;
    size_t shm_size = heap_size_ * nvshmemi_boot_handle.npes_node;
    int ret = snprintf(heap_name_, 100, "sysmem_symm_heap");
    if (ret < 0) {
        NVSHMEMI_ERROR_EXIT("snprintf failed\n");
    }

    if (nvshmemi_boot_handle.mype_node == 0) {
        if (shared_memory_create(heap_name_, shm_size, &heap_info_) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to create shared memory slab\n");
        }
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    if (nvshmemi_boot_handle.mype_node != 0) {
        if (shared_memory_open(heap_name_, shm_size, &heap_info_) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to open shared memory slab\n");
        }
    }

    atexit(nvshmemi_symmetric_heap_sysmem_static_shm::
               atexit_heap_handler); /* This forces sysmem shared memory heap to be only one */
    nvshmemi_symmetric_heap_sysmem_static_shm::infos_.push_back(heap_info_);
    /* Do first touch, for NUMA awareness */
    memset((char *)heap_info_.addr + nvshmemi_boot_handle.mype_node * heap_size_, 0, heap_size_);

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    CUDA_RUNTIME_CHECK(cudaHostRegister(heap_info_.addr, shm_size, cudaHostRegisterDefault));
    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&global_heap_base_, heap_info_.addr, 0));
    heap_base_ = (char *)global_heap_base_ + nvshmemi_boot_handle.mype_node * heap_size_;
    reserved_heap_size_ = shm_size;

    return status;
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::free_heap_memory(void *addr) {
    CUDA_RUNTIME_CHECK(cudaFree(addr));
    return NVSHMEMI_SUCCESS;
}

int nvshmemi_symmetric_heap_sysmem_static_shm::free_heap_memory(void *unused_addr
                                                                __attribute__((unused))) {
    CUDA_RUNTIME_CHECK(cudaHostUnregister(heap_info_.addr));
    shared_memory_close(heap_name_, &heap_info_);
    return NVSHMEMI_SUCCESS;
}

int nvshmemi_symmetric_heap_static::setup_mspace() {
    heap_mspace_ = new mspace(heap_base_, heap_size_);
    heap_mspace_->track_large_chunks(1);
    return 0;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::setup_mspace() {
    heap_mspace_ = new mspace(heap_base_, physical_internal_heap_size_);
    heap_mspace_->track_large_chunks(1);
    mmap_mspace_ = new mspace(heap_base_, physical_internal_heap_size_);
    mmap_mspace_->track_large_chunks(1);
    return 0;
}

/* Constructor for static/dynamic class */
nvshmemi_symmetric_heap_static::nvshmemi_symmetric_heap_static(nvshmemi_heap_config cfg) noexcept
    : nvshmemi_symmetric_heap(cfg) {
    set_p2p_transport(nvshmemi_mem_p2p_transport::get_instance(cfg.mype, cfg.npes));
    set_remote_transport(nvshmemi_mem_remote_transport::get_instance());
}

nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvshmemi_symmetric_heap_vidmem_dynamic_vmm(
    nvshmemi_heap_config cfg) noexcept
    : nvshmemi_symmetric_heap(cfg) {
    set_p2p_transport(nvshmemi_mem_p2p_transport::get_instance(cfg.mype, cfg.npes));
    set_remote_transport(nvshmemi_mem_remote_transport::get_instance());
    set_mem_handle_type((get_p2pref()->get_mem_handle_type()));
}

template <typename Heap>
std::unique_ptr<Heap> nvshmemi_symmetric_heap::create_reserved_impl(nvshmemi_heap_config cfg,
                                                                    int *status) {
    assert(status != nullptr);
    auto heap = std::make_unique<Heap>(cfg);
    nvshmemi_symmetric_heap *base_heap = heap.get();
    *status = base_heap->reserve_heap();
    if (*status != NVSHMEMX_SUCCESS) {
        return nullptr;
    }
    return heap;
}

std::unique_ptr<nvshmemi_symmetric_heap_vidmem_static_pinned>
nvshmemi_symmetric_heap_vidmem_static_pinned::create_reserved(nvshmemi_heap_config cfg,
                                                              int *status) {
    return create_reserved_impl<nvshmemi_symmetric_heap_vidmem_static_pinned>(cfg, status);
}

std::unique_ptr<nvshmemi_symmetric_heap_sysmem_static_shm>
nvshmemi_symmetric_heap_sysmem_static_shm::create_reserved(nvshmemi_heap_config cfg, int *status) {
    return create_reserved_impl<nvshmemi_symmetric_heap_sysmem_static_shm>(cfg, status);
}

std::unique_ptr<nvshmemi_symmetric_heap_vidmem_dynamic_vmm>
nvshmemi_symmetric_heap_vidmem_dynamic_vmm::create_reserved(nvshmemi_heap_config cfg, int *status) {
    return create_reserved_impl<nvshmemi_symmetric_heap_vidmem_dynamic_vmm>(cfg, status);
}

int nvshmemi_symmetric_heap_static::reserve_heap(void) {
    int status;
    size_t heapextra = 0, alignbytes = 0;
    mem_granularity_ = nvshmemi_options.CUMEM_GRANULARITY < NVSHMEMI_MAX_HANDLE_LENGTH
                           ? nvshmemi_options.CUMEM_GRANULARITY
                           : NVSHMEMI_MAX_HANDLE_LENGTH;
    set_heap_size_attr(mem_granularity_, &heapextra, &alignbytes, &(log2_mem_granularity_));
    heap_size_ = NVSHMEMU_ROUND_UP(nvshmemi_options.SYMMETRIC_SIZE + heapextra, mem_granularity_);
    physical_internal_heap_size_ = 0;

    bool data =
        true; /*A boolean attribute which when set, ensures that synchronous memory operations
                    initiated on the region of memory that ptr points to will always synchronize.*/

    allocate_heap_memory();

    status = CUPFN(
        nvshmemi_cuda_syms,
        cuPointerSetAttribute(&data, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)(heap_base_)));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                             out, "cuPointerSetAttribute failed \n");

    INFO(NVSHMEM_MEM,
         "[%d] heap type: %s heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
         cfg_.mype, typeid(this).name(), heap_base_, nvshmemi_options.SYMMETRIC_SIZE, heap_size_,
         heapextra);

    status = setup_mspace();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "memory space initialization failed \n");

    INFO(NVSHMEM_MEM, "[%d] heap type: %s cumem_granularity: %zu, log2_mem_granularity: %zu\n",
         cfg_.mype, typeid(decltype(*this)).name(), mem_granularity_, log2_mem_granularity_);

out:
    if (status) {
        free_heap_memory(heap_base_);
    }

    return status;
}

// Single logical endpoint for entire heap (1 per GPU)
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::reserve_unicast_endpoint(size_t size,
                                                                         bool counted_operations) {
    int status = 0;
    int status_endpoint_release = 0;
    CUdevice my_dev = 0;
    CUlogicalEndpointId le_id = 0;
    uint64_t le_bind_alignment_ = 0;  // granularity
    size_t le_max_size_ = 0;
    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&my_dev, cfg_.device_id));
    NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, "cuDeviceGet failed\n");
    // Create Unicast Endpoint and Attach it to leId.
    CUlogicalEndpointProp le_properties{};
    le_properties.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    le_properties.size = size;
    le_properties.unicast.device = my_dev;
    le_properties.ipcHandleTypes = LE_IPC_HANDLE_TYPE;
    le_properties.flags = counted_operations ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : 0;

    // check endpoint size
    status = CUPFN(nvshmemi_cuda_syms,
                   cuLogicalEndpointGetLimits(&le_bind_alignment_, &le_max_size_, &le_properties));
    NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                          "cuLogicalEndpointGetLimits failed\n");

    INFO(NVSHMEM_MEM,
         "[%d] Logical endpoint size: %zu, queried maximum size: %lu, queried bind alignment: %lu",
         cfg_.mype, size, le_max_size_, le_bind_alignment_);

    // For now, treating max size and bind alignment requirements as hard errors, alternatively we
    // can disable logical endpoints if these requirements are not met
    status = le_max_size_ < size;
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "size: %zu is greater than the maximum logical endpoint size: %lu. "
                          "Please adjust the MAX_MEMORY_PER_GPU\n",
                          size, le_max_size_);

    // mem granularity should be a multiple of bind alignment
    status = (mem_granularity_ % le_bind_alignment_) != 0;
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "mem granularity: %zu is not a multiple of bind alignment: %lu. Please "
                          "adjust the NVSHMEM_CUMEM_GRANULARITY\n",
                          mem_granularity_, le_bind_alignment_);
    le_granularity_ = mem_granularity_;

    status = (size % le_granularity_ != 0);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "size: %zu not aligned to le granularity %zu \n", size, le_granularity_);

    // unicast_endpoint_ids_with_flag_ should be empty
    status = unicast_endpoint_ids_with_flag_.size();
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "unicast_endpoint_ids_with_flag_ already allocated (size: %zu), "
                          "reserve_unicast_endpoint called multiple times\n",
                          unicast_endpoint_ids_with_flag_.size());
    unicast_endpoint_ids_with_flag_.resize(cfg_.npes, 0);
    le_id = 0;
    status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdReserve(&le_id, 1 /* count */));
    NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                          "cuLogicalEndpointIdReserve failed\n");

    status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointCreate(le_id, &le_properties));
    if (status != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuLogicalEndpointCreate failed, releasing logical endpoint id %u\n",
                             le_id);
        // release the logical endpoint id
        status_endpoint_release =
            CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdRelease(le_id, 1 /* count */));
        NVSHMEMI_NE_ERROR_RET(status_endpoint_release, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointIdRelease failed\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        return status;
    }
    // track the unicast endpoint id to release on cleanup
    unicast_endpoint_ids_with_flag_[cfg_.mype] = LE_ID_WITH_VALID_FLAG(le_id);
    INFO(NVSHMEM_MEM, "[%d] Reserved le id: %u", cfg_.mype, le_id);

    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::exchange_endpoints() {
    // export LE to other PEs and import LE  from other PEs
    // track the leIds in a vector similar to the P2P memory-handle sets

    int status = 0;
    int local_status = 0;
    int k = 0;
    int le_query_status = 0;
    CUlogicalEndpointId le_id = 0;
    nvshmemi_state_t *state = nvshmemi_state;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    std::vector<CUlogicalEndpointFabricHandle> local_le_handles_(state->num_initialized_transports);
    std::vector<CUlogicalEndpointFabricHandle> p2p_le_handles_(state->num_initialized_transports *
                                                               state->npes);

    // resize
    local_le_handles_.resize(state->num_initialized_transports);
    p2p_le_handles_.resize(state->num_initialized_transports * state->npes);

    NVSHMEMU_FOR_EACH_IF(
        i, state->num_initialized_transports,
        (NVSHMEMU_IS_BIT_SET(state->transport_bitmap, i) &&
         NVSHMEMI_TRANSPORT_IS_CAP(transports[i], state->mype,
                                   NVSHMEM_TRANSPORT_CAP_LOGICAL_ENDPOINT)),
        {
            INFO(NVSHMEM_MEM, "[%d] heap type: %s exporting logical endpoint %d", state->mype,
                 typeid(decltype(this)).name(), static_cast<int>(i));

            // Export the logical endpoint to LE fabric handle
            int export_status = CUPFN(
                nvshmemi_cuda_syms,
                cuLogicalEndpointExport(&local_le_handles_[i],
                                        PARSE_LE_ID(unicast_endpoint_ids_with_flag_[state->mype]),
                                        LE_IPC_HANDLE_TYPE));
            if (export_status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuLogicalEndpointExport failed for transport %d\n",
                                     static_cast<int>(i));
                local_status = NVSHMEMX_ERROR_INTERNAL;
            }
        });

    status = converge_unicast_endpoint_status(local_status);
    if (status != NVSHMEMX_SUCCESS) {
        destroy_unicast_endpoints();
        return status;
    }

    // Allgather LE fabric handles for remote connected PEs
    status = nvshmemi_boot_handle.allgather(
        (void *)local_le_handles_.data(), (void *)(p2p_le_handles_.data()),
        sizeof(CUlogicalEndpointFabricHandle) * state->num_initialized_transports,
        &nvshmemi_boot_handle);
    if (status != NVSHMEMX_SUCCESS) {
        destroy_unicast_endpoints();
        return NVSHMEMX_ERROR_INTERNAL;
    }

    // import the fabric handles into the logical endpoints
    k = (state->mype + 1) % state->npes;
    while (k != state->mype) {
        bool imported_endpoint = false;
        le_id = 0;

        NVSHMEMU_FOR_EACH_IF(
            j, state->num_initialized_transports,
            (NVSHMEMU_IS_BIT_SET(state->transport_map[state->mype * state->npes + k], j) &&
             !imported_endpoint && local_status == NVSHMEMX_SUCCESS &&
             NVSHMEMI_TRANSPORT_IS_CAP(state->transports[j], k,
                                       NVSHMEM_TRANSPORT_CAP_LOGICAL_ENDPOINT)),
            {
                int reserve_status =
                    CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdReserve(&le_id, 1 /* count */));
                if (reserve_status != CUDA_SUCCESS) {
                    NVSHMEMI_ERROR_PRINT("cuLogicalEndpointIdReserve failed for peer %d\n", k);
                    local_status = NVSHMEMX_ERROR_INTERNAL;
                    break;
                }

                int import_status =
                    CUPFN(nvshmemi_cuda_syms,
                          cuLogicalEndpointImport(
                              le_id, &p2p_le_handles_[k * state->num_initialized_transports + j],
                              LE_IPC_HANDLE_TYPE));
                if (import_status != CUDA_SUCCESS) {
                    NVSHMEMI_ERROR_PRINT("cuLogicalEndpointImport failed, releasing le_id %u\n",
                                         le_id);
                    int release_status =
                        CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdRelease(le_id, 1));
                    if (release_status != CUDA_SUCCESS) {
                        NVSHMEMI_ERROR_PRINT("cuLogicalEndpointIdRelease failed for id %u\n",
                                             le_id);
                    }
                    local_status = NVSHMEMX_ERROR_INTERNAL;
                    break;
                }

                // Set leId valid flag, if import is successful
                unicast_endpoint_ids_with_flag_[k] = LE_ID_WITH_VALID_FLAG(le_id);
                imported_endpoint = true;
                INFO(NVSHMEM_MEM,
                     "[%d] heap type: %s imported LE pe: %d, peer: %d, le id: %u, transport: %d",
                     state->mype, typeid(decltype(this)).name(), state->mype, k, le_id, j);
                break;  // as long as 1 transport is successful, we can break
            });
        if (!imported_endpoint) {
            NVSHMEMI_WARN_PRINT("[%d] No logical-endpoint-capable transport found for peer %d\n",
                                state->mype, k);
        }
        k = (k + 1) % state->npes;
    }

    status = converge_unicast_endpoint_status(local_status);
    if (status != NVSHMEMX_SUCCESS) {
        destroy_unicast_endpoints();
        return status;
    }

    // Wait till logical endpoints are ready
    for (int i = 0; i < state->npes; i++) {
        if (IS_VALID_LE_ID(unicast_endpoint_ids_with_flag_[i])) {
            le_query_status = 0;
            do {
                int query_status =
                    CUPFN(nvshmemi_cuda_syms,
                          cuLogicalEndpointQuery(PARSE_LE_ID(unicast_endpoint_ids_with_flag_[i]),
                                                 1 /* count */, &le_query_status));
                if (query_status != CUDA_SUCCESS) {
                    NVSHMEMI_ERROR_PRINT("cuLogicalEndpointQuery failed for peer %d\n", i);
                    local_status = NVSHMEMX_ERROR_INTERNAL;
                    break;
                }
            } while (le_query_status == 0);
        }
    }
    status = converge_unicast_endpoint_status(local_status);
    if (status != NVSHMEMX_SUCCESS) {
        destroy_unicast_endpoints();
    }
    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::converge_unicast_endpoint_status(int local_status) {
    std::vector<int> statuses(cfg_.npes, NVSHMEMX_SUCCESS);
    int status = nvshmemi_boot_handle.allgather(&local_status, statuses.data(),
                                                sizeof(local_status), &nvshmemi_boot_handle);
    if (status != NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_INTERNAL;
    }
    int converged_status = NVSHMEMX_SUCCESS;
    for (int peer_status : statuses) {
        if (peer_status == NVSHMEMX_SUCCESS) {
            continue;
        }
        if (peer_status != NVSHMEMX_ERROR_NOT_SUPPORTED) {
            return peer_status;
        }
        converged_status = NVSHMEMX_ERROR_NOT_SUPPORTED;
    }
    return converged_status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::bind_unicast_endpoint_memory(
    CUmemGenericAllocationHandle mem_handle, off_t heap_offset, off_t mem_offset, size_t size) {
    if (!le_unicast_enabled_) {
        return NVSHMEMX_SUCCESS;
    }
    int status =
        CUPFN(nvshmemi_cuda_syms,
              cuLogicalEndpointBindMem(PARSE_LE_ID(unicast_endpoint_ids_with_flag_[cfg_.mype]),
                                       cfg_.device_id, (unsigned long)heap_offset, mem_handle,
                                       (unsigned long)mem_offset, size, 0));
    return status == CUDA_SUCCESS ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::unbind_unicast_endpoint_memory(off_t heap_offset,
                                                                               size_t size) {
    if (!le_unicast_enabled_) {
        return NVSHMEMX_SUCCESS;
    }
    int status =
        CUPFN(nvshmemi_cuda_syms,
              cuLogicalEndpointUnbind(PARSE_LE_ID(unicast_endpoint_ids_with_flag_[cfg_.mype]),
                                      cfg_.device_id, (unsigned long)heap_offset, size));
    return status == CUDA_SUCCESS ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::destroy_unicast_endpoints() {
    int first_error = NVSHMEMX_SUCCESS;
    for (size_t i = 0; i < unicast_endpoint_ids_with_flag_.size(); i++) {
        if (!IS_VALID_LE_ID(unicast_endpoint_ids_with_flag_[i])) {
            continue;
        }
        int status =
            CUPFN(nvshmemi_cuda_syms,
                  cuLogicalEndpointDestroy(PARSE_LE_ID(unicast_endpoint_ids_with_flag_[i])));
        if (status != CUDA_SUCCESS && first_error == NVSHMEMX_SUCCESS) {
            first_error = NVSHMEMX_ERROR_INTERNAL;
        }
        status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdRelease(
                                               PARSE_LE_ID(unicast_endpoint_ids_with_flag_[i]), 1));
        if (status != CUDA_SUCCESS && first_error == NVSHMEMX_SUCCESS) {
            first_error = NVSHMEMX_ERROR_INTERNAL;
        }
    }
    INFO(NVSHMEM_MEM, "[%d] Released unicast logical endpoints", cfg_.mype);
    unicast_endpoint_ids_with_flag_.clear();
    return first_error;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::reserve_heap() {
    int status;
    size_t alignbytes = 0, heapextra = 0;
    CUmemAllocationProp prop = {};
    int p2p_npes = get_p2pref()->get_num_uc_ptr_connected_pes(cfg_.npes_node);

    set_cuda_mem_prop((void *)&prop, get_mem_handle_type());

    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemGetAllocationGranularity(&mem_granularity_, &prop,
                                                 CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemGetAllocationGranularity failed \n");
    mem_granularity_ = std::max(nvshmemi_options.CUMEM_GRANULARITY, mem_granularity_);
    mem_granularity_ = mem_granularity_ < NVSHMEMI_MAX_HANDLE_LENGTH ? mem_granularity_
                                                                     : NVSHMEMI_MAX_HANDLE_LENGTH;
    set_heap_size_attr(mem_granularity_, &heapextra, &alignbytes, &log2_mem_granularity_);
    INFO(NVSHMEM_MEM, "[%d] heap type: %s allocate_local_heap, heapextra = %lld", cfg_.mype,
         typeid(decltype(this)).name(), heapextra);
    heap_size_ = std::max(nvshmemi_options.MAX_MEMORY_PER_GPU, heapextra);
    heap_size_ = NVSHMEMU_ROUND_UP(heap_size_, mem_granularity_);

    if ((nvshmemi_options.LIMIT_PTR_P2P_ACCESS) ||
        ((p2p_npes * heap_size_) > NVSHMEMI_MAX_VA_SIZE)) {
        const bool has_cuda_clique_info = get_p2pref()->has_cuda_clique_info();
        const auto &unicast_pointer_pes = get_p2pref()->get_uc_ptr_connected_pes();
        const auto &restricted_pointer_pes =
            has_cuda_clique_info ? unicast_pointer_pes : get_p2pref()->get_mc_ptr_connected_pes();
        const size_t restricted_pointer_pe_count =
            std::count(restricted_pointer_pes.begin(), restricted_pointer_pes.end(), uint8_t{1});
        // Limit number of PEs mapped to VA
        if ((p2p_npes * heap_size_) > NVSHMEMI_MAX_VA_SIZE) {
            NVSHMEMI_WARN_PRINT(
                "[%d] Mapping %d p2p PEs would exceed maximum VA space (%lld bytes). "
                "Limiting pointer access to PEs within the %s.\n",
                cfg_.mype, p2p_npes, NVSHMEMI_MAX_VA_SIZE,
                has_cuda_clique_info ? "CUDA unicast-pointer clique" : "same rack");
        } else if (nvshmemi_options.LIMIT_PTR_P2P_ACCESS) {
            NVSHMEMI_WARN_PRINT(
                "[%d] LIMIT_PTR_P2P_ACCESS is set. "
                "Limiting pointer access to PEs within the %s.\n",
                cfg_.mype, has_cuda_clique_info ? "CUDA unicast-pointer clique" : "same rack");
        }

        status = (has_cuda_clique_info || nvshmemi_options.MNNVL_OVERRIDE_MC_CLIQUE_ID) ? 0 : 1;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Restricting pointer access requires CUDA fabric clique discovery "
                              "or MNNVL_OVERRIDE_MC_CLIQUE_ID to be set to true \n");

        // CUDA discovery supplies the unicast-pointer domain. In legacy discovery mode, the
        // optional override supplies the rack-local approximation.
        status = (restricted_pointer_pe_count * heap_size_ > NVSHMEMI_MAX_VA_SIZE);
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "Mapping the restricted PE domain (count: %zu) will exceed maximum VA space: %lld \n",
            restricted_pointer_pe_count, NVSHMEMI_MAX_VA_SIZE);

        get_p2pref()->update_uc_ptr_connected_pes(restricted_pointer_pes);

        // Update p2p_npes to reflect the restricted unicast-pointer domain.
        p2p_npes = get_p2pref()->get_num_uc_ptr_connected_pes(cfg_.npes_node);
    }

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    status = check_logical_endpoint_support();
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "check logical endpoint support failed\n");
#endif

    physical_internal_heap_size_ = 0;
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemAddressReserve((CUdeviceptr *)&global_heap_base_, p2p_npes * heap_size_,
                                       alignbytes, (CUdeviceptr)NULL, 0));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemAddressReserve failed \n");
    heap_base_ = (void *)((uintptr_t)global_heap_base_);
    mmap_base_ = (char *)heap_base_ + heap_size_;
    status = setup_mspace();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "memory space initialization failed \n");

    INFO(NVSHMEM_MEM,
         "[%d] heap type: %s heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
         cfg_.mype, typeid(decltype(this)).name(), heap_base_, nvshmemi_options.SYMMETRIC_SIZE,
         heap_size_, heapextra);
    reserved_heap_size_ = p2p_npes * heap_size_;

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if (le_unicast_enabled_) {
        int local_status =
            le_counted_operations_supported_ ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
        status = converge_unicast_endpoint_status(local_status);
        if (status != NVSHMEMX_SUCCESS && status != NVSHMEMX_ERROR_NOT_SUPPORTED) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "converge counted-operation capability failed\n");
        }

        const bool counted_operations = status == NVSHMEMX_SUCCESS;
        local_status = reserve_unicast_endpoint(heap_size_, counted_operations);
        status = converge_unicast_endpoint_status(local_status);
        if (status != NVSHMEMX_SUCCESS) {
            destroy_unicast_endpoints();
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "reserve unicast endpoint failed\n");
        }
        counted_operations_available_ = counted_operations;
    }
#else
    le_unicast_enabled_ = false;
    le_multicast_enabled_ = false;
#endif

out:
    return status;
}

/**
 * nvshmemi_symmetric_heap kind setup/cleanup heap functionality
 */
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::cleanup_symmetric_heap() {
    int status = 0;
    int teardown_status = 0;
    auto iter = get_mmapped_buf()->begin();
    INFO(NVSHMEM_MEM, "[%d] Entering %s::cleanup_symmetric_heap\n", cfg_.mype,
         typeid(decltype(this)).name());

    // cleanup mapped buffers
    while (iter != get_mmapped_buf()->end()) {
        status = unmap_mem(iter->first, iter->second);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, teardown_hooks,
                              "cleanup unmapped buffers failed \n");
        // the unmap_mem() erases entries from get_mmapped_buf, so resetting the iter
        iter = get_mmapped_buf()->begin();
    }

teardown_hooks:
    for (auto &obs : observers_) {
        int obs_status = obs->on_heap_teardown();
        if (teardown_status == 0 && obs_status != 0) {
            teardown_status = obs_status;
        }
    }
    /* Registration may be absent during partial initialization. */
    if (heap_registration_) {
        int reg_status = heap_registration_->teardown();
        if (teardown_status == 0 && reg_status != 0) {
            teardown_status = reg_status;
        }
    }
    if (status != 0) {
        goto out;
    }

    if (heap_base_ != NULL) {
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemUnmap((CUdeviceptr)heap_base_, physical_internal_heap_size_));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                                 out, "release memory failed for p2p on heap dynamic (my PE)\n");
    }

    NVSHMEMU_FOR_EACH(i, cumem_handles_.size()) {
        // Don't release mem handles corresponding to user buffer
        // which have been released as part of unmap
        if (is_cumem_handle_released(i)) {
            continue;
        }
        status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(std::get<0>(cumem_handles_[i])));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                                 out, "cuMemRelease failed \n");
    }
    cumem_handles_.clear();

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    // release logical endpoints after mmaped memory has been unmapped
    if (le_unicast_enabled_) {
        status = destroy_unicast_endpoints();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy unicast endpoints failed\n");
    }
#endif

    /* Release and Unmap memory for peer PE */
    if (!peer_heap_base_p2p_.empty()) {
        NVSHMEMU_FOR_EACH_IF(
            i, cfg_.npes, ((int)i != cfg_.mype) && peer_heap_base_p2p_[i] != NULL, {
                INFO(NVSHMEM_MEM, "calling release_memory on buf: %p size: %zu\n",
                     peer_heap_base_p2p_[i], heap_size_);
                status = release_memory(peer_heap_base_p2p_[i], heap_size_);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release memory failed for p2p on heap dynamic (peer PE)\n");
            });
    }

    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemAddressFree((CUdeviceptr)global_heap_base_, reserved_heap_size_));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemAddressFree failed \n");

    nvshmemi_mem_p2p_transport::destroy_instance();
    nvshmemi_mem_remote_transport::destroy_instance();

    INFO(NVSHMEM_MEM, "[%d] Leaving %s::cleanup_symmetric_heap\n", cfg_.mype,
         typeid(decltype(this)).name());
out:
    if (status == 0 && teardown_status != 0) {
        status = teardown_status;
    }
    return status;
}

int nvshmemi_symmetric_heap_static::cleanup_symmetric_heap() {
    int status = 0;
    int teardown_status = 0;
    INFO(NVSHMEM_MEM, "[%d] Entering %s::cleanup_symmetric_heap\n", cfg_.mype,
         typeid(decltype(this)).name());

    /* Run teardown hooks before releasing memory. */
    for (auto &obs : observers_) {
        int obs_status = obs->on_heap_teardown();
        if (teardown_status == 0 && obs_status != 0) {
            teardown_status = obs_status;
        }
    }
    /* Registration may be absent during partial initialization. */
    if (heap_registration_) {
        int reg_status = heap_registration_->teardown();
        if (teardown_status == 0 && reg_status != 0) {
            teardown_status = reg_status;
        }
    }

    if (!peer_heap_base_p2p_.empty()) {
        status = free_heap_memory(peer_heap_base_p2p_[cfg_.mype]);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "free_heap_memory failed \n");

        NVSHMEMU_FOR_EACH_IF(
            i, cfg_.npes, ((int)i != cfg_.mype) && peer_heap_base_p2p_[i] != NULL, {
                INFO(NVSHMEM_MEM, "calling release_memory on buf: %p \n", peer_heap_base_p2p_[i]);
                status = release_memory(peer_heap_base_p2p_[i]);
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release memory failed for p2p on heap static\n");
            });
    }

    INFO(NVSHMEM_MEM, "[%d] Leaving %s::cleanup_symmetric_heap\n", cfg_.mype,
         typeid(decltype(this)).name());

    nvshmemi_mem_p2p_transport::destroy_instance();
    nvshmemi_mem_remote_transport::destroy_instance();

out:
    if (status == 0 && teardown_status != 0) {
        status = teardown_status;
    }
    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::setup_symmetric_heap() {
    int status = 0;

    status = allgather_peer_base();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allgather PEs peer_base values\n");

    // perform exchange of endpoints
    if (le_unicast_enabled_) {
        status = exchange_endpoints();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "exchange endpoints failed\n");
        INFO(NVSHMEM_MEM, "[%d] Exchanged unicast endpoints\n", cfg_.mype);
    }
out:
    return status;
}

int nvshmemi_symmetric_heap_static::setup_symmetric_heap(void) {
    int status = 0;

    status = allgather_peer_base();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allgather PEs peer_base values\n");

out:
    if (status) {
        cleanup_symmetric_heap();
        if (heap_size_) {
            free_heap_memory(heap_base_);
        }
    }

    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::release_memory(void *buf, size_t size) {
    int status = 0;
    status = CUPFN(nvshmemi_cuda_syms, cuMemUnmap((CUdeviceptr)buf, size));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE,
                             out, "cuMemUnmap failed with error %d \n", status);
out:
    return status;
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::release_memory(void *buf, size_t /*size*/) {
    int status = 0;
    status = cudaIpcCloseMemHandle(buf);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "cudaIpcCloseMemHandle failed with error %d \n", status);
out:
    return status;
}

int nvshmemi_symmetric_heap_sysmem_static_shm::release_memory(void * /*buf*/, size_t /*size*/) {
    return 0; /** This is a NOOP for linux sysmem shared memory as entire memory is unmapped from
                 all PEs at cleanup time, so there is no step needed to release buffer range */
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_setup_multicast_endpoint(nvshmemi_team_t *team,
                                                                              uint64_t mem_size) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    int status = 0;
    int le_query_status = 0;
    if (!le_multicast_enabled_) {
        team->mc_leid_with_flag = 0;
        return 0;
    }
    CUlogicalEndpointId le_multicast_id = 0;
    CUlogicalEndpointFabricHandle le_multicast_fabric_handle{};

    CUlogicalEndpointProp le_multicast_properties{};
    le_multicast_properties.type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
    le_multicast_properties.size = mem_size;
    le_multicast_properties.multicast.numDevices = team->size;
    le_multicast_properties.ipcHandleTypes = LE_IPC_HANDLE_TYPE;

    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) {
        return 0;
    }

    // Keep the endpoint invalid until clique validation and endpoint setup both succeed.
    team->mc_leid_with_flag = 0;
    if (get_p2pref()->has_cuda_clique_info()) {
        for (int team_pe = 0; team_pe < team->size; team_pe++) {
            const int world_pe = nvshmemi_team_pe(team, team_pe);
            if (!get_p2pref()->is_mc_le_connected_pe(world_pe)) {
                INFO(NVSHMEM_TEAM,
                     "Multicast logical endpoints are unavailable for team %d because world PE %d "
                     "is outside its multicast logical-endpoint clique",
                     team->team_idx, world_pe);
                return 0;
            }
        }
    }

    status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdReserve(&le_multicast_id, 1 /* count */));
    NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                          "cuLogicalEndpointIdReserve for multicast team: %d failed\n",
                          team->team_idx);

    auto *nvls_obs = nvshmemi_state->nvls_obs;
    assert(nvls_obs != nullptr);

    bool le_multicast_created = false;
    auto le_multicast_cleanup = make_scope_guard([&]() noexcept {
        if (le_multicast_created) {
            int cleanup_status =
                CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointDestroy(le_multicast_id));
            if (cleanup_status != CUDA_SUCCESS) {
                NVSHMEMI_WARN_PRINT("cuLogicalEndpointDestroy failed for multicast id: %u\n",
                                    le_multicast_id);
            }
        }

        int cleanup_status =
            CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointIdRelease(le_multicast_id, 1 /* count */));
        if (cleanup_status != CUDA_SUCCESS) {
            NVSHMEMI_WARN_PRINT("cuLogicalEndpointIdRelease failed for multicast id: %u\n",
                                le_multicast_id);
        }
    });

    /* team PE0 will export MC endpoint */
    if (team->my_pe == 0) {
        status = CUPFN(nvshmemi_cuda_syms,
                       cuLogicalEndpointCreate(le_multicast_id, &le_multicast_properties));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointCreate for multicast failed\n");
        le_multicast_created = true;

        status =
            CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointExport(&le_multicast_fabric_handle,
                                                              le_multicast_id, LE_IPC_HANDLE_TYPE));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointExport for multicast failed\n");

        status = nvls_obs->nvls_broadcast_heap_handle_by_team(
            (char *)&le_multicast_fabric_handle, sizeof(le_multicast_fabric_handle), team);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "Broadcasting exported multicast endpoint for pe %d failed\n",
                              team->my_pe);

    } else {
        status = nvls_obs->nvls_broadcast_heap_handle_by_team(
            (char *)&le_multicast_fabric_handle, sizeof(le_multicast_fabric_handle), team);
        NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                              "Broadcasting exported multicast endpoint for pe %d failed\n",
                              team->my_pe);

        status = CUPFN(nvshmemi_cuda_syms,
                       cuLogicalEndpointImport(le_multicast_id, &le_multicast_fabric_handle,
                                               LE_IPC_HANDLE_TYPE));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointImport for multicast failed \n");
        le_multicast_created = true;
    }
    status = nvls_obj->subscribe_multicast_endpoint(le_multicast_id);
    NVSHMEMI_NZ_ERROR_RET(status, NVSHMEMX_ERROR_INTERNAL,
                          "Subscribing multicast endpoint for team: %d failed\n", team->team_idx);

    // Wait till endpoint is ready
    do {
        status = CUPFN(nvshmemi_cuda_syms,
                       cuLogicalEndpointQuery(le_multicast_id, 1 /* count */, &le_query_status));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointQuery for multicast id: %u failed\n",
                              le_multicast_id);
    } while (le_query_status == 0);

    team->mc_leid_with_flag = LE_ID_WITH_VALID_FLAG(le_multicast_id);
    nvshmem_barrier(team->team_idx);
    le_multicast_cleanup.dismiss();
    return status;
#else
    team->mc_leid_with_flag = 0;
    NVSHMEMI_UNUSED_ARG(mem_size);
    return 0;
#endif
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_setup_multicast_endpoint_by_team(
    nvshmemi_team_t *team) {
    return nvls_setup_multicast_endpoint(team, heap_size_);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_bind_multicast_endpoint(
    nvshmemi_team_t *team, CUmemGenericAllocationHandle mem_handle, off_t le_offset,
    off_t handle_offset, size_t size) {
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) {
        return 0;
    }
    assert(handle_offset == 0);
    // Return if the multicast endpoint is not valid
    if (!IS_VALID_LE_ID(team->mc_leid_with_flag)) {
        return 0;
    }

    status = CUPFN(nvshmemi_cuda_syms,
                   cuLogicalEndpointBindMem(PARSE_LE_ID(team->mc_leid_with_flag),
                                            nvls_obj->get_current_dev(), (unsigned long)(le_offset),
                                            mem_handle, handle_offset, size, /* flags = */ 0));
    NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                          "cuLogicalEndpointBindMem failed at offset: %lu for size: %zu\n",
                          le_offset, size);

    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_unbind_multicast_endpoint(
    nvshmemi_team_t *team, off_t le_offset, size_t size) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);

    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) {
        return 0;
    }
    if (!le_multicast_enabled_ || !IS_VALID_LE_ID(team->mc_leid_with_flag)) {
        return 0;
    }

    status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointUnbind(PARSE_LE_ID(team->mc_leid_with_flag),
                                                               nvls_obj->get_current_dev(),
                                                               (unsigned long)(le_offset), size));
    NVSHMEMI_NE_ERROR_RET(
        status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
        "cuLogicalEndpointUnbind for multicast endpoint failed at offset: %lu for size: %zu\n",
        le_offset, size);

    return status;
#else
    (void)team;
    (void)le_offset;
    (void)size;
    return 0;
#endif
}

void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::print_cumem_handles(void) const {
    NVSHMEMU_FOR_EACH(i, get_cumem_handle_size()) {
        INFO(NVSHMEM_MEM,
             "[%d] UC mem_handle: %lld mc_offset: %ld mmap_offset: %ld mmap_size: %zu\n", cfg_.mype,
             get_cumem_handle_ptr(i), get_cumem_handle_alloc_offset(i),
             get_cumem_handle_mmap_offset(i), get_cumem_handle_mmap_size(i));
    }
    return;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_destroy_multicast_endpoint_by_team(
    nvshmemi_team_t *team) {
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) {
        return status;
    }

    if (IS_VALID_LE_ID(team->mc_leid_with_flag)) {
        CUlogicalEndpointId multicast_endpoint_id = PARSE_LE_ID(team->mc_leid_with_flag);
        status = CUPFN(nvshmemi_cuda_syms, cuLogicalEndpointDestroy(multicast_endpoint_id));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointDestroy failed for le id: %u\n",
                              multicast_endpoint_id);

        status = CUPFN(nvshmemi_cuda_syms,
                       cuLogicalEndpointIdRelease(multicast_endpoint_id, 1 /* count */));
        NVSHMEMI_NE_ERROR_RET(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                              "cuLogicalEndpointIdRelease failed for le id: %u\n",
                              multicast_endpoint_id);
        team->mc_leid_with_flag = 0;
    }
    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_physical_memory_to_heap(size_t size) {
    size = ((size + mem_granularity_ - 1) / mem_granularity_) * mem_granularity_;
    INFO(NVSHMEM_MEM, "type: %s adding new physical backing of size %zu bytes",
         typeid(decltype(this)).name(), size);

    CUmemGenericAllocationHandle cumem_handle = {};
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};
    char *buf_start = nullptr;
    off_t heap_offset = 0;
    off_t mmap_offset =
        0; /* CUDA doesn't support non-zero mem_offset of a UC mem handle, so force to 0 */
    int status = NVSHMEMX_SUCCESS;
    bool handle_created = false;
    bool memory_mapped = false;
    size_t observers_notified = 0;
#ifdef NVSHMEM_CFT_HANDLES_SUPPORT
    bool le_bound = false;
#endif
    bool heap_owns_allocation = false;
    bool cleanup_heap = false;
    set_cuda_mem_prop((void *)&prop, get_mem_handle_type());

    status = ((physical_internal_heap_size_ + get_mmap_allocated_range() + size) >= heap_size_);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, local_done,
                          "Not enough space for allocating memory\n");

    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cfg_.device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    assert(size % mem_granularity_ == 0);
    assert(mem_granularity_ <= NVSHMEMI_MAX_HANDLE_LENGTH);

    buf_start = (char *)heap_base_ + physical_internal_heap_size_;

    // creating handle for the entire size
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemCreate(&cumem_handle, size, (const CUmemAllocationProp *)&prop, 0));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                             local_done, "cuMemCreate failed \n");
    handle_created = true;

    heap_offset = (off_t)(physical_internal_heap_size_);

    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemMap((CUdeviceptr)buf_start, size, mmap_offset, cumem_handle, 0));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                             local_done, "cuMemMap failed \n");
    memory_mapped = true;

    status = CUPFN(nvshmemi_cuda_syms, cuMemSetAccess((CUdeviceptr)buf_start, size,
                                                      (const CUmemAccessDesc *)&access, 1));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                             local_done, "cuMemSetAccess failed \n");

local_done:
    status = nvshmemi_bootstrap_aggregate_status(status, cfg_.npes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "VMM heap allocation failed on at least one PE\n");

    for (auto &obs : observers_) {
        status = obs->on_chunk_mapped((nvshmem_mem_handle_t *)&cumem_handle, (off_t)(heap_offset),
                                      mmap_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, observer_done,
                              "on_chunk_mapped failed\n");
        ++observers_notified;
    }

observer_done:
    status = nvshmemi_bootstrap_aggregate_status(status, cfg_.npes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "heap observer chunk map failed on at least one PE\n");

    // Bind Device Memory at unicast endpoint Offset.
#ifdef NVSHMEM_CFT_HANDLES_SUPPORT
    if (le_unicast_enabled_) {
        status = bind_unicast_endpoint_memory(cumem_handle, heap_offset, 0, size);
        le_bound = (status == NVSHMEMX_SUCCESS);
        if (!le_bound) {
            NVSHMEMI_ERROR_PRINT("cuLogicalEndpointBindMem failed at offset: %lu for size: %zu\n",
                                 heap_offset, size);
        }
    }
    status = nvshmemi_bootstrap_aggregate_status(status, cfg_.npes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "logical endpoint bind failed on at least one PE\n");
#endif

    assert(heap_registration_ != nullptr);
    status = heap_registration_->register_vmm_chunk(
        (nvshmem_mem_handle_t *)&cumem_handle, (off_t)(heap_offset), size,
        nvshmemi_allocation_kind::INTERNAL, std::nullopt);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "register_vmm_chunk failed\n");

    cumem_handles_.push_back(
        std::make_tuple(cumem_handle, heap_offset /*mc_offset*/, mmap_offset, size, false));

    /* Extend the heap mspace and size. */
    heap_mspace_->add_new_chunk((char *)heap_base_ + physical_internal_heap_size_, size);
    physical_internal_heap_size_ += size;
    heap_owns_allocation = true;

    status = nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* Wait for all PEs to setup the new memory */
    if (status != NVSHMEMX_SUCCESS) {
        cleanup_heap = true;
    }
out:
    if (status) {
        print_cumem_handles();
        if (!heap_owns_allocation) {
#ifdef NVSHMEM_CFT_HANDLES_SUPPORT
            if (le_bound) {
                const int unbind_status = unbind_unicast_endpoint_memory(heap_offset, size);
                if (unbind_status != NVSHMEMX_SUCCESS) {
                    NVSHMEMI_WARN_PRINT(
                        "cuLogicalEndpointUnbind failed while rolling back VMM heap allocation");
                }
            }
#endif
            while (observers_notified > 0) {
                --observers_notified;
                const int unbind_status =
                    observers_[observers_notified]->on_chunk_unmapped(heap_offset, size);
                if (unbind_status != NVSHMEMX_SUCCESS) {
                    NVSHMEMI_WARN_PRINT(
                        "heap observer chunk unmap failed while rolling back VMM heap allocation");
                }
            }
            nvshmemi_release_uncommitted_vmm_chunk(cumem_handle, buf_start, size, handle_created,
                                                   memory_mapped);
        }
        if (cleanup_heap) {
            cleanup_symmetric_heap();
        }
    }
    return status;
}

void *nvshmemi_symmetric_heap::allocate_virtual_memory_from_mspace(size_t size, size_t count,
                                                                   size_t alignment, int type) {
    void *ptr = NULL;
    switch (type) {
        case NVSHMEMX_MALLOC:
            ptr = heap_mspace_->allocate(size);
            break;
        case NVSHMEMX_CALLOC:
            ptr = heap_mspace_->allocate_zeroed(count, size);
            break;
        case NVSHMEMX_ALIGN:
            ptr = heap_mspace_->allocate_aligned(alignment, size);
            break;
        default:
            return NULL;
    }

    return ptr;
}

void *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_symmetric_memory(size_t size,
                                                                            size_t count,
                                                                            size_t alignment,
                                                                            int type) {
    int status = 0;
    void *ptr = NULL;

    ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
    if ((size > 0) && (ptr == NULL)) {
        if (type == NVSHMEMX_CALLOC) {
            status = allocate_physical_memory_to_heap((count * size) + alignment);
        } else {
            status = allocate_physical_memory_to_heap(size + alignment);
        }
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "allocate_physical_memory_to_heap failed\n");
        ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
        /* Only update the device state when physical heap is allocated successfully */
        status = nvshmemi_update_device_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemi_update_device_state failed\n");
    }

out:
    return ptr;
}

void *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::mmap_mem(void *buf_ptr, size_t size,
                                                           void *pref_addr, int flags) {
    void *ptr = NULL;
    int status = 0;

    CUmemGenericAllocationHandle userAllocHandle;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access[2];
    int numa_id;
    CUdevice my_dev;
    char *buf_start;
    off_t heap_offset = 0;
    size_t register_size = 0;
    size_t remaining_size;
    void *curr_ptr, *curr_buf_ptr;
    unsigned int ptr_mem_type;
    unsigned long long access_flags;
    bool is_egm = false;
    size_t pref_off = 0;
    size_t adjusted_max_handle_len =
        mem_granularity_ * (NVSHMEMI_MAX_HANDLE_LENGTH / mem_granularity_);
    off_t mmap_offset =
        0; /* CUDA doesn't support non-zero mem_offset of a UC mem handle, so force to 0 */
    set_cuda_mem_prop((void *)&prop, get_mem_handle_type());

    status = check_user_buffer_for_mmap(buf_ptr, size, &ptr_mem_type);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mmap user buffer check failed\n");

    // Get Access attributes from user buffer
    // Memory type can be device (VMM) or host (for EGM)
    if (ptr_mem_type == CU_MEMORYTYPE_DEVICE) {
        access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[0].location.id = cfg_.device_id;
    } else if (ptr_mem_type == CU_MEMORYTYPE_HOST) {
        // EGM memory
        is_egm = true;
        access[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&my_dev, cfg_.device_id));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS,
                                 NVSHMEMX_ERROR_INVALID_VALUE, out, "cuDeviceGet failed\n");
        status = CUPFN(nvshmemi_cuda_syms,
                       cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS,
                                 NVSHMEMX_ERROR_INVALID_VALUE, out,
                                 "cuDeviceGetAttribute NUMA ID failed\n");
        access[0].location.id = numa_id;

        access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[1].location.id = cfg_.device_id;
    }
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemGetAccess(&access_flags, &access[0].location, (CUdeviceptr)buf_ptr));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE,
                             out, "cuMemGetAccess failed\n");

    access[0].flags = (CUmemAccess_flags_enum)access_flags;
    access[1].flags = (CUmemAccess_flags_enum)access_flags;
    INFO(NVSHMEM_MEM, "type: %s Setting access permissions for mmap buffer: %llu",
         typeid(decltype(this)).name(), access_flags);

    INFO(NVSHMEM_MEM, "type: %s mmaping user buffer of size %zu bytes",
         typeid(decltype(this)).name(), size);

    assert(mem_granularity_ <= NVSHMEMI_MAX_HANDLE_LENGTH);

    NVSHMEMI_CHECK_ERROR_JMP(flags != 0, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                             "Non-zero flags not supported\n");

    // pref_off cannot be between heap_base_ and heap_base_ + physical_internal_heap_size_
    // as this region is used for internal alloc (nvshmem_malloc)

    // check if pref_off is already a hole with sufficient size
    // i.e. between (mmap_base_ - mmap_allocated_range()) till end
    if (pref_addr != NULL) {
        if ((pref_addr >= heap_base_) && (pref_addr < ((char *)heap_base_ + heap_size_))) {
            pref_off = (char *)pref_addr - (char *)heap_base_;
            INFO(NVSHMEM_MEM, "type: %s mmap with preferred addr: %p",
                 typeid(decltype(this)).name(), pref_addr);
            ptr = mmap_mspace_->allocate_at_preferred_addr(((char *)heap_base_ + pref_off), size);
        } else {
            WARN("Preferred mmap address %p not within heap range: %p : %p", pref_addr, heap_base_,
                 (char *)heap_base_ + heap_size_);
        }
    }
    if (ptr != NULL) {
        assert(ptr >= ((char *)mmap_base_ - get_mmap_allocated_range()));
        buf_start = (char *)ptr;
        status = (buf_start != ((char *)heap_base_ + pref_off));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Preferred allocate returned %p but expected %p\n", buf_start,
                              (char *)heap_base_ + pref_off);

        INFO(NVSHMEM_MEM, "Found hole at preferred offset buf start: %p  off: %lu for %zu bytes",
             buf_start, pref_off, size);
    } else if ((pref_addr) && (pref_off > physical_internal_heap_size_) &&
               ((pref_off + size) <= (heap_size_ - get_mmap_allocated_range()))) {
        // check if pref_off is between internal alloc (physical_internal_heap_size_) and
        // start of mmaped region (mmap_base_ - get_mmap_allocated_range())
        buf_start = (char *)heap_base_ + pref_off;
        ptr = (void *)buf_start;
        INFO(NVSHMEM_MEM,
             "Found preferred offset buf start: %p  off: %lu for %zu bytes by extending mmap "
             "allocated range",
             buf_start, pref_off, size);

    } else {
        if (pref_addr != NULL) {
            INFO(NVSHMEM_MEM, "Could not register user buffer at preferred address: %p", pref_addr);
        }
        // check if there is a mmap_mspace free chunk to accomodate the request
        ptr = mmap_mspace_->allocate(size);
        if (ptr != NULL) {
            buf_start = (char *)ptr;
            INFO(NVSHMEM_MEM, "Found hole in mmap_mspace buf start: %p for %zu bytes", buf_start,
                 size);
        } else {
            // check to ensure external alloc (mmap) doesn't cross over to internal alloc
            // (nvshmem_malloc) Not enough space for mapping user buffer
            status =
                ((physical_internal_heap_size_ + get_mmap_allocated_range() + size) >= heap_size_);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Not enough space for mmaping buffer %p size %zu\n", buf_ptr,
                                  size);
            buf_start = (char *)mmap_base_ - get_mmap_allocated_range() - size;
            ptr = (void *)buf_start;
            INFO(NVSHMEM_MEM, "Need to extend mmap space. start ptr: %p for %zu bytes", buf_start,
                 size);
        }
    }

    // Tracking mmaped user buffer alias, needed for ibv_reg_mr_iova(), gdr_pin_buffer()
    // workarounds. See nvbug: 5072809 for more details on this. Must be done in steps of
    // adjusted_max_handle_len
    curr_ptr = ptr;
    curr_buf_ptr = buf_ptr;
    remaining_size = size;
    do {
        status = alias_va_map_.count((char *)curr_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Addr: %p already tracked\n",
                              (char *)curr_ptr);
        register_size =
            remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;
        alias_va_map_[curr_ptr] = curr_buf_ptr;
        if (is_egm) {
            egm_map_[curr_ptr] = register_size;
        }
        remaining_size -= register_size;
        curr_ptr = (char *)curr_ptr + register_size;
        curr_buf_ptr = (char *)curr_buf_ptr + register_size;
    } while (remaining_size > 0);

    /* pointer positions
     *     |<----------------------------- heap_size_ --------------------------->|
     *     |<- phy_int_heap_size_  ->|     |<- size ->|<- mmap_allocated_range_ ->|
     *     |-------------------------|-----|----------|---------------------------|
     * heap_base_                       buf_start                             mmap_base_
     */

    heap_offset = (off_t)(buf_start - (char *)heap_base_);

    status = CUPFN(nvshmemi_cuda_syms, cuMemRetainAllocationHandle(&userAllocHandle, buf_ptr));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE,
                             out, "Failed to get handle for buffer\n");

    // Track these handles, so that when new teams are created, we can bind them
    // last entry in tuple indicates that this is a user buffer handle
    // During cleanup, these handles SHOULD NOT be released
    cumem_handles_.push_back(
        std::make_tuple(userAllocHandle, heap_offset /*mc_offset*/, mmap_offset, size, false));
    mmap_handle_idx_in_cumem_handles_[ptr] = cumem_handles_.size() - 1;

    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemMap((CUdeviceptr)buf_start, size, mmap_offset, userAllocHandle, 0));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemMap user buffer failed \n");
    if (is_egm) {
        status =
            CUPFN(nvshmemi_cuda_syms, cuMemSetAccess((CUdeviceptr)buf_start, size, &access[0], 2));
    } else {
        status =
            CUPFN(nvshmemi_cuda_syms, cuMemSetAccess((CUdeviceptr)buf_start, size, &access[0], 1));
    }
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemSetAccess failed \n");

#ifdef NVSHMEM_CFT_HANDLES_SUPPORT
    if (le_unicast_enabled_) {
        status = is_egm;  // EGM not supported currently with logical endpoints
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "Logical endpoint binding of EGM buffers for user buffer is not currently supported\n");
        status = bind_unicast_endpoint_memory(userAllocHandle, heap_offset, mmap_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuLogicalEndpointBindMem failed at offset: %lu for size: %zu\n",
                              heap_offset, size);
        INFO(NVSHMEM_INIT, "cuLogicalEndpointBindMem done at offset: %lu for size: %zu\n",
             heap_offset, size);
    }
#endif

    /* A buffer below the allocated mmap range requires extending mmap_mspace. */
    if (buf_start < (char *)mmap_base_ - get_mmap_allocated_range()) {
        void *mmap_alloc;
        char *buf_end = buf_start + size;
        char *mmap_base_offset = (char *)mmap_base_ - get_mmap_allocated_range();
        size_t pref_mmap_void_size = 0;

        /*
         * A preferred address may leave a gap between the buffer end and the existing mmap range.
         * Compute it before allocate_at_preferred_addr(), then add it as free space.
         */
        if (buf_end < mmap_base_offset) {
            pref_mmap_void_size = mmap_base_offset - buf_end;
        }
        mmap_mspace_->add_new_chunk(buf_start, size);

        /* add_new_chunk() must preserve this chunk and return its exact address on allocation. */
        mmap_alloc = mmap_mspace_->allocate_at_preferred_addr(buf_start, size);
        status = (mmap_alloc == NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mmap mspace alloc failed\n");
        status = (mmap_alloc != buf_start);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mmap mspace alloc returning different address\n");
        if (pref_mmap_void_size) {
            /* Add the gap between buf_end and the previous mmap range as a free chunk. */
            INFO(NVSHMEM_MEM,
                 "[%d] %p %lu adding free chunk for void due to preferred mmap at %p, %lu\n",
                 cfg_.mype, buf_start, size, buf_end, pref_mmap_void_size);
            mmap_mspace_->add_new_chunk(buf_end, pref_mmap_void_size);
        }
    }

    for (auto &obs : observers_) {
        status = obs->on_chunk_mapped((nvshmem_mem_handle_t *)&userAllocHandle,
                                      (off_t)(heap_offset), mmap_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, observer_done,
                              "on_chunk_mapped failed\n");
    }

observer_done:
    status = nvshmemi_bootstrap_aggregate_status(status, cfg_.npes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "heap observer chunk map failed on at least one PE\n");

    assert(heap_registration_ != nullptr);
    status = heap_registration_->register_vmm_chunk(
        (nvshmem_mem_handle_t *)&userAllocHandle, (off_t)(heap_offset), size,
        nvshmemi_allocation_kind::EXTERNAL, get_mmap_allocated_range());
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "register_vmm_chunk failed\n");

    status = nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* Wait for all PEs to setup the new memory */
out:
    if (status) {
        print_cumem_handles();
        cleanup_symmetric_heap();
        return nullptr;
    }

    return ptr;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::unmap_mem(void *ptr, size_t size) {
    int status = 0;
    // if size is not a multiple of heap granularity, we round up the size, so
    // doing the same here else size check will fail
    if (size % mem_granularity_) {
        size = ((size + mem_granularity_ - 1) / mem_granularity_) * mem_granularity_;
    }
    INFO(NVSHMEM_MEM, "type: %s unmap_mem ptr: %p size: %zu\n", typeid(decltype(this)).name(), ptr,
         size);
    size_t adjusted_max_handle_len =
        mem_granularity_ * (NVSHMEMI_MAX_HANDLE_LENGTH / mem_granularity_);
    off_t heap_offset = (char *)ptr - (char *)heap_base_;
    void *curr_ptr = ptr;
    size_t addr_idx;
    size_t remaining_size = size;
    size_t register_size = 0;

    // check if ptr is already mmaped
    status = (size > get_mmap_allocated_range()) || (size == 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "invalid size\n");
    status = ((ptr > mmap_base_) || (ptr < ((char *)mmap_base_ - get_mmap_allocated_range())));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Invalid Address\n");
    status = (mmap_mspace_->checkInuse(ptr, size) == false);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Address,size not mmapped\n");

    // unbind memory from logical endpoint
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    if (le_unicast_enabled_ && !is_egm(ptr)) {
        status = unbind_unicast_endpoint_memory(heap_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuLogicalEndpointUnbind failed at offset: %lu for size: %zu\n",
                              heap_offset, size);
    }
#endif

    /* Clear alias and EGM tracking for each registration chunk. */
    do {
        register_size =
            remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;
        alias_va_map_.erase((char *)curr_ptr);
        egm_map_.erase((char *)curr_ptr);
        assert(remaining_size >= register_size);
        remaining_size -= register_size;
        curr_ptr = (char *)curr_ptr + register_size;
    } while (remaining_size);

    /* Notify observers before releasing handles and unmapping. */
    for (auto &obs : observers_) {
        status = obs->on_chunk_unmapped(((char *)ptr - (char *)heap_base_), size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "on_chunk_unmapped failed\n");
    }
    assert(heap_registration_ != nullptr);
    status = heap_registration_->unregister_vmm_chunk(((char *)ptr - (char *)heap_base_), size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "unregister_vmm_chunk failed\n");
    idx_in_mmap_mc_handles_.erase(ptr);

    status = CUPFN(nvshmemi_cuda_syms, cuMemUnmap((CUdeviceptr)ptr, size));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuMemUnMap failed for user buffer\n");

    /* Release and Unmap memory for peer PE */
    if (!peer_heap_base_p2p_.empty()) {
        NVSHMEMU_FOR_EACH_IF(
            i, cfg_.npes, ((int)i != cfg_.mype) && peer_heap_base_p2p_[i] != NULL, {
                INFO(NVSHMEM_MEM, "release_memory as part of unmap_mem buf: %p size: %zu\n",
                     peer_heap_base_p2p_[i], heap_size_);
                status = release_memory((char *)peer_heap_base_p2p_[i] + heap_offset, size);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release memory failed for p2p on heap dynamic (peer PE)\n");
            });
    }

    // memory handles of user buffer retrieved using cuMemRetainAllocationHandle() need to be
    // released to ensure that when user releases the handle, the memory is released
    if (mmap_handle_idx_in_cumem_handles_.count(ptr)) {
        addr_idx = mmap_handle_idx_in_cumem_handles_[ptr];
        status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(std::get<0>(cumem_handles_[addr_idx])));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                                 out, "cuMemRelease failed \n");
        // Mark cumem_handles_ entry as released
        cumem_handles_[addr_idx] = std::make_tuple(
            std::get<0>(cumem_handles_[addr_idx]), std::get<1>(cumem_handles_[addr_idx]),
            std::get<2>(cumem_handles_[addr_idx]), std::get<3>(cumem_handles_[addr_idx]), true);
        mmap_handle_idx_in_cumem_handles_.erase(ptr);
    }

    mmap_mspace_->deallocate(ptr);

    /* Sync the mmap range after deallocation. */
    if (get_handle_table()) {
        get_handle_table()->set_mmap_allocated_range(get_mmap_allocated_range());
    }

out:
    return status;
}

int nvshmemi_symmetric_heap::check_buffers_on_same_device(bool onGPU, void *ptr) {
    int status = 0;
    int buf_loc_id, loc_id;
    CUdevice gpu_dev;
    if (!nvshmemi_options.ENABLE_ERROR_CHECKS) {
        return 0;
    }

    std::vector<int> scratch(cfg_.npes);

    if (onGPU) {  // check for MPG case
        buf_loc_id = cfg_.device_id;
    } else {  // for same socket EGM case
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&gpu_dev, cfg_.device_id));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS,
                                 NVSHMEMX_ERROR_INVALID_VALUE, out, "cuDeviceGet failed\n");
        status =
            CUPFN(nvshmemi_cuda_syms,
                  cuDeviceGetAttribute(&buf_loc_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, gpu_dev));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS,
                                 NVSHMEMX_ERROR_INVALID_VALUE, out,
                                 "cuDeviceGetAttribute failed\n");
    }

    status = nvshmemi_boot_handle.allgather(&buf_loc_id, scratch.data(), sizeof(int),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather in check_buffers_on_same_device failed \n");

    loc_id = scratch[0];
    for (int i = 1; i < cfg_.npes_node; i++) {
        status = (scratch[i] == loc_id) ? 1 : 0;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "Memory buffers allocated are on same device, disable NVLS "
                              "(NVSHMEM_DISABLE_NVLS=1) if user buffer %p on GPU or "
                              "allocate user buffer %p from distinct NUMA sockets\n",
                              ptr, ptr);
    }

out:
    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::check_logical_endpoint_support() {
    int status = 0;
    int le_attr_ = 0;
    CUdevice my_dev;
    bool is_mem_handle_fabric;

#if CUDART_VERSION < 13030
    NVSHMEMI_WARN_PRINT(
        "[%d] Logical endpoint support is not available on this CUDA version (%d)\n", cfg_.mype,
        CUDART_VERSION);
    le_unicast_enabled_ = false;
    le_multicast_enabled_ = false;
    le_counted_operations_supported_ = false;
    return status;
#endif

    if (!nvshmemi_options.ENABLE_LOGICAL_ENDPOINT) {
        le_unicast_enabled_ = false;
        le_multicast_enabled_ = false;
        le_counted_operations_supported_ = false;
        INFO(NVSHMEM_MEM, "[%d] Logical endpoint support is disabled by environment variable\n",
             cfg_.mype);
        return status;
    }

    is_mem_handle_fabric = (get_mem_handle_type() == CU_MEM_HANDLE_TYPE_FABRIC);

    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&my_dev, cfg_.device_id));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuDeviceGet failed for device %d\n", cfg_.device_id);

    // check device attribute for logical endpoint unicast support
    status = CUPFN(nvshmemi_cuda_syms,
                   cuDeviceGetAttribute(
                       &le_attr_, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED, my_dev));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuDeviceGetAttribute Logical Endpoint Unicast Supported failed\n");

    le_unicast_enabled_ = (is_mem_handle_fabric && le_attr_);

    // Counted operations require both usable unicast endpoints and device support.
    status =
        CUPFN(nvshmemi_cuda_syms,
              cuDeviceGetAttribute(
                  &le_attr_, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED, my_dev));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuDeviceGetAttribute Logical Endpoint Counted Ops Supported failed\n");

    le_counted_operations_supported_ = (le_unicast_enabled_ && le_attr_);

    if (!nvshmemi_options.DISABLE_NVLS) {
        // check device attribute for logical endpoint multicast support
        status =
            CUPFN(nvshmemi_cuda_syms,
                  cuDeviceGetAttribute(
                      &le_attr_, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED, my_dev));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuDeviceGetAttribute Logical Endpoint Multicast Supported failed\n");

        le_multicast_enabled_ = (is_mem_handle_fabric && le_attr_);
    }

    INFO(NVSHMEM_MEM,
         "Logical endpoint support status: unicast %d, multicast %d, counted operations %d\n",
         le_unicast_enabled_, le_multicast_enabled_, le_counted_operations_supported_);
out:
    return status;
}

/**
 * Validate a user buffer for symmetric registration via mmap.
 *
 * The buffer must have been allocated with cuMemCreate and its
 * requestedHandleTypes must include the effective handle type that
 * NVSHMEM will use for export/import (resolved via
 * get_effective_import_handle_type()). This allows buffers allocated
 * by external libraries (e.g. ncclMemAlloc on GB200) that request a
 * combined handle type mask (FABRIC | POSIX_FILE_DESCRIPTOR).
 */
static nvshmemx_status check_vmm_buffer_device(
    struct nvshmemi_cuda_fn_table *cuda_syms,
    [[maybe_unused]] const CUmemAllocationProp &alloc_prop, void *ptr, int expected_device_id) {
    unsigned int device_ordinal = 0;
#if CUDART_VERSION >= 13040
    if (alloc_prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN) {
        return (static_cast<int>(alloc_prop.location.localized.deviceId) != expected_device_id)
                   ? NVSHMEMX_ERROR_INVALID_VALUE
                   : NVSHMEMX_SUCCESS;
    }
#endif
    CUresult cu_status =
        CUPFN(cuda_syms, cuPointerGetAttribute(static_cast<void *>(&device_ordinal),
                                               CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                               reinterpret_cast<CUdeviceptr>(ptr)));
    if (cu_status != CUDA_SUCCESS) {
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return (static_cast<int>(device_ordinal) != expected_device_id) ? NVSHMEMX_ERROR_INVALID_VALUE
                                                                    : NVSHMEMX_SUCCESS;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::check_user_buffer_for_mmap(
    void *ptr, size_t &size, unsigned int *ptr_mem_type) {
    int status = 0;
    int cuMemRelease_status = 0;
    unsigned int ptrAttr;
    size_t userAllocGran;
    CUmemAllocationProp userAllocProp;
    CUmemGenericAllocationHandle userAllocHandle;

    status = (size == 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_SYMMETRY, return_out, "size argument is zero\n");

    status = is_symmetric(size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_SYMMETRY, return_out,
                          "symmetry check for size failed\n");

    status = size % mem_granularity_;
    NVSHMEMI_NZ_ERROR_JMP(
        status, NVSHMEMX_ERROR_INVALID_VALUE, return_out,
        "user buffer %p size %zu is not a multiple of heap granularity %zu. Please adjust the user "
        "buffer size or update NVSHMEM_CUMEM_GRANULARITY",
        ptr, size, mem_granularity_);

    // check if buffer (ptr) is allocated from cuMemCreate
    status = CUPFN(nvshmemi_cuda_syms, cuMemRetainAllocationHandle(&userAllocHandle, ptr));
    NVSHMEMI_CU_NE_ERROR_JMP(
        nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
        "Failed to get user alloc handle for buffer %p. Please check if buffer "
        "is allocated using CUDA VMM API\n",
        ptr);

    // Check if allocation is done for one of the supported types.
    status = CUPFN(nvshmemi_cuda_syms,
                   cuPointerGetAttribute((void *)&ptrAttr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         reinterpret_cast<CUdeviceptr>(ptr)));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "Failed to get pointer attribute of user buffer\n");
    *ptr_mem_type = ptrAttr;

    // Memory type can be device (VMM) or host (for EGM)
    status = !((ptrAttr == CU_MEMORYTYPE_DEVICE) || (ptrAttr == CU_MEMORYTYPE_HOST));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "user buffer not allocated in device or host(EGM) memory\n");

    // Get allocation properties before device check to handle localized memory nodes
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemGetAllocationPropertiesFromHandle(&userAllocProp, userAllocHandle));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to get allocation properties of user buffer %p\n", ptr);

    // check for buffer device ordinal
    if (ptrAttr == CU_MEMORYTYPE_DEVICE) {
        nvshmemx_status dev_status =
            check_vmm_buffer_device(nvshmemi_cuda_syms, userAllocProp, ptr, cfg_.device_id);
        NVSHMEMI_NZ_ERROR_JMP(dev_status, dev_status, out,
                              "user buffer %p not allocated in device %d\n", ptr, cfg_.device_id);
    }

    // check for MPG / same-socket EGM buffers
    // Currently, if EGM is allocated in same CPU socket and allocation property (location id)
    // is same between PEs, multicastBind fails (the buffers should be in different devices).
    // Same issue will happen with MPG. Ref to nvBug 5202270
    // We detect the case under debug and report it to user
    status = check_buffers_on_same_device(ptrAttr == CU_MEMORYTYPE_DEVICE, ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "Cannot register user buffer %p due to device aliasing\n", ptr);

    // Check if requestedHandleTypes includes the effective handle type that will be used
    // for export/import. When the heap has a combined bitmask (FABRIC | POSIX_FILE_DESCRIPTOR),
    // get_effective_import_handle_type() resolves to the single type actually used at runtime.
    // The user buffer must support at least that type.
    if (!(userAllocProp.requestedHandleTypes & get_effective_import_handle_type())) {
        NVSHMEMI_ERROR_PRINT(
            "user buffer %p requested handle type mask 0x%x doesn't include effective heap handle "
            "type 0x%x\n",
            ptr, userAllocProp.requestedHandleTypes, get_effective_import_handle_type());
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }

    // Get allocation granularity
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemGetAllocationGranularity(&userAllocGran, &userAllocProp,
                                                 CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "Failed to get allocation granularity of user buffer %p\n", ptr);

out:
    cuMemRelease_status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(userAllocHandle));
    if (!status) {
        status = cuMemRelease_status;
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, cuMemRelease_status, CUDA_SUCCESS,
                                 NVSHMEMX_ERROR_INTERNAL, return_out, "cuMemRelease failed \n");
    }
return_out:
    return status;
}

size_t nvshmemi_symmetric_heap_vidmem_dynamic_vmm::get_mmap_allocated_range() const {
    size_t range = 0;
    void *startInusePtr = mmap_mspace_->get_startInusePtr();
    if (startInusePtr) {
        range = (char *)mmap_base_ - (char *)startInusePtr;
    }
    return range;
}

bool nvshmemi_symmetric_heap_vidmem_dynamic_vmm::is_egm(void *addr) {
    if (egm_map_.count(addr)) {
        return true;
    }

    for (auto iter = egm_map_.begin(); iter != egm_map_.end(); ++iter) {
        if ((addr >= iter->first) && ((char *)addr < ((char *)iter->first + iter->second))) {
            return true;
        }
    }
    return false;
}

std::map<void *, size_t> *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::get_mmapped_buf() {
    return mmap_mspace_->get_inuse_chunks();
}

const std::map<void *, size_t> *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::get_mmapped_buf()
    const {
    const mspace *mmap_mspace = mmap_mspace_;
    return mmap_mspace->get_inuse_chunks();
}

void *nvshmemi_symmetric_heap_static::allocate_symmetric_memory(size_t size, size_t count,
                                                                size_t alignment, int type) {
    void *ptr = NULL;

    ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
    if ((count > 0 && type == NVSHMEMX_CALLOC) && (size > 0) && (ptr == NULL)) {
        NVSHMEMI_ERROR_EXIT(
            "nvshmem malloc failed (hint: check if total allocation has exceeded NVSHMEM "
            "symmetric size = %zu, NVSHMEM symmetric size can be increased using "
            "NVSHMEM_SYMMETRIC_SIZE environment variable) \n",
            nvshmemi_options.SYMMETRIC_SIZE);
    }

    return ptr;
}

extern "C" {

void nvshmemi_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }

    nvshmemi_state->heap_obj->heap_deallocate(ptr);
}

void *nvshmemi_malloc(size_t size) { return nvshmemi_state->heap_obj->heap_malloc(size); }

}  // extern "C"

void *nvshmem_malloc(size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    int ret = nvshmemi_check_state_and_init();
    if (ret) {
        nvshmem_error = 1;
        return ptr;
    }

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(size)) {
        return ptr;
    }

    ptr = nvshmemi_state->heap_obj->heap_malloc(size);
    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return ptr;
    }

    nvshmemi_barrier_all();

    return ptr;
}

void *nvshmem_calloc(size_t count, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    int ret = nvshmemi_check_state_and_init();
    if (ret) {
        nvshmem_error = 1;
        return ptr;
    }

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(count * size)) {
        return ptr;
    }

    ptr = nvshmemi_state->heap_obj->heap_calloc(size, count);
    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return ptr;
    }

    nvshmemi_barrier_all();

    return ptr;
}

void *nvshmem_align(size_t alignment, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    int ret = nvshmemi_check_state_and_init();
    if (ret) {
        nvshmem_error = 1;
        return ptr;
    }

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(size)) {
        return ptr;
    }

    ptr = nvshmemi_state->heap_obj->heap_align(size, alignment);
    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return ptr;
    }
    nvshmemi_barrier_all();

    return ptr;
}

void nvshmem_free(void *ptr) {
    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());

    NVSHMEMI_CHECK_INIT_STATUS();

    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return;
    }

    nvshmemi_barrier_all();

    nvshmemi_free(ptr);
}

void *nvshmemi_ptr(const void *ptr, int pe) {
    if (ptr == NULL) {
        return NULL;
    }
    /* nvshmem_ptr can be queried before init/finalize; return NULL instead of dereferencing
     * uninitialized global state. */
    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_initialized || nvshmemi_state == NULL ||
        nvshmemi_state->heap_obj == NULL || nvshmemi_device_state.heap_base == NULL) {
        return NULL;
    }
    if (pe >= 0 && pe < nvshmemi_state->npes && ptr >= nvshmemi_device_state.heap_base) {
        uintptr_t offset = (char *)ptr - (char *)nvshmemi_device_state.heap_base;

        if (offset < nvshmemi_device_state.heap_size) {
            void *peer_addr = nvshmemi_state->heap_obj->get_local_pe_bases()[pe];
            if (peer_addr != NULL) {
                peer_addr = (void *)((char *)peer_addr + offset);
            }
            return peer_addr;
        }
    }

    return NULL;
}

void *nvshmem_ptr(const void *ptr, int pe) { return nvshmemi_ptr(ptr, pe); }

void *nvshmemx_mc_ptr(nvshmem_team_t team, const void *ptr) {
    if (team < 0 || team >= nvshmemi_max_teams || nvshmemi_team_pool[team] == NULL) {
        return NULL;
    }
    uintptr_t offset = (char *)ptr - (char *)nvshmemi_device_state.heap_base;
    if (ptr >= nvshmemi_device_state.heap_base && offset < nvshmemi_device_state.heap_size) {
        nvls::nvshmemi_nvls_rsc *nvls =
            reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(nvshmemi_team_pool[team]->nvls_rsc);
        if (nvls == NULL) {
            return NULL;
        }
        void *mc_addr = nvls->get_mc_base();
        if (mc_addr != NULL) {
            mc_addr = (void *)((char *)mc_addr + offset);
        }
        return mc_addr;
    } else {
        return NULL;
    }
}

void *nvshmemx_buffer_register_symmetric(void *buf_ptr, size_t size, int flags) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    int ret = nvshmemi_check_state_and_init();
    if (ret) {
        nvshmem_error = 1;
        return ptr;
    }

    if (nvshmemi_state->vmm_heap == nullptr) {
        NVSHMEMI_ERROR_PRINT("Buffer registration requires dynamic VMM heap");
        return ptr;
    }
    ptr = nvshmemi_state->vmm_heap->mmap_mem(buf_ptr, size, NULL, flags);
    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return ptr;
    }
    nvshmemi_barrier_all();

    return ptr;
}

void *nvshmemx_buffer_register_symmetric_at_preferred_address(void *buf_ptr, size_t size,
                                                              void *preferred_addr, int flags) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    int ret = nvshmemi_check_state_and_init();
    if (ret) {
        nvshmem_error = 1;
        return ptr;
    }

    if (nvshmemi_state->vmm_heap == nullptr) {
        NVSHMEMI_ERROR_PRINT("Buffer registration requires dynamic VMM heap");
        return ptr;
    }
    ptr = nvshmemi_state->vmm_heap->mmap_mem(buf_ptr, size, preferred_addr, flags);
    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        return ptr;
    }
    nvshmemi_barrier_all();

    return ptr;
}

int nvshmemx_buffer_unregister_symmetric(void *ptr, size_t size) {
    int status = 0;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    std::lock_guard<std::mutex> cs_lock(get_cs_mutex());
    NVSHMEMI_CHECK_INIT_STATUS();

    if (nvshmemi_state->vmm_heap == nullptr) {
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    nvshmemi_barrier_all();

    status = nvshmemi_state->vmm_heap->unmap_mem(ptr, size);

    return status;
}
