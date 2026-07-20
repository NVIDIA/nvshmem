/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_SYMMETRIC_HEAP_HPP
#define NVSHMEMI_SYMMETRIC_HEAP_HPP

#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <map>
#include <memory>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "internal/host/nvshmemi_heap_registration.hpp"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/cudawrap.h"
#include "non_abi/nvshmemx_error.h"
#include "device_host/logical_endpoint_types.h"

// Forward declarations
class nvshmemi_mem_p2p_transport;
class nvshmemi_mem_remote_transport;
class nvshmemi_nvls_observer;
class nvshmemi_handle_table;

enum { NVSHMEMX_MALLOC = 0, NVSHMEMX_CALLOC, NVSHMEMX_ALIGN, NVSHMEMX_ALLOC_MAX };

struct nvshmemi_cumem_handle_info {
    CUmemGenericAllocationHandle handle;
    off_t alloc_offset;
    off_t mmap_offset;
    size_t mmap_size;
    bool released;
};

/**
 * Minimal configuration for heap construction.
 */
struct nvshmemi_heap_config {
    int mype;
    int npes;
    int npes_node;
    int device_id;
};

/** Observer interface for symmetric heap lifecycle events. */
class nvshmemi_heap_observer {
   public:
    virtual ~nvshmemi_heap_observer() = default;

    /** Called after a chunk is mapped and before transport registration. */
    virtual int on_chunk_mapped(nvshmem_mem_handle_t *handle, off_t mc_offset, off_t mmap_offset,
                                size_t size) = 0;

    /** Called before a chunk is unmapped. */
    virtual int on_chunk_unmapped(off_t mc_offset, size_t size) = 0;

    /** Called before heap teardown releases mappings or handles. */
    virtual int on_heap_teardown() = 0;
};

#define NVSHMEMI_SYMMETRIC_HEAP_OFFSET(base, off) (void *)((uint8_t *)(base) + off)

/**
 * This class manages symmetric heap per kind. Today, we support global heap kind for all teams. To
 * support multiple heap kinds, we must create multiple instances of this class for the desired
 * team. Class hierarchy is defined here to allow for sharing of common code as much as possible and
 * only bifurcating when there is a functional/behavior difference by memory kind.
 *
 *                              nvshmemi_symmetric_heap
 *                    -------------------------------------------------
 *                  |                                                   |
 *    nvshmemi_symmetric_heap_static                  vidmem_dynamic_vmm
 *     |                         |
 * sysmem_static              vidmem_static
 *       |                         |
 *      SHM                      PINNED
 *
 * Supported memory kinds: sysmem (linux shm), vidmem (cudaMalloc), vidmem (cuMemCreate)
 *
 * state->vmm_heap aliases heap_obj only for an active VMM heap.
 */

class nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap(nvshmemi_heap_config cfg) noexcept : cfg_(cfg) {}
    virtual ~nvshmemi_symmetric_heap();

    /** Getters and Setters of protected members */
    size_t get_mem_granularity() const { return mem_granularity_; }
    size_t get_log2_cumem_granularity() const { return log2_mem_granularity_; }
    uint64_t get_reserve_size(void) const { return reserved_heap_size_; }
    size_t get_physical_heap_size(void) const { return physical_internal_heap_size_; }
    uint64_t get_logical_heap_size(void) const { return heap_size_; }
    bool is_multicast_endpoint_enabled(void) const { return le_multicast_enabled_; }
    CUmemAllocationHandleType get_mem_handle_type(void) const { return mem_handle_type_; }
    /**
     * Derive the single concrete handle type to use for cuMem export/import operations.
     * When mem_handle_type_ is a combined bitmask (e.g. FABRIC | POSIX_FILE_DESCRIPTOR),
     * select FABRIC when the MNNVL fabric is active on this PE, otherwise POSIX_FILE_DESCRIPTOR.
     */
    CUmemAllocationHandleType get_effective_import_handle_type(void) const;
    /** True when the effective (resolved) handle type is POSIX file descriptor (IPC). */
    bool is_cuda_mem_handle_type_ipc(void) const {
        return (get_effective_import_handle_type() == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }
    /** True when the effective (resolved) handle type is MNNVL fabric. */
    bool is_cuda_mem_handle_type_fabric(void) const {
        return (get_effective_import_handle_type() == CU_MEM_HANDLE_TYPE_FABRIC);
    }

    /** Common to all memory kinds */
    /**
     * This function will statically reserve heap memory based on memory kind in child class
     * For dynamic vidmem types, only virtual memory is reserved and mspace is initialized
     * For static vidmem and sysmem, virtual and physical memory is reserved/pre-allocated and
     * mspace is initialized
     *
     * @param void
     * @return On success, return 0 and on failure return non-zero NVSHMEM internal error code.
     */
    virtual int reserve_heap(void) = 0;

    virtual int setup_symmetric_heap() = 0;
    virtual int cleanup_symmetric_heap() = 0;

    void *get_base() const { return heap_base_; }
    std::vector<void *> &get_local_pe_bases() { return peer_heap_base_p2p_; }
    const std::vector<void *> &get_local_pe_bases() const { return peer_heap_base_p2p_; }
    void *get_global_base() const { return global_heap_base_; }
    std::vector<void *> &get_remote_pe_bases() { return peer_heap_base_remote_; }
    const std::vector<void *> &get_remote_pe_bases() const { return peer_heap_base_remote_; }
    size_t get_size() const { return heap_size_; }
    std::map<pid_t, int> get_p2p_proc_map() const;

    uint64_t *get_unicast_le_ids() {
        if (le_unicast_enabled_) {
            return unicast_endpoint_ids_with_flag_.data();
        } else {
            return nullptr;
        }
    }

    const uint64_t *get_unicast_le_ids() const {
        if (le_unicast_enabled_) {
            return unicast_endpoint_ids_with_flag_.data();
        } else {
            return nullptr;
        }
    }

    uint64_t get_unicast_le_id(int pe) const {
        if (le_unicast_enabled_) {
            return unicast_endpoint_ids_with_flag_.at(pe);
        } else {
            return 0;
        }
    }

    /** Top-level public facing functions */
    virtual void *heap_malloc(size_t size);
    virtual void *heap_calloc(size_t size, size_t count);
    virtual void *heap_align(size_t size, size_t alignment);
    virtual void heap_deallocate(void *ptr);

    virtual size_t get_mmap_allocated_range() const { return 0; }

    void set_handle_table(std::unique_ptr<nvshmemi_handle_table> table) {
        handle_table_ = std::move(table);
    }

    nvshmemi_handle_table *get_handle_table() const { return handle_table_.get(); }

    void set_heap_registration(std::unique_ptr<nvshmemi_heap_registration> registration);

    void register_observer(std::unique_ptr<nvshmemi_heap_observer> obs) {
        observers_.push_back(std::move(obs));
    }

   protected:
    nvshmemi_mem_remote_transport *get_remoteref(void) const { return (remote_ref_); }
    nvshmemi_mem_p2p_transport *get_p2pref(void) const { return (p2p_ref_); }

    void set_p2p_transport(nvshmemi_mem_p2p_transport *obj) { p2p_ref_ = obj; }
    void set_remote_transport(nvshmemi_mem_remote_transport *obj) { remote_ref_ = obj; }
    void set_mem_handle_type(CUmemAllocationHandleType type) { mem_handle_type_ = type; }
    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment,
                                            int type) = 0;

    /**
     * Given a buf, release and unmap the heap from PE address space
     */
    virtual int release_memory(void *buf, size_t size = 0) = 0;

    /* internal allocation functions */
    virtual void *heap_allocate(size_t size, size_t count, size_t alignment, int type);

    /**
     * Given a value, this API will calculate collectively from all PEs in the team if memory
     * is symmetric with respect to value passed into this API.
     *
     * @param value     templated value to compare for all PEs
     * @return On success, return 0 and on failure return 1
     */
    template <typename T>
    int is_symmetric(T value);

    /*
     * Multicast expects memory buffers to be on different devices (GPU or CPU), which may
     * not be true in case of MPG and EGM buffers on same CPU socket. This function detects
     * the condition
     */
    int check_buffers_on_same_device(bool onGPU, void *ptr);

    /**
     * This function will initialize mspace container for managing virtual address range
     */
    virtual int setup_mspace() = 0;

    /**
     * This function will destroy the mspace container initialized in setup_mspace()
     */
    int cleanup_mspace(void);

    /**
     * Given a mem_granualarity, this API will compute heap size attributes such as heapextra
     * alignbytes and logarithmic2 value of mem_granularity
     */
    void set_heap_size_attr(size_t memgran, size_t *extra, size_t *align, size_t *log_memgran);
    /**
     * This function will allgather the base address for heap_base for p2p/remote transport
     */
    int allgather_peer_base(void);
    /**
     * Allocate virtual memory chunk from the previously init mspace
     */
    void *allocate_virtual_memory_from_mspace(size_t size, size_t count, size_t alignment,
                                              int type);
    nvshmemi_heap_config cfg_ = {};
    // Destroy optional observers and registration before the handle table.
    std::unique_ptr<nvshmemi_handle_table> handle_table_;
    std::unique_ptr<nvshmemi_heap_registration> heap_registration_;
    std::vector<std::unique_ptr<nvshmemi_heap_observer>> observers_;
    CUmemAllocationHandleType mem_handle_type_ = CU_MEM_HANDLE_TYPE_NONE;
    size_t mem_granularity_ = 0;
    size_t le_granularity_ = 0; /* bind alignment of Logical Endpoint*/
    size_t log2_mem_granularity_ = 0;
    size_t physical_internal_heap_size_ = 0;
    size_t heap_size_ = 0;
    uint64_t reserved_heap_size_ = 0;
    void *global_heap_base_ = nullptr;
    void *heap_base_ = nullptr;
    void *mmap_base_ = nullptr;
    std::vector<void *> peer_heap_base_remote_;
    std::vector<void *> peer_heap_base_p2p_;
    nvshmemi_mem_remote_transport *remote_ref_ =
        nullptr;                                     // holds an instance of remote abstraction
    nvshmemi_mem_p2p_transport *p2p_ref_ = nullptr;  // holds an instance of memp2p abstraction
    mspace *heap_mspace_ = nullptr;
    mspace *mmap_mspace_ = nullptr;  // mspace for mmaped region
    // track indices of mc_handles created for mmaped user buffers
    // needed for selectively unbinding them on unmap
    std::unordered_map<void *, size_t> idx_in_mmap_mc_handles_;
    std::unordered_map<void *, size_t> mmap_handle_idx_in_cumem_handles_;
    // track alias virtual addresses for mmaped user buffers
    // map heap addr -> user buffer addr
    std::unordered_map<void *, void *> alias_va_map_;
    std::unordered_map<void *, size_t> egm_map_;

    bool le_unicast_enabled_ = false;
    bool le_multicast_enabled_ = false;
    std::vector<uint64_t> unicast_endpoint_ids_with_flag_;  // 4 bytes valid, 4 bytes for le id
};

class nvshmemi_symmetric_heap_static : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_static(nvshmemi_heap_config cfg) noexcept;
    virtual ~nvshmemi_symmetric_heap_static() = default;

    virtual int reserve_heap(void);
    virtual int setup_symmetric_heap(void);
    virtual int cleanup_symmetric_heap(void);

   protected:
    virtual int allocate_heap_memory() = 0;
    virtual int free_heap_memory(void *addr) = 0;

    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);

    virtual int setup_mspace();
};

class nvshmemi_symmetric_heap_vidmem_static_pinned final : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_static_pinned(nvshmemi_heap_config cfg) noexcept
        : nvshmemi_symmetric_heap_static(cfg) {}
    ~nvshmemi_symmetric_heap_vidmem_static_pinned() = default;

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);

    int release_memory(void *buf, size_t size = 0);
};

class nvshmemi_symmetric_heap_vidmem_dynamic_vmm final : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_dynamic_vmm(nvshmemi_heap_config cfg) noexcept;
    ~nvshmemi_symmetric_heap_vidmem_dynamic_vmm() = default;
    int reserve_heap(void);
    int setup_symmetric_heap(void);
    int cleanup_symmetric_heap(void);

    size_t get_mmap_allocated_range() const override;

    std::map<void *, size_t> *get_mmapped_buf();
    const std::map<void *, size_t> *get_mmapped_buf() const;
    bool is_egm(void *addr);

    /* Functions to map and unmap user buffers
     * memory registered using nvshmemx_buffer_register_symmetric call
     * is refered to as external allocation
     * while memory allocated using nvshmem_malloc is referred to as internal
     * allocation in the code
     */
    void *mmap_mem(void *ptr, size_t size, void *pref_addr, int flags);
    int unmap_mem(void *ptr, size_t size);

    /* Functions for logical endpoints */
    int reserve_unicast_endpoint(size_t size);
    int exchange_endpoints();
    int nvls_setup_multicast_endpoint(nvshmemi_team_t *team, uint64_t mem_size);
    int nvls_setup_multicast_endpoint_by_team(nvshmemi_team_t *team);
    int nvls_bind_multicast_endpoint(nvshmemi_team_t *team, CUmemGenericAllocationHandle mem_handle,
                                     off_t mc_offset, off_t mmap_offset, size_t mmap_size);
    int nvls_unbind_multicast_endpoint(nvshmemi_team_t *team, off_t le_offset, size_t size);
    int nvls_destroy_multicast_endpoint_by_team(nvshmemi_team_t *team);

    size_t get_cumem_handle_count() const { return cumem_handles_.size(); }
    nvshmemi_cumem_handle_info get_cumem_handle_info(size_t i) const {
        const auto &handle = cumem_handles_.at(i);
        return {std::get<0>(handle), std::get<1>(handle), std::get<2>(handle), std::get<3>(handle),
                std::get<4>(handle)};
    }
    void print_cumem_handles(void) const;

    std::unordered_map<void *, void *> *get_alias_va_map() { return &alias_va_map_; }
    std::unordered_map<void *, size_t> *get_egm_map() { return &egm_map_; }

   protected:
    CUmemGenericAllocationHandle get_cumem_handle_ptr(int i) const {
        return (std::get<0>(cumem_handles_[i]));
    }
    off_t get_cumem_handle_alloc_offset(int i) const { return std::get<1>(cumem_handles_[i]); }
    off_t get_cumem_handle_mmap_offset(int i) const { return std::get<2>(cumem_handles_[i]); }
    size_t get_cumem_handle_mmap_size(int i) const { return std::get<3>(cumem_handles_[i]); }
    bool is_cumem_handle_released(int i) const { return std::get<4>(cumem_handles_[i]); }
    size_t get_cumem_handle_size(void) const { return cumem_handles_.size(); }
    int release_memory(void *buf, size_t size);
    void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);
    int setup_mspace();
    int allocate_physical_memory_to_heap(size_t size);

   private:
    void set_cuda_mem_prop(__attribute__((unused)) void *prop, int mem_handle_type) {
        CUmemAllocationProp *memprop = (CUmemAllocationProp *)(prop);
        (*memprop).type = CU_MEM_ALLOCATION_TYPE_PINNED;
        (*memprop).location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        (*memprop).location.id = static_cast<int>(this->cfg_.device_id);
        (*memprop).requestedHandleTypes = (CUmemAllocationHandleType)(mem_handle_type);
        (*memprop).allocFlags.gpuDirectRDMACapable = 1;
        return;
    }

    int check_user_buffer_for_mmap(void *ptr, size_t &size, unsigned int *ptr_mem_type);

    std::vector<std::tuple<CUmemGenericAllocationHandle, off_t, off_t, size_t, bool>>
        cumem_handles_;
    int check_logical_endpoint_support();
};

class nvshmemi_symmetric_heap_sysmem_static_shm final : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_sysmem_static_shm(nvshmemi_heap_config cfg) noexcept
        : nvshmemi_symmetric_heap_static(cfg) {}
    ~nvshmemi_symmetric_heap_sysmem_static_shm() = default;
    static void atexit_heap_handler(void) {
        // Iterate over all objects and close any stale fd
        for (auto i = 0U; i < nvshmemi_symmetric_heap_sysmem_static_shm::infos_.size(); i++) {
            INFO(NVSHMEM_MEM, "Closing file descriptor: %d for sym heap\n",
                 nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
            close(nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
        }
    }

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);

    /** No-op for sysmem: peer ranges are mapped at allocation time and released at cleanup. */
    int release_memory(void *buf, size_t size = 0);

   private:
    char heap_name_[NAME_MAX] = {0};
    nvshmemi_shared_memory_info_t heap_info_ = {};
    // shared global storage for all objects
    static std::vector<nvshmemi_shared_memory_info_t> infos_;
};

#endif /* SYMMETRIC_HEAP_HPP */
