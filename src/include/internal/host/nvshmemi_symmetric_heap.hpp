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
#include <tuple>
#include <unordered_map>
#include <vector>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/cudawrap.h"
#include "non_abi/nvshmemx_error.h"
#include "device_host/logical_endpoint_types.h"

/// Forward declarations for future friends
class nvshmemi_mem_p2p_transport;
class nvshmemi_mem_remote_transport;

enum { NVSHMEMX_MALLOC = 0, NVSHMEMX_CALLOC, NVSHMEMX_ALIGN, NVSHMEMX_ALLOC_MAX };

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

    /** Called after a physical chunk is mapped into the heap virtual range. */
    virtual int on_chunk_mapped(nvshmem_mem_handle_t *handle, off_t mc_offset, off_t mmap_offset,
                                size_t size) = 0;

    /** Called before a physical chunk is unmapped from the heap virtual range. */
    virtual int on_chunk_unmapped(off_t mc_offset, size_t size) = 0;

    /** Called before heap virtual memory and CUDA handles are released. */
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
    explicit nvshmemi_symmetric_heap(nvshmemi_heap_config cfg, nvshmemi_state_t *state) noexcept
        : cfg_(cfg), state_(state) {}
    virtual ~nvshmemi_symmetric_heap();

    /** Getters and Setters of protected members */
    size_t get_mem_granularity() const { return mem_granularity_; }
    size_t get_log2_cumem_granularity() const { return log2_mem_granularity_; }
    uint64_t get_reserve_size(void) const { return reserved_heap_size_; }
    size_t get_physical_heap_size(void) const { return physical_internal_heap_size_; }
    uint64_t get_logical_heap_size(void) const { return heap_size_; }
    CUmemAllocationHandleType get_mem_handle_type(void) { return mem_handle_type_; }
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

    /**
     * Given an address, size, pe index, and transport index, retrieve the corresponding
     * transport handle.
     */
    inline nvshmem_mem_handle *get_transport_mem_handle(void *addr, size_t *len, int pe,
                                                        int transport_idx);

    inline size_t get_mem_handle_addr_offset(void *addr);

    void *get_base() { return heap_base_; }
    void **get_local_pe_base() { return peer_heap_base_p2p_; }
    void *get_global_base() { return global_heap_base_; }
    void **get_remote_pe_base() { return peer_heap_base_remote_; }
    size_t get_size() { return heap_size_; }

    uint64_t *get_unicast_le_ids() {
        if (le_unicast_enabled_) {
            return unicast_endpoint_ids_with_flag_.data();
        } else {
            return nullptr;
        }
    }

    uint64_t get_unicast_le_id(int pe) {
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

    virtual size_t get_mmap_allocated_range() { return 0; }

    void register_observer(std::unique_ptr<nvshmemi_heap_observer> observer) {
        observers_.push_back(std::move(observer));
    }

   private:
    friend class nvshmemi_mem_p2p_transport;     // friend class declaration
    friend class nvshmemi_mem_remote_transport;  // friend class declaration

   protected:
    nvshmemi_mem_remote_transport *get_remoteref(void) { return (remote_ref_); }
    nvshmemi_mem_p2p_transport *get_p2pref(void) { return (p2p_ref_); }
    nvshmemi_state_t *get_state() { return state_; }

    void set_p2p_transport(nvshmemi_mem_p2p_transport *obj) { p2p_ref_ = obj; }
    void set_remote_transport(nvshmemi_mem_remote_transport *obj) { remote_ref_ = obj; }
    void set_mem_handle_type(CUmemAllocationHandleType type) { mem_handle_type_ = type; }
    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment,
                                            int type) = 0;

    void inc_heap_handle_cache(void) { heap_handle_cache_++; }
    bool empty_heap_handle_cache(void) { return heap_handle_cache_ == 0; }

    /**
     * Given a buf, size address range, map the heap into PE address space
     */
    virtual int map_heap_range_by_size(void *buf, size_t size);
    virtual int update_heap_handle_cache(void *buf, size_t size, bool ext_allocation = false);

    /**
     * Given a buffer, size, index to valid transport and PE#, map the buffer range into target PE
     * address space
     */
    virtual int map_heap_range_by_pe(int pe_id, int transport_idx, char *buf = nullptr,
                                     size_t size = 0) = 0;
    /**
     * Given an collection of local memory handles across all PEs, establish pairwise memory handles
     * for processes connected over p2p transport
     */
    virtual int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles) = 0;

    /**
     * Given a peer mem handle, import the buffer range to target buf object
     */
    virtual int import_memory(nvshmem_mem_handle_t *peer_handle, void **buf, size_t length = 0) = 0;
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
     * Given an address and size,
     */
    virtual void update_idx_in_handle(void *addr, size_t size, size_t idx,
                                      bool ext_allocation = false);

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
    nvshmemi_heap_config cfg_ = {};      // PE topology + device, captured at construction time
    nvshmemi_state_t *state_ = nullptr;  // store a reference of device state instance
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
    void **peer_heap_base_remote_ = nullptr;
    void **peer_heap_base_p2p_ = nullptr;
    int heap_handle_cache_ = 0;
    nvshmemi_mem_remote_transport *remote_ref_ =
        nullptr;                                     // holds an instance of remote abstraction
    nvshmemi_mem_p2p_transport *p2p_ref_ = nullptr;  // holds an instance of memp2p abstraction
    mspace *heap_mspace_ = nullptr;
    std::vector<std::vector<nvshmem_mem_handle>> remote_handles_;
    std::vector<std::vector<nvshmem_mem_handle>> p2p_handles_;
    std::vector<std::vector<nvshmem_mem_handle>> remote_mmap_handles_;
    mspace *mmap_mspace_ = nullptr;  // mspace for mmaped region
    std::vector<std::tuple<size_t, void *, size_t>> idx_in_handles_;
    std::map<size_t, std::tuple<size_t, void *, size_t>> idx_in_mmap_handles_;
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

inline nvshmem_mem_handle *nvshmemi_symmetric_heap::get_transport_mem_handle(void *addr,
                                                                             size_t *len, int pe,
                                                                             int transport_idx) {
    size_t addr_idx;
    size_t handle_idx;
    size_t handle_size;
    size_t handle_sub_index;
    size_t offset;
    nvshmem_mem_handle *return_handle;

    void *handle_start_addr;

    if (addr < heap_base_ || addr > (char *)heap_base_ + heap_size_) {
        return NULL;
    }

    offset = (char *)addr - (char *)heap_base_;
    addr_idx = offset >> log2_mem_granularity_;
    handle_sub_index = pe * get_state()->num_initialized_transports + transport_idx;

    // if address in within mmap range, use mmap_handles
    if (addr >= ((char *)heap_base_ + (heap_size_ - get_mmap_allocated_range()))) {
        handle_idx = std::get<0>(idx_in_mmap_handles_[addr_idx]);
        handle_start_addr = std::get<1>(idx_in_mmap_handles_[addr_idx]);
        handle_size = std::get<2>(idx_in_mmap_handles_[addr_idx]);
        return_handle = &((remote_mmap_handles_.at(handle_idx)).at(handle_sub_index));
    } else {
        handle_idx = std::get<0>(idx_in_handles_[addr_idx]);
        handle_start_addr = std::get<1>(idx_in_handles_[addr_idx]);
        handle_size = std::get<2>(idx_in_handles_[addr_idx]);
        return_handle = &remote_handles_[handle_idx][handle_sub_index];
    }

    // getting the remainder space within chunk - why?
    if (len) {
        *len = handle_size - ((char *)addr - (char *)handle_start_addr);
    }
    return return_handle;
}

inline size_t nvshmemi_symmetric_heap::get_mem_handle_addr_offset(void *addr) {
    size_t addr_idx;
    size_t heap_offset;
    size_t offset;
    void *start_addr;

    heap_offset = (char *)addr - (char *)heap_base_;
    addr_idx = heap_offset >> log2_mem_granularity_;
    start_addr = std::get<1>(idx_in_handles_[addr_idx]);

    offset = (char *)addr - (char *)start_addr;

    return offset;
}

class nvshmemi_symmetric_heap_static : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_static(nvshmemi_heap_config cfg,
                                            nvshmemi_state_t *state) noexcept;
    virtual ~nvshmemi_symmetric_heap_static() = default;

    virtual int reserve_heap(void);
    virtual int setup_symmetric_heap(void);
    virtual int cleanup_symmetric_heap(void);

   protected:
    virtual int allocate_heap_memory() = 0;
    virtual int free_heap_memory(void *addr) = 0;

    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);

    virtual int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx,
                                            void *buf, size_t size,
                                            nvshmem_transport_t current) = 0;
    /**
     * Given a buffer, size and input memory handle, register the heap into PE address space
     */
    virtual int register_heap_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size);
    virtual int map_heap_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size);
    virtual int register_heap_chunk_by_size(void *buf, size_t size);
    virtual int setup_mspace();

    /**
     * Given a buf, length, export the buffer range to target mem handle
     */
    virtual int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length) = 0;

   private:
    bool gather_mem_handles_done_ = false;
};

class nvshmemi_symmetric_heap_vidmem_static_pinned final : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_static_pinned(nvshmemi_heap_config cfg,
                                                          nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(cfg, state) {}
    ~nvshmemi_symmetric_heap_vidmem_static_pinned() = default;

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);
    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx, void *buf,
                                    size_t size, nvshmem_transport_t current);
    int map_heap_range_by_pe(int pe_id, int transport_idx, char *buf = nullptr, size_t size = 0);

    int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length = 0);
    int release_memory(void *buf, size_t size = 0);
};

class nvshmemi_symmetric_heap_vidmem_dynamic_vmm final : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_dynamic_vmm(nvshmemi_heap_config cfg,
                                                        nvshmemi_state_t *state) noexcept;
    ~nvshmemi_symmetric_heap_vidmem_dynamic_vmm() = default;
    int reserve_heap(void);
    int setup_symmetric_heap(void);
    int cleanup_symmetric_heap(void);

    /* Operates on the complete heap state. */
    int nvls_create_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_bind_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_map_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unmap_heap_memory_by_size(nvshmemi_team_t *team, off_t mc_offset, uint64_t size);
    void nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unmap_heap_memory(off_t mc_offset, uint64_t size);
    void nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_unbind_heap_memory_by_size(off_t mc_offset, size_t size);
    size_t get_mmap_allocated_range();
    bool is_egm(void *addr);
    std::map<void *, size_t> *get_mmapped_buf();

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

    std::unordered_map<void *, void *> *get_alias_va_map() { return &alias_va_map_; }
    std::unordered_map<void *, size_t> *get_egm_map() { return &egm_map_; }

   protected:
    CUmemGenericAllocationHandle get_cumem_handle_ptr(int i) {
        return (std::get<0>(cumem_handles_[i]));
    }
    off_t get_cumem_handle_alloc_offset(int i) { return std::get<1>(cumem_handles_[i]); }
    off_t get_cumem_handle_mmap_offset(int i) { return std::get<2>(cumem_handles_[i]); }
    size_t get_cumem_handle_mmap_size(int i) { return std::get<3>(cumem_handles_[i]); }
    bool is_cumem_handle_released(int i) { return std::get<4>(cumem_handles_[i]); }
    size_t get_cumem_handle_size(void) { return cumem_handles_.size(); }
    void print_cumem_handles(void);
    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int map_heap_range_by_pe(int pe_id, int transport_idx, char *buf, size_t size);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length);
    int export_memory(nvshmem_mem_handle_t *mem_handle, nvshmem_mem_handle_t *mem_handle_in);
    int release_memory(void *buf, size_t size);
    void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);
    int setup_mspace();
    /** Registers a VMM chunk; ext_allocation identifies user-provided mmap memory. */
    int register_heap_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size,
                             bool ext_allocation = false);
    int map_heap_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size);
    int register_heap_chunk_by_size(void *buf, size_t size, bool ext_allocation = false);
    int allocate_physical_memory_to_heap(size_t size);
    int nvls_broadcast_heap_handle_fabric(char *shareable_handle, size_t length, int root,
                                          nvshmemi_team_t *team);
    int nvls_broadcast_heap_handle_ipc(char *shareable_handle, int root, nvshmemi_team_t *team);
    int nvls_broadcast_heap_handle_by_team(char *shareable_handle, size_t length,
                                           nvshmemi_team_t *team);
    /* Operates on a given allocation request of size mem_size */
    int nvls_create_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size);
    int nvls_bind_heap_memory_by_size(nvshmemi_team_t *team, nvshmem_mem_handle_t *mem_handle,
                                      off_t mc_offset, off_t mmap_offset, size_t mmap_size);
    int nvls_map_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size, off_t mmap_offset,
                                     off_t mc_offset);
    int nvls_create_heap_memory(uint64_t mem_size);
    int nvls_bind_heap_memory(nvshmem_mem_handle_t *mem_handle, off_t mc_offset, off_t mmap_offset,
                              size_t mmap_size);
    int nvls_map_heap_memory(uint64_t mem_size, off_t mmap_offset, off_t mc_offset);

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
    explicit nvshmemi_symmetric_heap_sysmem_static_shm(nvshmemi_heap_config cfg,
                                                       nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(cfg, state) {}
    ~nvshmemi_symmetric_heap_sysmem_static_shm() = default;
    static void atexit_heap_handler(void) {
        // Iterate over all objects and close any stale fd
        for (auto i = 0U; i < nvshmemi_symmetric_heap_sysmem_static_shm::infos_.size(); i++) {
            INFO(NVSHMEM_MEM, "Closing file descriptor: %d for sym heap\n",
                 nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
            close(nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
        }
    }
    int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx, void *buf,
                                    size_t size, nvshmem_transport_t current);

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);

    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int map_heap_range_by_pe(int pe_id, int transport_idx, char *buf = nullptr, size_t size = 0);

    /** Stubbed out for P2P transport as export is non-action, import is done at allocation time,
     * release is done at cleanup time **/
    int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length = 0);
    int release_memory(void *buf, size_t size = 0);

   private:
    char heap_name_[NAME_MAX] = {0};
    nvshmemi_shared_memory_info_t heap_info_ = {};
    // shared global storage for all objects
    static std::vector<nvshmemi_shared_memory_info_t> infos_;
};

#endif /* SYMMETRIC_HEAP_HPP */
