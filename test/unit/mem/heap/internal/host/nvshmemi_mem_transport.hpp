#ifndef NVSHMEMI_MEM_TRANSPORT_HPP
#define NVSHMEMI_MEM_TRANSPORT_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <cstdbool>
#include <climits>
#include <memory>
#include <map>
#include <algorithm>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvmlwrap.h"

static_assert(sizeof(nvshmem_mem_handle_t) % sizeof(uint64_t) == 0,
              "nvshmem_mem_handle_t size is not a multiple of 8B");

/**
 * This is a singleton class managing memory kind specific business logic for p2p transport
 */
class nvshmemi_mem_p2p_transport {
   public:
    ~nvshmemi_mem_p2p_transport() {
        if (p2p_objref_ != nullptr) p2p_objref_ = nullptr;
    }
    nvshmemi_mem_p2p_transport(const nvshmemi_mem_p2p_transport &obj) = delete;
    static nvshmemi_mem_p2p_transport *get_instance(int mype, int npes) {
        if (p2p_objref_ == nullptr) {
            p2p_objref_ = new nvshmemi_mem_p2p_transport(mype, npes);
            return p2p_objref_;
        } else {
            return p2p_objref_;
        }
    }

    static void destroy_instance(void) {
        if (p2p_objref_ != nullptr) {
            delete p2p_objref_;
            p2p_objref_ = nullptr;
        }
    }

    void print_mem_handle(int pe_id, int transport_idx, nvshmemi_symmetric_heap &obj) { return; }
    MOCK_METHOD0(get_nvml_ftable, struct nvml_function_table());

    MOCK_METHOD0(get_mem_handle_type, CUmemAllocationHandleType());
    MOCK_METHOD0(is_mnnvl_fabric, bool());
    MOCK_METHOD0(is_initialized, bool());
    MOCK_METHOD1(create_proc_map, int(nvshmemi_symmetric_heap &obj));
    MOCK_METHOD0(get_proc_map, std::map<pid_t, int>());
    MOCK_METHOD1(get_num_p2p_connected_pes, int(nvshmemi_symmetric_heap &obj));
    MOCK_METHOD1(is_nvl_connected_pe, bool(int pe));

    void set_mem_granularity_(nvshmemi_symmetric_heap &heap, size_t size) {
        heap.mem_granularity_ = size;
    }

    void set_heap_size_(nvshmemi_symmetric_heap &heap, size_t size) { heap.heap_size_ = size; }
    void set_heap_base_(nvshmemi_symmetric_heap &heap) {
        auto &storage = heap_base_storage_[&heap];
        if (!storage) storage = std::make_unique<int>(1);
        heap.heap_base_ = storage.get();
    }

    void set_peer_heap_base_p2p_(nvshmemi_symmetric_heap &heap) {
        auto &storage = peer_heap_base_p2p_storage_[&heap];
        if (!storage) storage = std::make_unique<int>(1);
        void *ptr = storage.get();

        if (heap.peer_heap_base_p2p_ == nullptr) {
            heap.peer_heap_base_p2p_ = (void **)std::calloc(1, sizeof(void *));
        }
        heap.peer_heap_base_p2p_[0] = ptr;
    }

   private:
    static nvml_function_table nvml_ftable_;
    static void *nvml_handle_;
    std::map<nvshmemi_symmetric_heap *, std::unique_ptr<int>> heap_base_storage_;
    std::map<nvshmemi_symmetric_heap *, std::unique_ptr<int>> peer_heap_base_p2p_storage_;
    explicit nvshmemi_mem_p2p_transport(int mype, int npes){};
    static nvshmemi_mem_p2p_transport *p2p_objref_;  // singleton instance
};

/**
 * This is a singleton class managing memory kind specific business logic for remote transport
 */
class nvshmemi_mem_remote_transport {
   public:
    ~nvshmemi_mem_remote_transport() {
        if (remote_objref_ != nullptr) remote_objref_ = nullptr;
    }
    nvshmemi_mem_remote_transport(const nvshmemi_mem_remote_transport &obj) = delete;
    nvshmemi_mem_remote_transport(nvshmemi_mem_remote_transport &&obj) = delete;
    static nvshmemi_mem_remote_transport *get_instance(void) noexcept {
        if (remote_objref_ == nullptr) {
            remote_objref_ = new nvshmemi_mem_remote_transport();
            return remote_objref_;
        } else {
            return remote_objref_;
        }
    }

    static void destroy_instance(void) {
        if (remote_objref_ != nullptr) {
            delete remote_objref_;
            remote_objref_ = nullptr;
        }
    }

    MOCK_METHOD4(gather_mem_handles, int(nvshmemi_symmetric_heap &obj, uint64_t heap_offset,
                                         size_t size, bool ext_alloc));
    MOCK_METHOD3(gather_mem_handles,
                 int(nvshmemi_symmetric_heap &obj, uint64_t heap_offset, size_t size));
    MOCK_METHOD5(register_mem_handle, int(nvshmem_mem_handle_t *local_handles, int transport_idx,
                                          void *buf, size_t size, nvshmem_transport_t current));
    MOCK_METHOD2(release_mem_handles,
                 int(nvshmem_mem_handle_t *handles, nvshmemi_symmetric_heap &obj));

   private:
    explicit nvshmemi_mem_remote_transport(void){};
    static nvshmemi_mem_remote_transport *remote_objref_;  // singleton instance
};

#endif
