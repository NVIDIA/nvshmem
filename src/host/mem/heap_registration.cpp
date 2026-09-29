/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/common/error_codes_internal.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_handle_table.hpp"
#include "internal/host/nvshmemi_heap_registration.hpp"
#include "internal/host/nvshmemi_mem_transport.hpp"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/sockets.h"
#include "internal/host/util.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"
#include "non_abi/c/nvshmem_build_options.h"
#include "non_abi/c/nvshmemi_error_macros.h"

namespace {

class UniqueFd {
   public:
    UniqueFd() noexcept = default;
    explicit UniqueFd(int fd) noexcept : fd_{fd} { assert(fd >= 0); }
    ~UniqueFd() { reset(); }

    UniqueFd(const UniqueFd &) = delete;
    UniqueFd &operator=(const UniqueFd &) = delete;

    UniqueFd(UniqueFd &&other) noexcept : fd_{std::exchange(other.fd_, std::nullopt)} {}

    UniqueFd &operator=(UniqueFd &&other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, std::nullopt);
        }
        return *this;
    }

    int get() const noexcept {
        assert(fd_.has_value());
        return *fd_;
    }

   private:
    void reset() noexcept {
        if (fd_) {
            close(*fd_);
            fd_.reset();
        }
    }

    std::optional<int> fd_;
};

class UniqueCudaAllocationHandle {
   public:
    explicit UniqueCudaAllocationHandle(CUmemGenericAllocationHandle handle) noexcept
        : handle_{handle} {}
    ~UniqueCudaAllocationHandle() { (void)reset(); }

    UniqueCudaAllocationHandle(const UniqueCudaAllocationHandle &) = delete;
    UniqueCudaAllocationHandle &operator=(const UniqueCudaAllocationHandle &) = delete;

    UniqueCudaAllocationHandle(UniqueCudaAllocationHandle &&other) noexcept
        : handle_{std::exchange(other.handle_, std::nullopt)} {}

    UniqueCudaAllocationHandle &operator=(UniqueCudaAllocationHandle &&other) noexcept {
        if (this != &other) {
            UniqueCudaAllocationHandle moved{std::move(other)};
            swap(moved);
        }
        return *this;
    }

    CUmemGenericAllocationHandle get() const noexcept {
        assert(handle_.has_value());
        return *handle_;
    }

    CUresult reset() noexcept {
        if (!handle_) {
            return CUDA_SUCCESS;
        }
        CUresult status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(*handle_));
        if (status == CUDA_SUCCESS) {
            handle_.reset();
        }
        return status;
    }

   private:
    void swap(UniqueCudaAllocationHandle &other) noexcept { handle_.swap(other.handle_); }

    std::optional<CUmemGenericAllocationHandle> handle_;
};

template <typename MapPeer>
int map_p2p_peers(int mype, int npes, nvshmemi_transport_view &transports,
                  std::vector<void *> &peer_heap_bases, MapPeer &&map_peer) {
    for (int pe = (mype + 1) % npes; pe != mype; pe = (pe + 1) % npes) {
        for (int transport = 0; transport < transports.num_transports(); transport++) {
            if (!transports.active_between_has_cap(mype, pe, npes, transport,
                                                   NVSHMEM_TRANSPORT_CAP_MAP)) {
                continue;
            }

            int status = map_peer(pe);
            if (status == NVSHMEMX_SUCCESS) {
                break;
            }
            if (status != NVSHMEMX_ERROR_INVALID_VALUE) {
                return status;
            }

            transports.clear_cap(transport, pe,
                                 NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                                     NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
                                     NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS);
            peer_heap_bases[pe] = nullptr;
        }
    }
    return NVSHMEMX_SUCCESS;
}

int exchange_posix_fds(const UniqueFd &local_fd, const std::map<pid_t, int> &p2p_processes,
                       std::map<int, UniqueFd> *peer_fds) {
    int status = NVSHMEMX_SUCCESS;
    ipcHandle *send_socket = nullptr;
    std::map<pid_t, ipcHandle *> receive_sockets;
    const pid_t pid = getpid();

    NVSHMEMI_IPC_CHECK(ipcOpenSocket(send_socket, pid, pid));
    for (const auto &entry : p2p_processes) {
        const pid_t peer_pid = entry.first;
        if (peer_pid == pid) {
            continue;
        }
        ipcHandle *socket = nullptr;
        NVSHMEMI_IPC_CHECK(ipcOpenSocket(socket, peer_pid, pid));
        receive_sockets.emplace(peer_pid, socket);
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, close_sockets,
                          "bootstrap barrier failed before FD exchange\n");

    for (const auto &entry : p2p_processes) {
        const pid_t peer_pid = entry.first;
        if (peer_pid == pid) {
            continue;
        }
        NVSHMEMI_IPC_CHECK(ipcSendFd(send_socket, local_fd.get(), pid, peer_pid));
    }

    for (const auto &[peer_pid, pe] : p2p_processes) {
        if (peer_pid == pid) {
            continue;
        }
        int received_fd{-1};
        NVSHMEMI_IPC_CHECK(ipcRecvFd(receive_sockets.at(peer_pid), &received_fd));
        peer_fds->emplace(pe, UniqueFd{received_fd});
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

close_sockets:
    NVSHMEMI_IPC_CHECK(ipcCloseSocket(send_socket));
    for (const auto &entry : receive_sockets) {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(entry.second));
    }
    return status;
}

int consume_cuda_runtime_error_for_failed_ipc(cudaError_t api_status, const char *api_name) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        return NVSHMEMX_SUCCESS;
    }
    if (status != api_status) {
        NVSHMEMI_ERROR_PRINT("%s returned %d (%s), but cudaGetLastError returned %d (%s)\n",
                             api_name, api_status, cudaGetErrorString(api_status), status,
                             cudaGetErrorString(status));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return NVSHMEMX_SUCCESS;
}

}  // namespace

nvshmemi_heap_registration::nvshmemi_heap_registration(
    nvshmemi_handle_table *table, nvshmemi_heap_registration_config cfg) noexcept
    : table_(table),
      policy_(cfg.policy),
      remote_transport_(cfg.remote_transport),
      p2p_transport_(cfg.p2p_transport),
      peer_heap_base_p2p_(cfg.peer_heap_base_p2p),
      geometry_(cfg.geometry),
      effective_handle_type_(cfg.effective_handle_type),
      transports_(cfg.transports),
      mype_(cfg.mype),
      npes_(cfg.npes),
      npes_node_(cfg.npes_node) {
    assert(table_ != nullptr);
    assert(peer_heap_base_p2p_.size() >= static_cast<size_t>(npes_));
    assert(geometry_.heap_base != nullptr);
    assert(geometry_.logical_heap_size > 0);
    assert(geometry_.mem_granularity > 0);
}

int nvshmemi_heap_registration::setup() {
    switch (policy_) {
        case nvshmemi_heap_registration_policy::DYNAMIC_VMM:
            return plan_vmm_peer_bases();
        case nvshmemi_heap_registration_policy::STATIC_VIDMEM_FULL_HEAP:
        case nvshmemi_heap_registration_policy::STATIC_SYSMEM_FULL_HEAP:
            return register_static_heap();
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemi_heap_registration::teardown() {
    nvshmemi_mem_remote_transport &remote_tran = remote_transport();
    auto &internal_reg = table_->internal_reg();
    int first_status = NVSHMEMX_SUCCESS;

    for (size_t j = 0; j < internal_reg.num_handle_sets(); j++) {
        if (j > 0 && nvshmemi_device_state.enable_rail_opt) {
            continue;
        }
        nvshmem_mem_handle_t *handles = internal_reg.get_mem_handle(j, mype_, 0);
        int status = remote_tran.release_mem_handles(handles, transports_);
        if (status) {
            if (first_status == NVSHMEMX_SUCCESS) {
                first_status = status;
            }
            INFO(NVSHMEM_MEM, "release_mem_handles failed during heap teardown (j=%zu)\n", j);
        }
    }

    return first_status;
}

nvshmemi_mem_remote_transport &nvshmemi_heap_registration::remote_transport() {
    return remote_transport_;
}

nvshmemi_mem_p2p_transport &nvshmemi_heap_registration::p2p_transport() { return p2p_transport_; }

bool nvshmemi_heap_registration::is_node_local_pe(int pe_id) const {
    assert(pe_id >= 0 && pe_id < npes_);
    if (nvshmemi_host_hashes != nullptr) {
        return nvshmemi_host_hashes[pe_id] == nvshmemi_host_hashes[mype_];
    }

    return (pe_id / npes_node_) == (mype_ / npes_node_);
}

int nvshmemi_heap_registration::node_local_index(int pe_id) const {
    assert(is_node_local_pe(pe_id));
    if (nvshmemi_host_hashes == nullptr) {
        return pe_id % npes_node_;
    }

    int local_index = 0;
    for (int i = 0; i < pe_id; i++) {
        if (nvshmemi_host_hashes[i] == nvshmemi_host_hashes[pe_id]) {
            local_index++;
        }
    }
    return local_index;
}

bool nvshmemi_heap_registration::has_p2p_mapping() const {
    for (int pe = 0; pe < npes_; pe++) {
        for (int transport = 0; transport < transports_.num_transports(); transport++) {
            if (transports_.active_has_cap(transport, pe, NVSHMEM_TRANSPORT_CAP_MAP)) {
                return true;
            }
        }
    }
    return false;
}

int nvshmemi_heap_registration::map_dynamic_p2p_chunk(CUmemGenericAllocationHandle handle,
                                                      void *buf, size_t size) {
    if (!has_p2p_mapping()) {
        return NVSHMEMX_SUCCESS;
    }

    switch (effective_handle_type_) {
        case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
            return map_posix_p2p_chunk(handle, buf, size);
        case CU_MEM_HANDLE_TYPE_FABRIC:
            return map_fabric_p2p_chunk(handle, buf, size);
        default:
            NVSHMEMI_ERROR_PRINT("Unsupported P2P VMM handle type: %d\n", effective_handle_type_);
            return NVSHMEMX_ERROR_INVALID_VALUE;
    }
}

int nvshmemi_heap_registration::map_posix_p2p_chunk(CUmemGenericAllocationHandle handle, void *buf,
                                                    size_t size) {
    int raw_fd{-1};
    int status = CUPFN(
        nvshmemi_cuda_syms,
        cuMemExportToShareableHandle(&raw_fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    UniqueFd local_fd{};
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
    } else {
        local_fd = UniqueFd{raw_fd};
    }
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("POSIX FD export failed on at least one PE\n");
        return status;
    }

    std::map<int, UniqueFd> peer_fds;
    status = exchange_posix_fds(local_fd, p2p_transport().get_proc_map(), &peer_fds);
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("POSIX FD exchange failed on at least one PE\n");
        return status;
    }

    status = map_p2p_peers(mype_, npes_, transports_, peer_heap_base_p2p_, [&](int pe) -> int {
        auto it = peer_fds.find(pe);
        if (it == peer_fds.end()) {
            NVSHMEMI_ERROR_PRINT("Missing POSIX FD for PE %d\n", pe);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        UniqueFd fd{std::move(it->second)};
        peer_fds.erase(it);

        CUmemGenericAllocationHandle peer_handle{};
        int import_status =
            CUPFN(nvshmemi_cuda_syms,
                  cuMemImportFromShareableHandle(
                      &peer_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd.get())),
                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        if (import_status != CUDA_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("cuMemImportFromShareableHandle failed for PE %d\n", pe);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        return map_imported_p2p_memory(pe, peer_handle, buf, size);
    });
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    return status;
}

int nvshmemi_heap_registration::map_fabric_p2p_chunk(CUmemGenericAllocationHandle handle, void *buf,
                                                     size_t size) {
    CUmemFabricHandle local_handle = {};
    int status =
        CUPFN(nvshmemi_cuda_syms,
              cuMemExportToShareableHandle(&local_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
    }
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("CUDA Fabric handle export failed on at least one PE\n");
        return status;
    }

    std::vector<CUmemFabricHandle> peer_handles(npes_);
    status = nvshmemi_boot_handle.allgather(&local_handle, peer_handles.data(),
                                            sizeof(local_handle), &nvshmemi_boot_handle);
    if (status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("CUDA Fabric handle allgather failed\n");
        return status;
    }

    status = map_p2p_peers(mype_, npes_, transports_, peer_heap_base_p2p_, [&](int pe) {
        return map_fabric_p2p_memory(pe, peer_handles[pe], buf, size);
    });
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    return status;
}

int nvshmemi_heap_registration::map_static_vidmem_p2p_chunk(void *buf) {
    if (!has_p2p_mapping()) {
        return NVSHMEMX_SUCCESS;
    }

    cudaIpcMemHandle_t local_handle = {};
    const cudaError_t cuda_status = cudaIpcGetMemHandle(&local_handle, buf);
    int status = NVSHMEMX_SUCCESS;
    if (cuda_status != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("cudaIpcGetMemHandle failed with error %d (%s)\n", cuda_status,
                             cudaGetErrorString(cuda_status));
        status = consume_cuda_runtime_error_for_failed_ipc(cuda_status, "cudaIpcGetMemHandle");
        if (status == NVSHMEMX_SUCCESS) {
            status = NVSHMEMX_ERROR_INVALID_VALUE;
        }
    }
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    if (status != NVSHMEMX_SUCCESS) {
        return status;
    }

    std::vector<cudaIpcMemHandle_t> peer_handles(npes_);
    status = nvshmemi_boot_handle.allgather(&local_handle, peer_handles.data(),
                                            sizeof(local_handle), &nvshmemi_boot_handle);
    if (status != NVSHMEMX_SUCCESS) {
        return status;
    }

    status = map_p2p_peers(mype_, npes_, transports_, peer_heap_base_p2p_, [&](int pe) {
        return map_static_vidmem_p2p_memory(pe, peer_handles[pe]);
    });
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    return status;
}

int nvshmemi_heap_registration::map_static_sysmem_p2p_chunk() {
    peer_heap_base_p2p_[mype_] = geometry_.heap_base;
    int status = map_p2p_peers(mype_, npes_, transports_, peer_heap_base_p2p_,
                               [&](int pe) { return map_static_sysmem_p2p_memory(pe); });
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    return status;
}

int nvshmemi_heap_registration::map_fabric_p2p_memory(int pe_id, CUmemFabricHandle handle,
                                                      void *buf, size_t size) {
    CUmemGenericAllocationHandle peer_handle{};
    int status = CUPFN(nvshmemi_cuda_syms, cuMemImportFromShareableHandle(
                                               &peer_handle, &handle, CU_MEM_HANDLE_TYPE_FABRIC));
    if (status != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuMemImportFromShareableHandle failed for PE %d\n", pe_id);
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return map_imported_p2p_memory(pe_id, peer_handle, buf, size);
}

int nvshmemi_heap_registration::map_imported_p2p_memory(int pe_id,
                                                        CUmemGenericAllocationHandle handle,
                                                        void *buf, size_t size) {
    UniqueCudaAllocationHandle imported_handle{handle};
    CUdevice gpu_device_id{};
    int status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuCtxGetDevice failed\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    void *buf_map = reinterpret_cast<void *>(static_cast<char *>(buf) -
                                             static_cast<char *>(geometry_.heap_base) +
                                             static_cast<char *>(peer_heap_base_p2p_[pe_id]));
    const auto buf_map_device_ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(buf_map));
    status =
        CUPFN(nvshmemi_cuda_syms, cuMemMap(buf_map_device_ptr, size, 0, imported_handle.get(), 0));
    if (status != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("cuMemMap failed to map %zu bytes at %p\n", size, buf_map);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = imported_handle.reset();
    if (status != CUDA_SUCCESS) {
        (void)CUPFN(nvshmemi_cuda_syms, cuMemUnmap(buf_map_device_ptr, size));
        NVSHMEMI_ERROR_PRINT("cuMemRelease failed\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = gpu_device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = CUPFN(nvshmemi_cuda_syms, cuMemSetAccess(buf_map_device_ptr, size, &access, 1));
    if (status != CUDA_SUCCESS) {
        (void)CUPFN(nvshmemi_cuda_syms, cuMemUnmap(buf_map_device_ptr, size));
        NVSHMEMI_ERROR_PRINT("cuMemSetAccess failed\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return NVSHMEMX_SUCCESS;
}

int nvshmemi_heap_registration::map_static_vidmem_p2p_memory(int pe_id,
                                                             const cudaIpcMemHandle_t &handle) {
    const cudaError_t cuda_status =
        cudaIpcOpenMemHandle(&peer_heap_base_p2p_[pe_id], handle, cudaIpcMemLazyEnablePeerAccess);
    if (cuda_status == cudaSuccess) {
        return NVSHMEMX_SUCCESS;
    }

    NVSHMEMI_ERROR_PRINT("cudaIpcOpenMemHandle failed with error %d (%s)\n", cuda_status,
                         cudaGetErrorString(cuda_status));
    int status = consume_cuda_runtime_error_for_failed_ipc(cuda_status, "cudaIpcOpenMemHandle");
    return status == NVSHMEMX_SUCCESS ? NVSHMEMX_ERROR_INVALID_VALUE : status;
}

int nvshmemi_heap_registration::map_static_sysmem_p2p_memory(int pe_id) {
    if (!is_node_local_pe(pe_id)) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    peer_heap_base_p2p_[pe_id] = static_cast<char *>(geometry_.global_heap_base) +
                                 node_local_index(pe_id) * geometry_.logical_heap_size;
    return NVSHMEMX_SUCCESS;
}

int nvshmemi_heap_registration::register_remote_chunk(void *buf, size_t size,
                                                      nvshmemi_allocation_kind alloc_kind) {
    int status = NVSHMEMX_SUCCESS;
    bool local_handles_owned = true;
    nvshmemi_mem_remote_transport &remotetran = remote_transport();
    std::vector<nvshmem_mem_handle_t> local_handles(transports_.num_transports());
    std::vector<nvshmem_mem_handle_t> gathered(transports_.num_transports() * npes_);
    void *remote_buf = buf;
    size_t remote_size = size;

    // Rail opt: static sysmem registers the entire heap range once for reuse.
    if (policy_ == nvshmemi_heap_registration_policy::STATIC_SYSMEM_FULL_HEAP &&
        nvshmemi_device_state.enable_rail_opt == 1) {
        remote_buf = geometry_.global_heap_base;
        remote_size = geometry_.logical_heap_size * npes_node_;
    }

    /* Register local handles for the requested buffer range. */
    for (int i = 0; i < transports_.num_transports(); i++) {
        if (!transports_.is_active(i) || transports_.has_cap(i, mype_, NVSHMEM_TRANSPORT_CAP_MAP)) {
            continue;
        }
        status = remotetran.register_mem_handle(local_handles.data(), i, remote_buf, remote_size,
                                                transports_);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "register_mem_handle failed\n");
    }

    /* Allgather memory handles for all PEs. */
    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles.data(), (void *)gathered.data(),
        sizeof(nvshmem_mem_handle_t) * transports_.num_transports(), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed\n");

    /* Complete transport setup before publishing handles and their lookup index. */
    if (alloc_kind == nvshmemi_allocation_kind::INTERNAL &&
        nvshmemi_device_state.enable_rail_opt == 1) {
        if (!gather_mem_handles_done_) {
            status = remotetran.gather_mem_handles(transports_, gathered.data(), 0,
                                                   geometry_.logical_heap_size * npes_node_);
            if (status == NVSHMEMX_SUCCESS) {
                gather_mem_handles_done_ = true;
            }
        }
    } else {
        status = remotetran.gather_mem_handles(transports_, gathered.data(),
                                               ((char *)buf - (char *)geometry_.heap_base), size);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gather_mem_handles failed\n");

    /* Store gathered handles in the allocation-specific registry. */
    if (alloc_kind == nvshmemi_allocation_kind::EXTERNAL) {
        table_->mmap_reg().push_mem_handles(std::move(gathered));
    } else {
        table_->internal_reg().push_mem_handles(std::move(gathered));
    }

    /* Update lookup state with the retrieved memory handles. */
    update_handle_index(buf, size, alloc_kind);
    local_handles_owned = false;

out:
    if (local_handles_owned) {
        const int cleanup_status =
            remotetran.release_mem_handles(local_handles.data(), transports_);
        if (cleanup_status != NVSHMEMX_SUCCESS) {
            NVSHMEMI_WARN_PRINT("release mem handles failed after registration failure\n");
            if (status == NVSHMEMX_SUCCESS) {
                status = cleanup_status;
            }
        }
    }
    return status;
}

void nvshmemi_heap_registration::update_handle_index(void *buf, size_t size,
                                                     nvshmemi_allocation_kind alloc_kind) {
    auto &internal_reg = table_->internal_reg();
    auto &mmap_reg = table_->mmap_reg();
    size_t granularity = geometry_.mem_granularity;

    /* Rail-optimized internal memory reuses one index for the full heap. */
    if (nvshmemi_device_state.enable_rail_opt == 1 &&
        alloc_kind == nvshmemi_allocation_kind::INTERNAL) {
        if (table_->empty_handle_cache()) {
            uint64_t full_heap_size = geometry_.logical_heap_size * npes_node_;
            internal_reg.append_index(full_heap_size / granularity,
                                      internal_reg.num_handle_sets() - 1,
                                      (char *)geometry_.global_heap_base, full_heap_size);
        }
    } else if (alloc_kind == nvshmemi_allocation_kind::EXTERNAL) {
        /* User-provided mmap allocations use address-keyed sparse entries. */
        size_t addr_idx =
            ((char *)buf - (char *)geometry_.heap_base) >> geometry_.log2_mem_granularity;
        for (size_t idx = 0; idx < size / granularity; idx++) {
            mmap_reg.set_index(addr_idx + idx, mmap_reg.num_handle_sets() - 1, (char *)buf, size);
        }
    } else {
        /* Internal allocations append dense entries in heap order. */
        internal_reg.append_index(size / granularity, internal_reg.num_handle_sets() - 1,
                                  (char *)buf, size);
    }

    if (table_->empty_handle_cache()) {
        table_->inc_handle_cache();
    }
}

int nvshmemi_heap_registration::register_vmm_chunk(CUmemGenericAllocationHandle handle,
                                                   off_t mc_offset, size_t size,
                                                   nvshmemi_allocation_kind alloc_kind,
                                                   std::optional<size_t> mmap_allocated_range) {
    int status = NVSHMEMX_SUCCESS;
    void *buf = (char *)geometry_.heap_base + mc_offset;
    size_t granularity = geometry_.mem_granularity;
    size_t adjusted_max_handle_len = granularity * (NVSHMEMI_MAX_HANDLE_LENGTH / granularity);
    char *buf_start = (char *)buf;
    size_t remaining = size;

    /* Register the entire range in one operation for P2P. */
    status = map_dynamic_p2p_chunk(handle, buf, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map_dynamic_p2p_chunk failed\n");

    /* Remote transports limit each registration to NVSHMEMI_MAX_HANDLE_LENGTH. */
    do {
        size_t reg_size = remaining > adjusted_max_handle_len ? adjusted_max_handle_len : remaining;
        assert(reg_size < NVSHMEMI_DMA_BUF_MAX_LENGTH);
        status = register_remote_chunk(buf_start, reg_size, alloc_kind);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "register_remote_chunk failed\n");
        remaining -= reg_size;
        buf_start += reg_size;
    } while (remaining);

    if (alloc_kind == nvshmemi_allocation_kind::EXTERNAL) {
        assert(mmap_allocated_range.has_value());
        table_->set_mmap_allocated_range(*mmap_allocated_range);
    }

out:
    return status;
}

int nvshmemi_heap_registration::unregister_vmm_chunk(off_t mc_offset, size_t size) {
    int status = NVSHMEMX_SUCCESS;
    nvshmemi_mem_remote_transport &remote_tran = remote_transport();
    size_t granularity = geometry_.mem_granularity;
    size_t adjusted_max_handle_len = granularity * (NVSHMEMI_MAX_HANDLE_LENGTH / granularity);
    void *ptr = (char *)geometry_.heap_base + mc_offset;
    void *curr_ptr = ptr;
    size_t remaining_size = size;
    size_t addr_idx;
    size_t register_size = 0;
    size_t handle_idx = 0;

    auto &mmap_reg = table_->mmap_reg();

    /* Process the same chunks used for remote registration. */
    do {
        register_size =
            remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;
        bool local_handle_set_released = false;

        for (size_t idx = 0; idx < register_size / granularity; ++idx) {
            addr_idx =
                ((char *)curr_ptr - (char *)geometry_.heap_base) >> geometry_.log2_mem_granularity;
            addr_idx += idx;

            /* A failed registration may not have published this chunk's sparse index. */
            if (!mmap_reg.has_index(addr_idx)) {
                continue;
            }

            const auto &entry = mmap_reg.get_index(addr_idx);
            status = ((entry.start_addr != curr_ptr) || (entry.size != register_size));
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "address indexing error\n");

            handle_idx = entry.handle_idx;
            /* Release each handle set once per registered chunk. */
            if (!local_handle_set_released) {
                nvshmem_mem_handle_t *my_handles = mmap_reg.get_mem_handle(handle_idx, mype_, 0);
                status = remote_tran.release_mem_handles(my_handles, transports_);
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release mem handles failed for mmaped buffer \n");
                local_handle_set_released = true;
            }
            /* Clear index entries to prevent stale lookups if the buffer is remapped. */
            mmap_reg.clear_index(addr_idx);
        }

        assert(remaining_size >= register_size);
        remaining_size -= register_size;
        curr_ptr = (char *)curr_ptr + register_size;
    } while (remaining_size);

out:
    return status;
}

int nvshmemi_heap_registration::plan_vmm_peer_bases() {
    int status = NVSHMEMX_SUCCESS;
    int p2p_counter = 1;

    for (int i = ((mype_ + 1) % npes_); i != mype_; i = ((i + 1) % npes_)) {
        for (int j = 0; j < transports_.num_transports(); j++) {
            if (!transports_.active_between_has_cap(mype_, i, npes_, j,
                                                    NVSHMEM_TRANSPORT_CAP_MAP)) {
                continue;
            }
            void *peer_base = (void *)((uintptr_t)geometry_.global_heap_base +
                                       geometry_.logical_heap_size * p2p_counter++);
            peer_heap_base_p2p_[i] = peer_base;
            INFO(NVSHMEM_MEM, "[%d] Peer Heap Base [%d]: %p\n", mype_, i, peer_heap_base_p2p_[i]);
            break;
        }
    }

    nvshmemi_mem_p2p_transport &p2ptran = p2p_transport();
    /* Retrieve the PE-to-process map or initialize it once. */
    if (p2ptran.get_proc_map().size() == 0) {
        status = p2ptran.create_proc_map(npes_, transports_);
        if (p2ptran.get_proc_map().size() == 0) {
            INFO(NVSHMEM_MEM,
                 "Peer PE to PID map (nvshmemi_mem_p2p_transport::proc_map_) is empty as either "
                 "P2P is disabled or P2P initialized failed\n");
        }
    }

    return status;
}

int nvshmemi_heap_registration::register_static_heap() {
    void *buf = geometry_.heap_base;
    size_t size = geometry_.logical_heap_size;
    if (size == 0) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    size_t remaining_size;
    size_t registration_size;
    char *buf_start = (char *)buf;
    int status = NVSHMEMX_SUCCESS;
    size_t granularity = geometry_.mem_granularity;
    size_t adjusted_max_handle_len = granularity * (NVSHMEMI_MAX_HANDLE_LENGTH / granularity);

    assert(buf != nullptr);

    if (policy_ == nvshmemi_heap_registration_policy::STATIC_VIDMEM_FULL_HEAP) {
        status = map_static_vidmem_p2p_chunk(buf);
    } else {
        status = map_static_sysmem_p2p_chunk();
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "static heap P2P mapping failed\n");

    if (nvshmemi_device_state.enable_rail_opt == 1) {
        status = register_remote_chunk(buf, size, nvshmemi_allocation_kind::INTERNAL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "register_remote_chunk on heap static \n");
    } else {
        remaining_size = size;
        /* Remote transports register at most NVSHMEMI_MAX_HANDLE_LENGTH per chunk. */
        do {
            registration_size =
                remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;
            assert(registration_size < NVSHMEMI_DMA_BUF_MAX_LENGTH);
            status = register_remote_chunk(buf_start, registration_size,
                                           nvshmemi_allocation_kind::INTERNAL);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "register_remote_chunk on heap static \n");

            assert(remaining_size >= registration_size);
            remaining_size -= registration_size;
            buf_start += registration_size;
        } while (remaining_size);
    }

out:
    return status;
}
