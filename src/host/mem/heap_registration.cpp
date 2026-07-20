/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>
#include <cstring>
#include <map>
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
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmemx_error.h"

namespace {

int load_posix_fd(const nvshmem_mem_handle_t &handle) noexcept {
    static_assert(sizeof(int) <= sizeof(handle));
    int fd;
    std::memcpy(&fd, &handle, sizeof(fd));
    return fd;
}

void store_posix_fd(nvshmem_mem_handle_t *handle, int fd) noexcept {
    assert(handle != nullptr);
    static_assert(sizeof(fd) <= sizeof(*handle));
    std::memcpy(handle, &fd, sizeof(fd));
}

int consume_cuda_runtime_error_for_failed_ipc(cudaError_t api_status, const char *api_name) {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) return NVSHMEMX_SUCCESS;
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

/* Close imported POSIX FDs retained in P2P handle sets. */
nvshmemi_heap_registration::~nvshmemi_heap_registration() {
    if (effective_handle_type_ != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) return;

    for (int i = 0; i < npes_; i++) {
        for (int j = 0; j < transports_.num_transports(); j++) {
            bool is_p2p_transport = transports_.active_has_cap(j, i, NVSHMEM_TRANSPORT_CAP_MAP);
            if (!is_p2p_transport) continue;

            for (size_t k = 0; k < table_->num_p2p_handle_sets(); k++) {
                close(load_posix_fd(table_->get_p2p_mem_handle(k, i, j)));
            }
        }
    }
}

int nvshmemi_heap_registration::teardown() {
    nvshmemi_mem_remote_transport &remote_tran = remote_transport();
    auto &internal_reg = table_->internal_reg();
    int first_status = NVSHMEMX_SUCCESS;

    for (size_t j = 0; j < internal_reg.num_handle_sets(); j++) {
        if (j > 0 && nvshmemi_device_state.enable_rail_opt) continue;
        nvshmem_mem_handle_t *handles = internal_reg.get_mem_handle(j, mype_, 0);
        int status = remote_tran.release_mem_handles(handles, transports_);
        if (status) {
            if (first_status == NVSHMEMX_SUCCESS) first_status = status;
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
    if (nvshmemi_host_hashes == nullptr) return pe_id % npes_node_;

    int local_index = 0;
    for (int i = 0; i < pe_id; i++) {
        if (nvshmemi_host_hashes[i] == nvshmemi_host_hashes[pe_id]) local_index++;
    }
    return local_index;
}

int nvshmemi_heap_registration::map_p2p_chunk(nvshmem_mem_handle_t *handle, void *buf,
                                              size_t size) {
    int status = NVSHMEMX_SUCCESS;
    std::vector<nvshmem_mem_handle_t> local_handles(transports_.num_transports());
    std::vector<nvshmem_mem_handle_t> gathered(transports_.num_transports() * npes_);

    /* Iterate over P2P transports. */
    for (int i = 0; i < transports_.num_transports(); i++) {
        if (!transports_.active_has_cap(i, mype_, NVSHMEM_TRANSPORT_CAP_MAP)) continue;
        status = export_p2p_memory(&local_handles[i], buf, size, handle);
        if (status != NVSHMEMX_SUCCESS) break;
    }
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "export_memory failed for p2p on at least one PE\n");

    /* Allgather memory handles for all PEs. */
    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles.data(), (void *)gathered.data(),
        sizeof(nvshmem_mem_handle_t) * transports_.num_transports(), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of p2p mem handles failed\n");

    /* Exchange send/receive memory handles for P2P-connected PEs. */
    status = exchange_p2p_memory_handle(&local_handles[0], gathered.data());
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "exchange_p2p_memory_handle failed\n");

    /* Map handles for all mapping-capable transports. */
    status = map_p2p_range(buf, size, gathered);
    status = nvshmemi_bootstrap_aggregate_status(status, npes_);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "map_p2p_range failed on at least one PE\n");

    table_->push_p2p_mem_handles(std::move(gathered));

out:
    return status;
}

int nvshmemi_heap_registration::map_p2p_range(void *buf, size_t size,
                                              std::vector<nvshmem_mem_handle_t> &gathered_handles) {
    nvshmemi_mem_p2p_transport &p2ptran = p2p_transport();
    int status = NVSHMEMX_SUCCESS;
    int i = (mype_ + 1) % npes_;
    while (i != mype_) {
        for (int j = 0; j < transports_.num_transports(); j++) {
            if (!transports_.active_between_has_cap(mype_, i, npes_, j, NVSHMEM_TRANSPORT_CAP_MAP))
                continue;
            INFO(NVSHMEM_MEM, "Mapping Buf: %p Size: %zu PE ID: %d, P2P Transport Idx: %d\n", buf,
                 size, i, j);
            nvshmem_mem_handle_t *in_handle =
                &gathered_handles[i * transports_.num_transports() + j];
            status = map_p2p_memory(i, in_handle, buf, size);
            if (status) {
                if (status != NVSHMEMX_ERROR_INVALID_VALUE) return status;
                /* Map failed: remove all map-related capabilities. */
                transports_.clear_cap(j, i,
                                      NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                                          NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD |
                                          NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS);
                status = NVSHMEMX_SUCCESS;
                peer_heap_base_p2p_[i] = nullptr;
                continue;
            }

            p2ptran.print_mem_handle(in_handle, mype_);
            INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle tobuf %p", mype_, peer_heap_base_p2p_[i]);
            break;
        }

        i = (i + 1) % npes_;
    }

    return status;
}

int nvshmemi_heap_registration::map_p2p_memory(int pe_id, nvshmem_mem_handle_t *in_handle,
                                               void *buf, size_t size) {
    switch (policy_) {
        case nvshmemi_heap_registration_policy::DYNAMIC_VMM: {
            CUdevice gpu_device_id;
            int status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuCtxGetDevice failed\n");
                return NVSHMEMX_ERROR_INTERNAL;
            }

            CUmemGenericAllocationHandle peer_handle;
            if (effective_handle_type_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
                const int fd = load_posix_fd(*in_handle);
                status =
                    CUPFN(nvshmemi_cuda_syms,
                          cuMemImportFromShareableHandle(
                              &peer_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
                              effective_handle_type_));
            } else {
                status = CUPFN(
                    nvshmemi_cuda_syms,
                    cuMemImportFromShareableHandle(
                        &peer_handle, reinterpret_cast<void *>(in_handle), effective_handle_type_));
            }
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT(
                    "cuMemImportFromShareableHandle failed state->device_id : %d \n",
                    gpu_device_id);
                return NVSHMEMX_ERROR_INTERNAL;
            }

            void *buf_map = reinterpret_cast<void *>(
                static_cast<char *>(buf) - static_cast<char *>(geometry_.heap_base) +
                static_cast<char *>(peer_heap_base_p2p_[pe_id]));
            INFO(NVSHMEM_MEM,
                 "calling cuMemMap on buf: %p size: %zu heap_base: %p "
                 "peer_heap_base_p2p[%d]: %p\n",
                 buf, size, geometry_.heap_base, pe_id, peer_heap_base_p2p_[pe_id]);

            const auto buf_map_device_ptr =
                static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(buf_map));
            status =
                CUPFN(nvshmemi_cuda_syms, cuMemMap(buf_map_device_ptr, size, 0, peer_handle, 0));
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuMemMap failed to map %zu bytes handle at address: %p\n",
                                     size, buf_map);
                (void)CUPFN(nvshmemi_cuda_syms, cuMemRelease(peer_handle));
                return NVSHMEMX_ERROR_INTERNAL;
            }

            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = gpu_device_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(peer_handle));
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuMemRelease failed \n");
                return NVSHMEMX_ERROR_INTERNAL;
            }

            status =
                CUPFN(nvshmemi_cuda_syms, cuMemSetAccess(buf_map_device_ptr, size, &access, 1));
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuMemSetAccess failed \n");
                return NVSHMEMX_ERROR_INTERNAL;
            }
            return NVSHMEMX_SUCCESS;
        }
        case nvshmemi_heap_registration_policy::STATIC_VIDMEM_FULL_HEAP: {
            auto *ipc_handle = reinterpret_cast<cudaIpcMemHandle_t *>(in_handle);
            const cudaError_t cuda_status = cudaIpcOpenMemHandle(
                &peer_heap_base_p2p_[pe_id], *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            if (cuda_status != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("cudaIpcOpenMemHandle failed with error %d (%s) \n",
                                     cuda_status, cudaGetErrorString(cuda_status));
                int status =
                    consume_cuda_runtime_error_for_failed_ipc(cuda_status, "cudaIpcOpenMemHandle");
                if (status != NVSHMEMX_SUCCESS) return status;
                return NVSHMEMX_ERROR_INVALID_VALUE;
            }
            return NVSHMEMX_SUCCESS;
        }
        case nvshmemi_heap_registration_policy::STATIC_SYSMEM_FULL_HEAP:
            if (!is_node_local_pe(pe_id)) return NVSHMEMX_ERROR_INVALID_VALUE;
            peer_heap_base_p2p_[mype_] = geometry_.heap_base;
            peer_heap_base_p2p_[pe_id] = static_cast<char *>(geometry_.global_heap_base) +
                                         node_local_index(pe_id) * geometry_.logical_heap_size;
            return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemi_heap_registration::exchange_p2p_memory_handle(nvshmem_mem_handle_t *local_handle,
                                                           nvshmem_mem_handle_t *recv_handles) {
    /* POSIX handles are exchanged for intra-node GPU communication. */
    if (policy_ != nvshmemi_heap_registration_policy::DYNAMIC_VMM ||
        effective_handle_type_ != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        return NVSHMEMX_SUCCESS;
    }

    int status = NVSHMEMX_SUCCESS;
    ipcHandle *myIpcHandle = nullptr;
    std::map<pid_t, ipcHandle *> recvIpcHandles;
    pid_t pid = getpid();
    const auto p2p_processes = p2p_transport().get_proc_map();

    NVSHMEMI_IPC_CHECK(ipcOpenSocket(myIpcHandle, pid, pid));
    /**
     * myIpcHandle PE0: /tmp/socket-100-100
     * myIpcHandle PE1: /tmp/socket-101-101
     *
     * recvIpcHandle PE0: /tmp/socket-101-100
     * recvIpcHandle PE1: /tmp/socket-100-101
     *
     * sendFd PE0: myIpcHandle from 100 to 101
     * sendFd PE1: myIpcHandle from 101 to 100
     */

    /* Open all sockets. */
    for (const auto &entry : p2p_processes) {
        pid_t sending_process = entry.first;
        if (pid == sending_process) continue;

        ipcHandle *recvIpcHandle = nullptr;
        NVSHMEMI_IPC_CHECK(ipcOpenSocket(recvIpcHandle, sending_process, pid));
        recvIpcHandles[sending_process] = recvIpcHandle;
    }

    /* Wait for all processes to open their sockets. */
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, close_sockets,
                          "bootstrap barrier failed before FD exchange\n");

    /* Send all FDs. */
    for (const auto &entry : p2p_processes) {
        pid_t receiving_process = entry.first;
        if (pid == receiving_process) continue;

        NVSHMEMI_IPC_CHECK(
            ipcSendFd(myIpcHandle, load_posix_fd(*local_handle), pid, receiving_process));
    }

    /* Receive all FDs. */
    for (const auto &[sending_process, pe] : p2p_processes) {
        if (pid == sending_process) continue;

        int received_fd;
        NVSHMEMI_IPC_CHECK(ipcRecvFd(recvIpcHandles[sending_process], &received_fd));
        store_posix_fd(&recv_handles[pe * transports_.num_transports()], received_fd);
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
close_sockets:
    /* Close all sockets after the exchange or an early barrier failure. */
    NVSHMEMI_IPC_CHECK(ipcCloseSocket(myIpcHandle));
    for (const auto &entry : recvIpcHandles) {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(entry.second));
    }

    return status;
}

int nvshmemi_heap_registration::export_p2p_memory(nvshmem_mem_handle_t *out, void *buf, size_t size,
                                                  nvshmem_mem_handle_t *vmm_handle) {
    switch (policy_) {
        case nvshmemi_heap_registration_policy::DYNAMIC_VMM: {
            if (vmm_handle == nullptr) {
                NVSHMEMI_ERROR_PRINT("cuMemExportToShareableHandle requires a VMM handle\n");
                return NVSHMEMX_ERROR_INVALID_VALUE;
            }

            CUmemGenericAllocationHandle *handle_in =
                reinterpret_cast<CUmemGenericAllocationHandle *>(vmm_handle);
            INFO(NVSHMEM_MEM, "calling cuMemExportToShareableHandle on handle: %p", handle_in);
            const int status = CUPFN(
                nvshmemi_cuda_syms,
                cuMemExportToShareableHandle((void *)out, *handle_in, effective_handle_type_, 0));
            if (status != CUDA_SUCCESS) {
                NVSHMEMI_ERROR_PRINT("cuMemExportToShareableHandle failed \n");
                return NVSHMEMX_ERROR_INTERNAL;
            }
            return NVSHMEMX_SUCCESS;
        }
        case nvshmemi_heap_registration_policy::STATIC_VIDMEM_FULL_HEAP: {
            auto *ipc_handle = reinterpret_cast<cudaIpcMemHandle_t *>(out);
            assert(sizeof(cudaIpcMemHandle_t) <= NVSHMEM_MEM_HANDLE_SIZE);
            INFO(NVSHMEM_MEM, "calling cuIpcGetMemHandle on buf: %p size: %zu", buf, size);
            const cudaError_t cuda_status = cudaIpcGetMemHandle(ipc_handle, buf);
            if (cuda_status != cudaSuccess) {
                NVSHMEMI_ERROR_PRINT("cudaIpcGetMemHandle failed with error %d (%s)\n", cuda_status,
                                     cudaGetErrorString(cuda_status));
                int status =
                    consume_cuda_runtime_error_for_failed_ipc(cuda_status, "cudaIpcGetMemHandle");
                if (status != NVSHMEMX_SUCCESS) return status;
                return NVSHMEMX_ERROR_INVALID_VALUE;
            }
            return NVSHMEMX_SUCCESS;
        }
        case nvshmemi_heap_registration_policy::STATIC_SYSMEM_FULL_HEAP:
            /* No-op for sysmem: the entire heap is mapped to local PEs at allocation time. */
            return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemi_heap_registration::register_remote_chunk(nvshmem_mem_handle_t * /* handle */,
                                                      void *buf, size_t size,
                                                      nvshmemi_allocation_kind alloc_kind) {
    int status = NVSHMEMX_SUCCESS;
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
        if (!transports_.is_active(i) || transports_.has_cap(i, mype_, NVSHMEM_TRANSPORT_CAP_MAP))
            continue;
        status = remotetran.register_mem_handle(&local_handles[0], i, remote_buf, remote_size,
                                                transports_);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "register_mem_handle failed\n");
    }

    /* Allgather memory handles for all PEs. */
    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles.data(), (void *)gathered.data(),
        sizeof(nvshmem_mem_handle_t) * transports_.num_transports(), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed\n");

    /* Store gathered handles in the allocation-specific registry. */
    if (alloc_kind == nvshmemi_allocation_kind::EXTERNAL) {
        table_->mmap_reg().push_mem_handles(std::move(gathered));
    } else {
        table_->internal_reg().push_mem_handles(std::move(gathered));
    }

    {
        nvshmem_mem_handle_t *handle_data =
            alloc_kind == nvshmemi_allocation_kind::EXTERNAL
                ? table_->mmap_reg().get_mem_handle(table_->mmap_reg().num_handle_sets() - 1, 0, 0)
                : table_->internal_reg().get_mem_handle(
                      table_->internal_reg().num_handle_sets() - 1, 0, 0);

        /* Publish rail-optimized full-heap handles exactly once. */
        if (alloc_kind == nvshmemi_allocation_kind::INTERNAL &&
            nvshmemi_device_state.enable_rail_opt == 1) {
            if (!gather_mem_handles_done_) {
                status = remotetran.gather_mem_handles(transports_, handle_data, 0,
                                                       geometry_.logical_heap_size * npes_node_);
                gather_mem_handles_done_ = true;
            }
        } else {
            status = remotetran.gather_mem_handles(
                transports_, handle_data, ((char *)buf - (char *)geometry_.heap_base), size);
        }
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gather_mem_handles failed\n");

    /* Update lookup state with the retrieved memory handles. */
    update_handle_index(buf, size, alloc_kind);

out:
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

    if (table_->empty_handle_cache()) table_->inc_handle_cache();
}

int nvshmemi_heap_registration::register_vmm_chunk(nvshmem_mem_handle_t *handle, off_t mc_offset,
                                                   size_t size, nvshmemi_allocation_kind alloc_kind,
                                                   std::optional<size_t> mmap_allocated_range) {
    int status = NVSHMEMX_SUCCESS;
    void *buf = (char *)geometry_.heap_base + mc_offset;
    size_t granularity = geometry_.mem_granularity;
    size_t adjusted_max_handle_len = granularity * (NVSHMEMI_MAX_HANDLE_LENGTH / granularity);
    char *buf_start = (char *)buf;
    size_t remaining = size;

    /* Register the entire range in one operation for P2P. */
    status = map_p2p_chunk(handle, buf, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map_p2p_chunk failed\n");

    /* Remote transports limit each registration to NVSHMEMI_MAX_HANDLE_LENGTH. */
    do {
        size_t reg_size = remaining > adjusted_max_handle_len ? adjusted_max_handle_len : remaining;
        assert(reg_size < NVSHMEMI_DMA_BUF_MAX_LENGTH);
        status = register_remote_chunk(handle, buf_start, reg_size, alloc_kind);
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

        for (size_t idx = 0; idx < register_size / granularity; ++idx) {
            addr_idx =
                ((char *)curr_ptr - (char *)geometry_.heap_base) >> geometry_.log2_mem_granularity;
            addr_idx += idx;

            const auto &entry = mmap_reg.get_index(addr_idx);
            status = ((entry.start_addr != curr_ptr) || (entry.size != register_size));
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "address indexing error\n");

            handle_idx = entry.handle_idx;
            /* Release each handle set once per registered chunk. */
            if (!idx) {
                nvshmem_mem_handle_t *my_handles = mmap_reg.get_mem_handle(handle_idx, mype_, 0);
                status = remote_tran.release_mem_handles(my_handles, transports_);
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release mem handles failed for mmaped buffer \n");
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
            if (!transports_.active_between_has_cap(mype_, i, npes_, j, NVSHMEM_TRANSPORT_CAP_MAP))
                continue;
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

    /* Register the entire heap in one operation for P2P. */
    status = map_p2p_chunk(nullptr, buf, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map_p2p_chunk on heap static \n");

    if (nvshmemi_device_state.enable_rail_opt == 1) {
        status = register_remote_chunk(nullptr, buf, size, nvshmemi_allocation_kind::INTERNAL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "register_remote_chunk on heap static \n");
    } else {
        remaining_size = size;
        /* Remote transports register at most NVSHMEMI_MAX_HANDLE_LENGTH per chunk. */
        do {
            registration_size =
                remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;
            assert(registration_size < NVSHMEMI_DMA_BUF_MAX_LENGTH);
            status = register_remote_chunk(nullptr, buf_start, registration_size,
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
