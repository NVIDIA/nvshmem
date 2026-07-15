/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_nvls_observer.hpp"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <typeinfo>
#include <vector>
#include "device_host/nvshmem_types.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvshmemi_mem_transport.hpp"
#include "internal/host/nvshmemi_team.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/sockets.h"
#include "internal/host/util.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host/nvshmemi_nvls_rsc.hpp"

using namespace nvls;

namespace {
bool nvshmemi_should_process_nvls_team_pool_entry(size_t team_idx) {
    const size_t mc_shared_idx = static_cast<size_t>(NVSHMEM_TEAM_MC_SHARED_INDEX);
    const size_t shared_idx = static_cast<size_t>(NVSHMEM_TEAM_SHARED_INDEX);

    if (nvshmemi_team_pool == NULL || nvshmemi_max_teams <= 0) return false;

    const size_t max_teams = static_cast<size_t>(nvshmemi_max_teams);
    if (team_idx >= max_teams || nvshmemi_team_pool[team_idx] == NULL) return false;

    bool is_mc_shared_alias = team_idx == mc_shared_idx && max_teams > mc_shared_idx &&
                              nvshmemi_team_pool[mc_shared_idx] == nvshmemi_team_pool[shared_idx];

    return !is_mc_shared_alias && nvshmemi_team_support_nvls(nvshmemi_team_pool[team_idx]);
}
}  // namespace

/* ---- nvshmemi_heap_observer callbacks ---- */

int nvshmemi_nvls_observer::on_chunk_mapped(nvshmem_mem_handle_t *handle, off_t mc_offset,
                                            off_t mmap_offset, size_t size) {
    return nvls_bind_heap_memory(handle, mc_offset, mmap_offset, size);
}

int nvshmemi_nvls_observer::on_chunk_unmapped(off_t mc_offset, size_t size) {
    if (!state_->is_platform_nvls) return 0;
    INFO(NVSHMEM_MEM, "unbinding and releasing nvls memory mc_offset: %ld size: %zu\n", mc_offset,
         size);
    return nvls_unbind_heap_memory_by_size(mc_offset, size);
}

/* ---- Broadcast helpers ---- */

int nvshmemi_nvls_observer::nvls_broadcast_heap_handle_ipc(char *shareable_handle, int root,
                                                           nvshmemi_team_t *team) {
    pid_t pid = getpid();
    ipcHandle *myIpcHandle = NULL;
    ipcHandle *recvIpcHandle = NULL;
    auto p2p_processes = heap_->get_p2pref()->get_proc_map();
    pid_t root_process;
    int fd = -1;
    /**
     * myIpcHandle PE0: /tmp/socket-100-100
     * recvIpcHandle PE1: /tmp/socket-100-101
     *
     * sendFd PE0: myIpcHandle from 100 to 101
     * recvFd PE1: recvIpcHandle from 100 to 101
     */

    if (team->my_pe == root) {
        /* Open socket to send to all processes */
        NVSHMEMI_IPC_CHECK(ipcOpenSocket(myIpcHandle, pid, pid));
        root_process = pid;
    } else {
        /* Open socket to recv from root process */
        for (auto it = p2p_processes.begin(); it != p2p_processes.end(); ++it) {
            if (it->second == nvshmemi_team_pe(team, root)) {
                root_process = it->first;
                NVSHMEMI_IPC_CHECK(ipcOpenSocket(recvIpcHandle, root_process, pid));
                break;
            }
        }
    }

    /* Wait for all processes in the team to open their sockets */
    nvshmem_barrier(team->team_idx);

    if (root == team->my_pe) {
        /* Send fd from root to all */
        for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin(); it1 != p2p_processes.end();
             ++it1) {
            pid_t receiving_process = it1->first;
            if (pid != receiving_process &&
                nvshmemi_team_translate_pe(nvshmemi_team_pool[NVSHMEM_TEAM_WORLD], it1->second,
                                           team) != NVSHMEM_TEAM_INVALID) {
                /* Don't send to yourself or don't send it to a PE not in the team */
                NVSHMEMI_IPC_CHECK(
                    ipcSendFd(myIpcHandle, *(int *)shareable_handle, pid, receiving_process));
            }
        }
        INFO(NVSHMEM_INIT, "Sending shareable handle from PID %d over IPC Socket Handle %p\n", pid,
             myIpcHandle);
    } else {
        /* Recv fd at all from root */
        NVSHMEMI_IPC_CHECK(ipcRecvFd(recvIpcHandle, &fd));
        INFO(NVSHMEM_INIT,
             "Receiving shareable handle to PID %d over IPC Socket Handle %p => converted fd "
             "%d\n",
             pid, recvIpcHandle, fd);
    }

    /* Wait for all processes to finish send/recv */
    nvshmem_barrier(team->team_idx);
    if (team->my_pe == root) {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(myIpcHandle));
        fd = *(int *)shareable_handle;
        close(fd);
    } else {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(recvIpcHandle));
        memcpy(shareable_handle, &fd, sizeof(int));
    }

    return 0;
}

int nvshmemi_nvls_observer::nvls_broadcast_heap_handle_fabric(char *buffer, size_t length, int root,
                                                              nvshmemi_team_t *team) {
    /* This is technical debt where we are using the REDUCE op psync as scratchpad for src/dst of
     * broadcast broadcast's psync is used for LL8 and other algorithms, making it non-trivial to
     * share when issued from the host as a src or dest buffer.
     *
     * When reduce coll supports LL8 algorithm, we need to clean this up as a independent scratch
     * space
     */
    long *pWrk = nvshmemi_team_get_psync(team, REDUCE);
    if (team->my_pe == root) {
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, buffer, length, cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < team->size; i++) {
            int next_pe = nvshmemi_team_translate_pe_to_team_world_wrap(team, i);
            nvshmemx_char_put_nbi_on_stream((char *)pWrk, (const char *)pWrk, length, next_pe,
                                            (cudaStream_t)0);
        }
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier(team->team_idx);
    } else {
        nvshmem_barrier(team->team_idx);
        CUDA_RUNTIME_CHECK(cudaMemcpy(buffer, pWrk, length, cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }
    return 0;
}

int nvshmemi_nvls_observer::nvls_broadcast_heap_handle_by_team(char *buffer, size_t length,
                                                               nvshmemi_team_t *team) {
    int status = NVSHMEMX_ERROR_INTERNAL;
    int root = 0; /* every team's PE0 */
    if (heap_->is_cuda_mem_handle_type_fabric()) {
        status = nvls_broadcast_heap_handle_fabric(buffer, length, root, team);
    } else {
        status = nvls_broadcast_heap_handle_ipc(buffer, root, team);
    }
    return status;
}

/* ---- Per-size helpers ---- */

int nvshmemi_nvls_observer::nvls_create_heap_memory_by_size(nvshmemi_team_t *team,
                                                            uint64_t mem_size) {
    int status = -1;
    char shareable_handle[64] = {0};
    CUmemGenericAllocationHandle *my_handle;
    CUmemGenericAllocationHandle peer_handle;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* team PE0 will export MC group */
    if (team->my_pe == 0) {
        status = nvls_obj->export_group(mem_size, &shareable_handle[0]);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Exporting multicast group failed for pe %d\n", team->my_pe);

        /* Get the most recently allocated mc_handle */
        my_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
        NVSHMEMI_NULL_ERROR_JMP(my_handle, status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                                "No active multicast group for pe %d\n", team->my_pe);

        status = nvls_broadcast_heap_handle_by_team(&shareable_handle[0], sizeof(shareable_handle),
                                                    team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Broadcasting exported multicast group for pe %d failed\n",
                              team->my_pe);

        status = nvls_obj->subscribe_group(my_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Subscribing multicast group failed for pe %d\n", team->my_pe);
    } else {
        status = nvls_broadcast_heap_handle_by_team(&shareable_handle[0], sizeof(shareable_handle),
                                                    team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Broadcasting exported multicast group for pe %d failed\n",
                              team->my_pe);

        status = nvls_obj->import_group(&shareable_handle[0], &peer_handle, mem_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Importing multicast group failed for pe %d\n", team->my_pe);

        status = nvls_obj->subscribe_group(&peer_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Subscribing multicast group failed for pe %d\n", team->my_pe);
    }

    nvshmem_barrier(team->team_idx);
cleanup:
    return status;
}

int nvshmemi_nvls_observer::nvls_create_heap_memory(uint64_t mem_size) {
    nvshmemi_team_t *team = NULL;
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */
    if (!state_->is_platform_nvls) return status;

    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams, nvshmemi_should_process_nvls_team_pool_entry(i), {
        team = nvshmemi_team_pool[i];
        status = nvls_create_heap_memory_by_size(team, mem_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Creating mc handle for team ID: %d failed\n", team->team_idx);
        INFO(NVSHMEM_INIT, "Setting up mcHandle for team ID: %d\n", team->team_idx);
    });

cleanup:
    return status;
}

int nvshmemi_nvls_observer::nvls_bind_heap_memory_by_size(nvshmemi_team_t *team,
                                                          nvshmem_mem_handle_t *mem_handle,
                                                          off_t mc_offset, off_t mmap_offset,
                                                          size_t mmap_size) {
    int status = -1;
    CUmemGenericAllocationHandle *mc_handle = NULL;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* Get the most recently allocated mc_handle */
    mc_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
    NVSHMEMI_NULL_ERROR_JMP(mc_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "No active MC group for team idx %d\n", team->team_idx);

    INFO(NVSHMEM_MEM,
         "type: %s binding multicast group %ld to memory handle %p mmap size %zu, mc "
         "offset %lx mmap offset %lx\n",
         typeid(decltype(this)).name(), *mc_handle, mem_handle, mmap_size, mc_offset, mmap_offset);
    status = nvls_obj->bind_group_mem(mc_handle, mem_handle, mmap_size, mmap_offset, mc_offset);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Binding mem_handle %p to MC group %lld failed \n", mem_handle,
                          *mc_handle);
out:
    if (status) {
        heap_->print_cumem_handles();
        exit(1); /* Treating bind errors as a fatal error */
    }
    return status;
}

int nvshmemi_nvls_observer::nvls_bind_heap_memory(nvshmem_mem_handle_t *mem_handle, off_t mc_offset,
                                                  off_t mmap_offset, size_t mmap_size) {
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */
    std::vector<nvshmemi_team_t *> bound_teams;
    if (!state_->is_platform_nvls) return status;

    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams, nvshmemi_should_process_nvls_team_pool_entry(i), {
        nvshmemi_team_t *team = nvshmemi_team_pool[i];
        status = nvls_bind_heap_memory_by_size(team, mem_handle, mc_offset, mmap_offset, mmap_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Binding MC handle for team ID: %d failed\n", team->team_idx);
        bound_teams.push_back(team);
        INFO(NVSHMEM_INIT, "Binding mc handle for team ID: %d\n", team->team_idx);

        if (heap_->le_multicast_enabled_) {
            status = heap_->nvls_bind_multicast_endpoint(
                team, *reinterpret_cast<CUmemGenericAllocationHandle *>(mem_handle), mc_offset,
                mmap_offset, mmap_size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Binding multicast endpoint for team ID: %d failed\n",
                                  team->team_idx);
            INFO(NVSHMEM_INIT, "Binding multicast endpoint for team ID: %d\n", team->team_idx);
        }
    });
out:
    if (status != NVSHMEMX_SUCCESS) {
        while (!bound_teams.empty()) {
            nvshmemi_team_t *team = bound_teams.back();
            const int cleanup_status = nvls_unbind_heap_memory_by_size(team, mc_offset, mmap_size);
            if (cleanup_status != NVSHMEMX_SUCCESS) {
                NVSHMEMI_WARN_PRINT("Failed to roll back NVLS binding for team ID: %d\n",
                                    team->team_idx);
            }
            bound_teams.pop_back();
        }
    }
    return status;
}

int nvshmemi_nvls_observer::nvls_map_heap_memory_by_size(nvshmemi_team_t *team, uint64_t size,
                                                         off_t mmap_offset, off_t mc_offset) {
    int status = -1;
    CUmemGenericAllocationHandle *mc_handle = NULL;

    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* Get the most recently allocated mc_handle */
    mc_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
    NVSHMEMI_NULL_ERROR_JMP(mc_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "No active MC group for team idx %d\n", team->team_idx);

    INFO(NVSHMEM_MEM,
         "type: %s mapping multicast group %ld of size %zu, mc offset %lx mmap offset %lx\n",
         typeid(decltype(this)).name(), *mc_handle, size, mc_offset, mmap_offset);

    status = nvls_obj->map_group_mem(mc_handle, size, mmap_offset, mc_offset);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Mapping mem size %zu to MC group %lld failed \n", size, *mc_handle);
out:
    if (status) {
        heap_->print_cumem_handles();
    }
    return status;
}

int nvshmemi_nvls_observer::nvls_map_heap_memory(uint64_t size, off_t mmap_offset,
                                                 off_t mc_offset) {
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */
    if (!state_->is_platform_nvls) return status;

    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams, nvshmemi_should_process_nvls_team_pool_entry(i), {
        status = nvls_map_heap_memory_by_size(nvshmemi_team_pool[i], size, mmap_offset, mc_offset);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Mapping MC handle for team ID: %d failed\n",
                              nvshmemi_team_pool[i]->team_idx);
        INFO(NVSHMEM_INIT, "Mapping mc handle for team ID: %d\n", nvshmemi_team_pool[i]->team_idx);
    });
out:
    return status;
}

/* ---- Per-team public methods (called from team_internal.cpp) ---- */

int nvshmemi_nvls_observer::nvls_create_heap_memory_by_team(nvshmemi_team_t *team) {
    return nvls_create_heap_memory_by_size(team, heap_->heap_size_);
}

int nvshmemi_nvls_observer::nvls_bind_heap_memory_by_team(nvshmemi_team_t *team) {
    int status = 0;
    CUmemGenericAllocationHandle mem_handle;
    off_t mc_offset, mmap_offset;
    size_t mmap_size;

    /* Iterate over heap's list of tuple <mem_handle, mc_offset, mmap_offset, mmap_size> */
    NVSHMEMU_FOR_EACH(i, heap_->get_cumem_handle_size()) {
        if (heap_->is_cumem_handle_released(i)) continue;
        mem_handle = heap_->get_cumem_handle_ptr(i);
        mc_offset = heap_->get_cumem_handle_alloc_offset(i);
        mmap_offset = heap_->get_cumem_handle_mmap_offset(i);
        mmap_size = heap_->get_cumem_handle_mmap_size(i);
        /* Bind UC handles to MC handle at heap_offset */
        status = nvls_bind_heap_memory_by_size(team, (nvshmem_mem_handle_t *)&mem_handle, mc_offset,
                                               mmap_offset, mmap_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Binding multicast groups to UC mem handle %lld, mmap size %zu, mc "
                              "offset %ld, mmap offset %ld failed for pe %d team ID %d\n",
                              mem_handle, mmap_size, mc_offset, mmap_offset, team->my_pe,
                              team->team_idx);
        if (heap_->le_multicast_enabled_) {
            status = heap_->nvls_bind_multicast_endpoint(team, mem_handle, mc_offset, mmap_offset,
                                                         mmap_size);
            NVSHMEMI_NZ_ERROR_JMP(
                status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                "Binding multicast endpoint to UC mem handle %lld, mmap size %zu, mc "
                "offset %ld, mmap offset %ld failed for pe %d team ID %d\n",
                mem_handle, mmap_size, mc_offset, mmap_offset, team->my_pe, team->team_idx);
        }
    }

cleanup:
    return status;
}

int nvshmemi_nvls_observer::nvls_map_heap_memory_by_team(nvshmemi_team_t *team) {
    /* Map MC handle + mmap_offset = 0 to mc base + mc_offset=0 */
    return nvls_map_heap_memory_by_size(team, heap_->heap_size_, 0, 0);
}

int nvshmemi_nvls_observer::nvls_unmap_heap_memory_by_size(nvshmemi_team_t *team, off_t mc_offset,
                                                           uint64_t mem_size) {
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) return status;
    status = nvls_obj->unmap_group_mem(mc_offset, mem_size);
    return status;
}

void nvshmemi_nvls_observer::nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team) {
    nvls_unmap_heap_memory_by_size(team, 0, heap_->heap_size_);
}

int nvshmemi_nvls_observer::nvls_unmap_heap_memory(off_t mc_offset, uint64_t size) {
    int status = 0;
    if (!state_->is_platform_nvls) return status;

    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams, nvshmemi_should_process_nvls_team_pool_entry(i), {
        status = nvls_unmap_heap_memory_by_size(nvshmemi_team_pool[i], mc_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unmapping MC handle for team ID: %d failed\n",
                              nvshmemi_team_pool[i]->team_idx);
    });
out:
    return status;
}

void nvshmemi_nvls_observer::nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team) {
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) return;

    /* Here we unbind for only the size that has been bound by real UC handles i.e physical heap
     * size to optimize for performance of unbind
     */
    NVSHMEMU_FOR_EACH(i, nvls_obj->get_mc_handle_size()) {
        nvls_obj->unbind_group_mem(nvls_obj->get_mc_handle_ptr(i), 0,
                                   heap_->physical_internal_heap_size_);

        // unbind mmaped buffers
        for (auto iter = heap_->get_mmapped_buf()->begin(); iter != heap_->get_mmapped_buf()->end();
             ++iter) {
            // iter->first => ptr in heap of mmaped buffer
            // iter->second => size
            off_t mc_offset = (char *)iter->first - (char *)heap_->heap_base_;
            nvls_obj->unbind_group_mem(nvls_obj->get_mc_handle_ptr(i), mc_offset, iter->second);
        }
    }

    if (heap_->physical_internal_heap_size_) {
        status =
            heap_->nvls_unbind_multicast_endpoint(team, 0, heap_->physical_internal_heap_size_);
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "cuLogicalEndpointUnbind for multicast endpoint failed at offset: %lu for size: %zu\n",
            0UL, heap_->physical_internal_heap_size_);
    }

    for (auto iter = heap_->get_mmapped_buf()->begin(); iter != heap_->get_mmapped_buf()->end();
         ++iter) {
        off_t mc_offset = (char *)iter->first - (char *)heap_->heap_base_;
        status = heap_->nvls_unbind_multicast_endpoint(team, mc_offset, iter->second);
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "cuLogicalEndpointUnbind for multicast endpoint failed at offset: %lu for size: %zu\n",
            mc_offset, iter->second);
    }
out:
    return;
}

int nvshmemi_nvls_observer::nvls_unbind_heap_memory_by_size(nvshmemi_team_t *team, off_t mc_offset,
                                                            size_t size) {
    int status = 0;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) return status;

    if (nvls_obj->get_mc_handle_size() == 0) {
        NVSHMEMI_ERROR_PRINT("No active MC group for team ID: %d\n", team->team_idx);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    int unbind_status = nvls_obj->unbind_group_mem(
        nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1), mc_offset, size);
    if (unbind_status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("unbind_group_mem for team ID: %d failed. Status: %d\n",
                             team->team_idx, unbind_status);
        status = NVSHMEMX_ERROR_INTERNAL;
    }

    unbind_status = heap_->nvls_unbind_multicast_endpoint(team, mc_offset, size);
    if (unbind_status != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT(
            "cuLogicalEndpointUnbind for multicast endpoint failed at offset: %lu for size: %zu\n",
            mc_offset, size);
        status = NVSHMEMX_ERROR_INTERNAL;
    }

    return status;
}

int nvshmemi_nvls_observer::nvls_unbind_heap_memory_by_size(off_t mc_offset, size_t size) {
    int status = 0;

    // for all teams unbind mc_handle
    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams, nvshmemi_should_process_nvls_team_pool_entry(i), {
        status = nvls_unbind_heap_memory_by_size(nvshmemi_team_pool[i], mc_offset, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unbinding NVLS memory for team ID: %d failed. Status: %d\n",
                              nvshmemi_team_pool[i]->team_idx, status);
    });
out:
    return status;
}
