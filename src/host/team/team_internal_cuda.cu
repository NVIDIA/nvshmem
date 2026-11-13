/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "non_abi/device/common/nvshmem_common.cuh"
#include "non_abi/device/coll/utils.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "device/nvshmemx_defines.h"
#include "team_internal.h"
#include "internal/host/util.h"

#ifdef __CUDA_ARCH__
static_assert(offsetof(nvshmemi_team_creation_pe_info_t, state_idx) >= CFT_HANDLE_TX_SIZE,
              "pe_in_team must occupy a complete CFT transfer lane");
static_assert(offsetof(nvshmemi_team_creation_psync_t, pe_info) >= CFT_HANDLE_TX_SIZE,
              "uniqueid must occupy a complete CFT transfer lane");
#endif

template <typename TYPE, rdxn_ops_t OP>
extern __global__ void nvshmemi_reduce_kernel(int start, int stride, int size, TYPE *dst,
                                              const TYPE *source, size_t nreduce, TYPE *pWrk,
                                              volatile long *pSync, volatile long *sync_counter);

template <typename T>
__global__ void nvshmemi_init_array_kernel(T *array, int len, T val) {
    for (int i = 0; i < len; i++) {
        array[i] = val;
    }
}

template <typename T>
void nvshmemi_call_init_array_kernel(T *array, int len, T val) {
    nvshmemi_init_array_kernel<T><<<1, 1>>>(array, len, val);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_init_array_kernel<nvshmemi_team_t *>(nvshmemi_team_t **, int,
                                                                 nvshmemi_team_t *);
template void nvshmemi_call_init_array_kernel<long>(long *, int, long);
template void nvshmemi_call_init_array_kernel<uint64_t>(uint64_t *, int, uint64_t);
template void nvshmemi_call_init_array_kernel<nvshmemi_team_creation_pe_info_t>(
    nvshmemi_team_creation_pe_info_t *, int, nvshmemi_team_creation_pe_info_t);
template void nvshmemi_call_init_array_kernel<unsigned char>(unsigned char *, int, unsigned char);

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reduce_kernel(int start, int stride, int size, TYPE *dst, const TYPE *source,
                                 size_t nreduce, TYPE *pWrk, volatile long *pSync,
                                 volatile long *sync_counter) {
    const size_t smem_size =
        nvshmemi_get_tma_cft_smem_size(nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    nvshmemi_reduce_kernel<TYPE, OP>
        <<<1, 1, smem_size>>>(start, stride, size, dst, source, nreduce, pWrk, pSync, sync_counter);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_reduce_kernel<unsigned char, (rdxn_ops)0>(
    int, int, int, unsigned char *, unsigned char const *, unsigned long, unsigned char *,
    long volatile *, long volatile *);

template void nvshmemi_call_reduce_kernel<int, (rdxn_ops)4>(int, int, int, int *, int const *,
                                                            unsigned long, int *, long volatile *,
                                                            long volatile *);

#ifdef __CUDA_ARCH__
__device__ void nvshmemi_team_creation_state_barrier(
    nvshmemi_team_t *myteam, volatile nvshmemi_team_creation_pe_info_t *pe_info,
    nvshmemi_team_creation_pe_state state) {
    int num_peers_complete = 0;
    int mype = nvshmemi_device_state_d.mype;

    nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
    pe_info[mype].state_idx = pe_info[mype].state_idx | state;
    for (int i = 0; i < myteam->size; i++) {
        int remote_pe = myteam->pe_mapping[i];
        if (remote_pe == mype) {
            continue;
        }
        nvshmemi_signal_for_barrier<int>((int *)&pe_info[mype].state_idx, pe_info[mype].state_idx,
                                         remote_pe);
    }

    do {
        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            do {
                if (pe_info[global_pe_index].state_idx & state) {
                    num_peers_complete++;
                }
            } while (!(pe_info[global_pe_index].state_idx & state));
        }
    } while (num_peers_complete < myteam->size);

    if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY) {
        nvshmemi_transfer_enforce_consistency_at_target(false);
    }
}
#endif
template <threadgroup_t SCOPE>
__global__ void nvshmemi_team_index_kernel(
    nvshmemi_team_t *myteam, nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync,
    unsigned char *team_index_array, long N_PSYNC_BYTES) {
#ifdef __CUDA_ARCH__
    extern __shared__ __align__(16) unsigned char nvshmemi_team_index_smem[];
    nvshmemx_give_smem(nvshmemi_team_index_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));

    unsigned char *local_ptr, *peer_ptr;

    local_ptr =
        nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].team_index_array;

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) {
        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_READ_PE_IN_TEAM);

        for (int i = 0; i < N_PSYNC_BYTES; i++) {
            team_index_array[i] = local_ptr[i];
        }

        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            if (global_pe_index == nvshmemi_device_state_d.mype) {
                continue;
            }
            nvshmemi_put_nbi<unsigned char, NVSHMEMI_THREADGROUP_THREAD>(
                local_ptr, local_ptr, N_PSYNC_BYTES, global_pe_index);
        }

        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_WROTE_INDEX);

        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            peer_ptr = nvshmemi_team_creation_psync->pe_info[global_pe_index].team_index_array;
            for (int j = 0; j < N_PSYNC_BYTES; j++) {
                team_index_array[j] &= peer_ptr[j];
            }
        }

        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_DONE);

        myteam->team_idx = nvshmemi_bit_1st_nonzero(team_index_array, N_PSYNC_BYTES);
    }

    nvshmemx_release_smem();
#endif
}

template <threadgroup_t SCOPE>
__global__ void nvshmemi_team_mapping_kernel(
    uint64_t uniqueid, int npes, int *num_peers_found, int *pe_mapping,
    nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync) {
#ifdef __CUDA_ARCH__
    extern __shared__ __align__(16) unsigned char nvshmemi_team_mapping_smem[];
    nvshmemx_give_smem(nvshmemi_team_mapping_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));

    int mype_in_team =
        nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team;
    int nThreads = nvshmemi_threadgroup_size<SCOPE>();
    int myThread = nvshmemi_thread_id_in_threadgroup<SCOPE>();
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
    unsigned char *peer_uniqueid_lane =
        reinterpret_cast<unsigned char *>(
            nvshmemi_tma_data_buffer(reinterpret_cast<uintptr_t>(nvshmemi_team_mapping_smem))) +
        myThread * CFT_HANDLE_TX_SIZE;
#endif

    if (myThread == 0) {
        *num_peers_found = 1;
    }

    __syncthreads();

    /* write my psync and pe indexes to all other teams. */
    for (int i = myThread; i < nvshmemi_device_state_d.npes; i += nThreads) {
        if (i == nvshmemi_device_state_d.mype) {
            pe_mapping[mype_in_team] = nvshmemi_device_state_d.mype;
            pe_mapping[npes + nvshmemi_device_state_d.mype] = mype_in_team;
            continue;
        }
        /* pe_in_team begins an aligned 16-byte lane and state_idx starts at the next lane.
         * Transfer the complete first lane to avoid the unsupported masked sub-16-byte CFT PUT. */
        nvshmemi_put_nbi<unsigned char, NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_REGION_OPERATION_NONE,
                         nvshmemi_tma_use::disallowed>(
            reinterpret_cast<unsigned char *>(
                &nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team),
            reinterpret_cast<const unsigned char *>(
                &nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team),
            CFT_HANDLE_TX_SIZE, i);
    }
    nvshmemi_quiet<SCOPE>();
    if (myThread == 0) {
        nvshmemi_team_creation_psync->uniqueid = uniqueid;
        __threadfence_system();
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    /* wait for all other teams to write their psync and pe indexes to me. */
    do {
        for (int i = myThread; i < nvshmemi_device_state_d.npes; i += nThreads) {
            if (i == nvshmemi_device_state_d.mype) {
                continue;
            }

            uint64_t peer_uniqueid;
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
            const unsigned char *uniqueid_lane =
                reinterpret_cast<const unsigned char *>(&nvshmemi_team_creation_psync->uniqueid);
            if (nvshmemi_is_le_implemented<nvshmemi_le_size_rule::tx_size_multiple>(
                    i, CFT_HANDLE_TX_SIZE, NVSHMEMI_THREADGROUP_THREAD, uniqueid_lane,
                    peer_uniqueid_lane)) {
                /* Scalar g has no CFT route. Fetch its aligned 16-byte lane into a per-thread
                 * shared slot when pointer access is unavailable. */
                nvshmemi_get<unsigned char, NVSHMEMI_THREADGROUP_THREAD,
                             nvshmemi_tma_use::disallowed>(peer_uniqueid_lane, uniqueid_lane,
                                                           CFT_HANDLE_TX_SIZE, i);
                peer_uniqueid = *reinterpret_cast<uint64_t *>(peer_uniqueid_lane);
            } else
#endif
            {
                peer_uniqueid = nvshmemi_g<uint64_t>(&nvshmemi_team_creation_psync->uniqueid, i);
            }

            if (peer_uniqueid == uniqueid) {
                nvshmemi_transfer_syncapi_update_mem();

                // Wait until peer's pe_in_team is valid
                nvshmemi_wait_until_greater_than_equals<int>(
                    (volatile int *)&nvshmemi_team_creation_psync->pe_info[i].pe_in_team,
                    0,  // wait until value >= 0
                    NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE);
                int peer_index_in_team = nvshmemi_team_creation_psync->pe_info[i].pe_in_team;

                // Now we have a valid peer_index_in_team, add to mapping if not already done
                if (pe_mapping[npes + i] == -1) {
                    pe_mapping[peer_index_in_team] = i;
                    pe_mapping[npes + i] = peer_index_in_team;
                    atomicAdd(num_peers_found, 1);
                }
            }
        }
        if (myThread == 0) {
            __threadfence_system();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    } while (*num_peers_found < npes);

    nvshmemx_release_smem();
#endif
}

void nvshmemi_call_team_mapping_kernel(
    uint64_t uniqueid, int npes, int *pe_mapping,
    nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync) {
    int *num_peers_found;
    CUDA_RUNTIME_CHECK(cudaMalloc(&num_peers_found, sizeof(int)));
    const size_t smem_size =
        nvshmemi_get_tma_cft_smem_size(nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    nvshmemi_team_mapping_kernel<NVSHMEMI_THREADGROUP_WARP><<<1, 32, smem_size>>>(
        uniqueid, npes, num_peers_found, pe_mapping, nvshmemi_team_creation_psync);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    CUDA_RUNTIME_CHECK(cudaFree(num_peers_found));
}

void nvshmemi_call_team_index_kernel(nvshmemi_team_t *myteam,
                                     nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync,
                                     long N_PSYNC_BYTES) {
    unsigned char *team_index_array;

    INFO(NVSHMEM_COLL, "in team index kernel mype: %d\n", nvshmemi_state->mype);
    CUDA_RUNTIME_CHECK(cudaMalloc(&team_index_array, N_PSYNC_BYTES));
    const size_t smem_size =
        nvshmemi_get_tma_cft_smem_size(nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    nvshmemi_team_index_kernel<NVSHMEMI_THREADGROUP_THREAD><<<1, 1, smem_size>>>(
        myteam, nvshmemi_team_creation_psync, team_index_array, N_PSYNC_BYTES);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    CUDA_RUNTIME_CHECK(cudaFree(team_index_array));
}
