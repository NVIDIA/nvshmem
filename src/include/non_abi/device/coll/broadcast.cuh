/*
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef BROADCAST_DEVICE_CUH
#define BROADCAST_DEVICE_CUH

#if !defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/common/nvshmemi_tile_utils.cuh"
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_tensor.h"
#if defined(NVSHMEM_ENABLE_ALL_DEVICE_INLINING) || defined(__NVSHMEM_NUMBA_SUPPORT__)
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"

#ifdef __CUDA_ARCH__
template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_intranode_tree_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if (teami->bcast_sync_offset + nelems * sizeof(T) * 2 >
            sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }

    /*if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        nvshmemi_threadgroup_sync<SCOPE>(); // wait for block scoped recvLL/packLL to complete
        int warp_id = myIdx / 32;
        int num_warps = (groupSize + 31) / 32;
        size_t size_per_peer = nelems * sizeof(T) * 2;
        int num_peers = k;
        size_t total_size = size_per_peer * num_peers;
        int size_per_warp = min (nelems * sizeof(T) * 2, 512l);
        for (size_t start = warp_id * size_per_warp; start < total_size; start += num_warps *
    size_per_warp) { int peer_id = start / size_per_peer; size_t offset = start % size_per_peer; int
    child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + peer_id + 1; if
    (child_in_team >= teami->size) break; child_in_team = (child_in_team + PE_root) % teami->size;
            int child =
                nvshmemi_team_translate_pe(teami, child_in_team,
    nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]);
            //printf("sending data %d to %d at offset %llu, peer_id: %d, size_per_peer: %llu,
    nelems: %llu\n", size_per_warp, child, offset, peer_id, size_per_peer, nelems);
            //printf("myIdx: %d, start: %lld, num_warps: %d\n", myIdx, start, num_warps);
            nvshmemi_put_nbi<char, NVSHMEMI_THREADGROUP_WARP>(pWrk + recv_offset +
    offset, pWrk + recv_offset + offset, size_per_warp, child);
        }
    } else  */
    {
        nvshmemi_threadgroup_sync<SCOPE>();
        for (int i = 0; i < k; i++) {
            int child_in_team =
                ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
            if (child_in_team >= teami->size) break;
            child_in_team = (child_in_team + PE_root) % teami->size;
            int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

            nvshmemii_put_nbi<uint64_t, SCOPE>((uint64_t *)(pWrk + recv_offset),
                                               (uint64_t *)(pWrk + recv_offset),
                                               nelems * sizeof(T) / sizeof(uint32_t), child);
        }
    }
    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_internode_tree_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + nelems * sizeof(T) * 2) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        nvshmemi_put_nbi<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
            (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
            nelems * sizeof(T) / sizeof(uint32_t), child);
    }
    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_tree_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + nelems * sizeof(T) * 2) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    /* Do remote transfers first */
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);
        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (is_remote)
            nvshmemi_put_nbi<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
                (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
                nelems * sizeof(T) / sizeof(uint32_t), child);
    }

    /* Do P2P transfers */
    for (int i = 0; i < k; i++) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (!is_remote)
            nvshmemii_put_nbi<uint64_t, SCOPE>((uint64_t *)(pWrk + recv_offset),
                                               (uint64_t *)(pWrk + recv_offset),
                                               nelems * sizeof(T) / sizeof(uint32_t), child);
    }

    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_nonLL_tree_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + sizeof(uint64_t)) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_wait_until<uint64_t>((uint64_t *)(pWrk + recv_offset), NVSHMEM_CMP_EQ, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    /* Do remote transfers first */
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);
        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (is_remote)
            nvshmemi_put_signal<T, NVSHMEMI_THREADGROUP_THREAD>(
                dest, (PE_root == my_pe_in_team) ? source : dest, nelems,
                (uint64_t *)(pWrk + recv_offset), ll_flag, NVSHMEMI_AMO_SIGNAL_SET, child, 1);
    }

    /* Do P2P transfers */
    for (int i = 0; i < k; i++) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (!is_remote)
            nvshmemii_put_signal<T, SCOPE>(dest, (PE_root == my_pe_in_team) ? source : dest, nelems,
                                           (uint64_t *)(pWrk + recv_offset), ll_flag,
                                           NVSHMEMI_AMO_SIGNAL_SET, child, 1);
    }

    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
        teami->bcast_sync_offset +=
            2 * sizeof(uint64_t); /* incrementing minimally by 16 bytes because this buffer is used
                                     by packLL and packLL does 16byte writes */
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_put2all_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int root = nvshmemi_team_translate_pe(team, PE_root, NVSHMEM_TEAM_WORLD_INDEX);
    if (root == nvshmemi_device_state_d.mype) {
        for (i = 0; i < teami->size; i++) {
            int pe = nvshmemi_team_translate_pe_to_team_world_wrap(teami, i);
            nvshmemi_put_nbi<T, SCOPE>(dest, source, nelems, pe);
        }
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_put2all_direct_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int root = nvshmemi_team_translate_pe(team, PE_root, NVSHMEM_TEAM_WORLD_INDEX);
    T *dst_ptr;
    if (root == nvshmemi_device_state_d.mype) {
        for (i = 0; i < teami->size; i++) {
            int pe = nvshmemi_team_translate_pe_to_team_world_wrap(teami, i);
            dst_ptr = (T *)nvshmemi_ptr(dest, pe);
            nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
        }
    }

    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_bcast_hierarchical_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    if (teami->is_team_same_mype_node) {
        nvshmemi_bcast_internode_tree_threadgroup<T, SCOPE>(
            team, dest, source, nelems, PE_root,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    } else if (teami->is_team_node) {
        nvshmemi_bcast_intranode_tree_threadgroup<T, SCOPE>(
            team, dest, source, nelems, PE_root,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    } else {
        int team_npes_node = nvshmemi_team_n_pes(teami->team_node);
        int PE_root_idx_in_team_node = PE_root % team_npes_node;
        int my_idx_in_team_node = nvshmemi_team_my_pe(team) % team_npes_node;
        if (PE_root_idx_in_team_node == my_idx_in_team_node) {
            nvshmemi_bcast_internode_tree_threadgroup<T, SCOPE>(
                teami->team_same_mype_node, dest, source, nelems, PE_root / team_npes_node,
                nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
        }
        nvshmemi_bcast_intranode_tree_threadgroup<T, SCOPE>(
            teami->team_node, dest, dest, nelems, PE_root_idx_in_team_node,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_broadcast_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems, int PE_root) {
    int bcast_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo;
    switch (bcast_algo) {
        case 0:
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0 &&
                nelems * sizeof(T) <= 16384) { /* LL algos */
                if (nvshmemi_team_n_pes(team) > 32 &&
                    nvshmemi_device_state_d.pe_dist ==
                        NVSHMEMI_PE_DIST_BLOCK) { /* hierarchical topo-aware */
                    bcast_algo = 2;
                } else
                    bcast_algo = 3;
            } else /* non-LL algorithm */
                bcast_algo = 4;
            break;
        case 1: /* Brutefoce algorithm: send one to all followed by barrier */
            break;
        case 2: /* Topology aware - two level hierarchical algorithm with LL approach */
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0 &&
                nvshmemi_device_state_d.pe_dist == NVSHMEMI_PE_DIST_BLOCK) {
            } else {
                /*printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n", \
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        case 3: /* Topology unaware tree algrithm with LL approach*/
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0) {
            } else {
                /*printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n",
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        case 4: /* Topology unaware flat tree algrithm with LL approach*/
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0) {
            } else {
                /* printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n",
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        default:
            printf("Specified bcast algo:%d not supported, aborting...\n", bcast_algo);
            assert(0);
            break;
    }

    switch (bcast_algo) {
        case 1: /* Brutefoce algorithm: send one to all followed by barrier */
            if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
                nvshmemi_bcast_put2all_direct_threadgroup<T, SCOPE>(team, dest, source, nelems,
                                                                    PE_root);
            } else {
                nvshmemi_bcast_put2all_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
            }
            break;
        case 2: /* Topology aware - two level hierarchical algorithm with LL approach */
            nvshmemi_bcast_hierarchical_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
            break;
        case 3: /* Topology unaware tree algrithm with LL approach*/
            nvshmemi_bcast_tree_threadgroup<T, SCOPE>(
                team, dest, source, nelems, PE_root,
                nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
            break;
        case 4: /* Topology unaware flat tree algrithm with LL approach*/
            nvshmemi_bcast_nonLL_tree_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root,
                                                            nvshmemi_team_n_pes(team) - 1);
            break;
        default:
            assert(0);
            break;
    }
}

#if defined(__cplusplus) && __cplusplus >= 201703L
// ************** Tile broadcast **************/

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_bcast_threadgroup_v4(
    int4 *dest, const int4 *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /* src_stride_major_dim == 0 && dst_stride_major_dim == 0 for vectorized implementation */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/
    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                    dest + ((j) % nelem_major_dim) +
                    (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                : "memory");
        }
    } else {
        using vtype = int4;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            /* nelem_major_dim is in vector units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u & elem_coord_major in elemType units
                     */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /*end of if else*/
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_bcast_threadgroup_v2(
    uint64_t *dest, const uint64_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /* src_stride_major_dim == 0 && dst_stride_major_dim == 0 for vectorized implementation */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/

    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            asm("ld.global.b64 %0, [%1];"
                : "=l"(val1)
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("multimem.st.global.u64 [%0], %1;" ::"l"(
                    dest + ((j) % nelem_major_dim) +
                    (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "l"(val1)
                : "memory");
        }
    } else {
        using vtype = uint64_t;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            /* nelem_major_dim is in vector units*/
            /* compute elem_coord_major in elemType units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.b64 %0, [%1];"
                    : "=l"(val1)
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("multimem.st.global.u64 [%0], %1;" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "l"(val1)
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u and elem_coord_major in elemType
                     * units */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /*end of if else*/
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_bcast_threadgroup_v1(
    uint32_t *dest, const uint32_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/
    using vtype = uint32_t;
    using cxx_type = uint32_t;
    if constexpr (std::is_empty<tuple_t>::value) {
        cxx_type val;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            asm("ld.global.b32 %0, [%1];"
                : "=r"(val)
                : "l"(source + ((j % nelem_major_dim) * src_stride_major_dim) +
                      ((j / nelem_major_dim) * src_stride_minor_dim)));

            asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                    dest + ((j % nelem_major_dim) * dst_stride_major_dim) +
                    ((j / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(val)
                : "memory");
        }
    } else {
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            /* nelem_major_dim is in vector units*/
            /* compute elem_coord_major in elemType units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);
            cxx_type val;

            /* convert elem_coord_major from elemType to cxx_type units */
            /* no change to elem_coord_minor */
            elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

            for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                /* check if elem is within boundary, use u and elem_coord_major in elemType units */
                if (is_less_than<tuple_t, major_dim>(
                        start_coord,
                        create_coord_tuple<major_dim>(
                            ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                            elem_coord_minor),
                        boundary)) {
                    /* convert strides from vector to cxx_type units */
                    asm("ld.global.b32 %0, [%1];"
                        : "=r"(val)
                        : "l"(reinterpret_cast<const cxx_type *>(source) + (elem_coord_major + u) +
                              (elem_coord_minor * src_stride_minor_dim *
                               (sizeof(vtype) / sizeof(cxx_type)))));

                    asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                            reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                            (elem_coord_minor * dst_stride_minor_dim *
                             (sizeof(vtype) / sizeof(cxx_type)))),
                        "r"(val)
                        : "memory");
                }
            }
        }
    } /*end of if else*/
}

// Select implementation based on the operation, datatype
template <typename vtype, typename T, threadgroup_t scope, typename tuple_t, int major_dim,
          int minor_dim>
__device__ inline void nvshmemi_tile_bcast_nvls_threadgroup_vec(
    nvshmem_team_t team, T *src, T *dst, const int size_major_dim, const int size_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    // src is local, dst is multicast address
    vtype *src_v = reinterpret_cast<vtype *>(src);
    vtype *dst_v = reinterpret_cast<vtype *>(nvshmemx_mc_ptr(team, dst));
    assert((dst_v != nullptr) && "Failed to get multicast ptr for destination");

    int src_stride_minor_dim_v = src_stride_minor_dim;
    if (src_stride_minor_dim > 1) {
        src_stride_minor_dim_v = (src_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_minor_dim_v = dst_stride_minor_dim;
    if (dst_stride_minor_dim > 1) {
        dst_stride_minor_dim_v = (dst_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int src_stride_major_dim_v = src_stride_major_dim;  // keep stride as is if ==1
    if (src_stride_major_dim > 1) {
        src_stride_major_dim_v = (src_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_major_dim_v = dst_stride_major_dim;
    if (dst_stride_major_dim > 1) {
        dst_stride_major_dim_v = (dst_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }

    int nelem_major_dim = (size_major_dim * sizeof(T)) / sizeof(vtype);
    int nelem_minor_dim = size_minor_dim;

    if constexpr (std::is_same<vtype, int4>::value) {
        nvshmemi_tile_bcast_threadgroup_v4<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint64_t>::value) {
        nvshmemi_tile_bcast_threadgroup_v2<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint32_t>::value) {
        nvshmemi_tile_bcast_threadgroup_v1<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);
    }
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_bcast_nvls_dim(nvshmem_team_t team, src_tensor_t src_tensor,
                                                    dst_tensor_t dst_tensor, tuple_t start_coord,
                                                    tuple_t boundary) {
    using T = typename src_tensor_t::value_type;

    // check for vector len == 4
    // Conditions: ptr must be aligned to int4, shape must be a multiple of 16, stride must be a
    // multiple of 16
    if (((size_t)src_tensor.data() % sizeof(int4) == 0) &&
        ((size_t)dst_tensor.data() % sizeof(int4) == 0) &&
        (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(int4)) == 0)) {
        nvshmemi_tile_bcast_nvls_threadgroup_vec<int4, T, scope, tuple_t, major_dim, minor_dim>(
            team, src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary);

    } else if (((size_t)src_tensor.data() % sizeof(uint64_t) == 0) &&
               ((size_t)dst_tensor.data() % sizeof(uint64_t) == 0) &&
               (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0)) {
        nvshmemi_tile_bcast_nvls_threadgroup_vec<uint64_t, T, scope, tuple_t, major_dim, minor_dim>(
            team, src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary);

    } else {  // vector len 1
        nvshmemi_tile_bcast_nvls_threadgroup_vec<uint32_t, T, scope, tuple_t, major_dim, minor_dim>(
            team, src_tensor.data(), dst_tensor.data(), get_shape_element<major_dim>(src_tensor),
            get_shape_element<minor_dim>(src_tensor), get_stride_element<minor_dim>(src_tensor),
            get_stride_element<minor_dim>(dst_tensor), get_stride_element<major_dim>(src_tensor),
            get_stride_element<major_dim>(dst_tensor), start_coord, boundary);
    }
}
// specialize for the vectorization
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope>
__device__ inline void nvshmemi_tile_bcast_nvls_threadgroup(nvshmem_team_t team,
                                                            src_tensor_t src_tensor,
                                                            dst_tensor_t dst_tensor,
                                                            tuple_t start_coord, tuple_t boundary) {
    using T = typename src_tensor_t::value_type;
    if constexpr ((get_constant(safe_get<0>(decltype(src_tensor.stride()){})) == 1) &&
                  (get_constant(safe_get<0>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 0 major
        constexpr int major_dim = 0;
        constexpr int minor_dim = 1;

        ASSERT_FP16_ALIGNMENT(T, src_tensor, dst_tensor, major_dim);
        nvshmemi_tile_bcast_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                     minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                boundary);
    } else if constexpr ((get_constant(safe_get<1>(decltype(src_tensor.stride()){})) == 1) &&
                         (get_constant(safe_get<1>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 1 major
        constexpr int major_dim = 1;
        constexpr int minor_dim = 0;

        ASSERT_FP16_ALIGNMENT(T, src_tensor, dst_tensor, major_dim);
        nvshmemi_tile_bcast_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                     minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                boundary);
    } else {
        // No contiguous dimension found at compile time
        // TODO support when major dimension for src and tensor are different
        if ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1)) {
            constexpr int major_dim = 1;
            constexpr int minor_dim = 0;

            ASSERT_FP16_ALIGNMENT(T, src_tensor, dst_tensor, major_dim);
            nvshmemi_tile_bcast_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                         minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                    boundary);
        } else {
            // setting major_dim to 0, minor_dim to 1
            constexpr int major_dim = 0;
            constexpr int minor_dim = 1;

            ASSERT_FP16_ALIGNMENT(T, src_tensor, dst_tensor, major_dim);
            nvshmemi_tile_bcast_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                         minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                    boundary);
        }
    }
}

#endif /* __cplusplus >= 201703L */

// Tile broadcast entrypoint
template <nvshmemx::tile_coll_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope>
__device__ inline int nvshmemi_tile_bcast(nvshmem_team_t team, src_tensor_t src_tensor,
                                          dst_tensor_t dst_tensor, tuple_t start_coord,
                                          tuple_t boundary, uint64_t flag) {
#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
#else
    using T = typename src_tensor_t::value_type;

    static_assert(
        std::is_same<typename src_tensor_t::value_type, typename dst_tensor_t::value_type>::value,
        "Source and destination tensors must have the same type");

    static_assert((algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI),
                  "Unsupported tile Broadcast algorithm. "
                  "Currently NVLS_ONE_SHOT_PUSH_NBI is supported for tile broadcast");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check shape
    assert((get_shape_element<0>(src_tensor) * get_shape_element<1>(src_tensor) *
            nvshmem_team_n_pes(team)) &&
           (get_shape_element<0>(dst_tensor) * get_shape_element<1>(dst_tensor)));

    // TODO add other data types
    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value) ||
                   (is_cutlass_half<T>()) || (is_cutlass_bfloat<T>)),
                  "Unsupported datatype");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    // NVLS Bcast only has one-shot push support currently
    if constexpr (algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI) {
        // check for NVLS support in hardware
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010

        // NVLS ONE_SHOT broadcast is PUSH based algo, so we can directly start communicating
        // User should ensure src data is ready

        nvshmemi_tile_bcast_nvls_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope>(
            team, src_tensor, dst_tensor, start_coord, boundary);
#else
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010 &&
               "Unsupported NVLS on this platform");
#endif
        return 0;
    } else {
        // Extend as other algorithms are added
        return 0;
    }
#endif /* __cplusplus >= 201703L */
}

#endif /* __CUDA_ARCH__ */

#endif
