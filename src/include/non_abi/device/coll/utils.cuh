/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_DEVICE_COLL_UTILS_H_
#define _NVSHMEMI_DEVICE_COLL_UTILS_H_

#include <cuda_runtime.h>
#if !defined __CUDACC_RTC__
#include <type_traits>
#else
#include <cuda/std/type_traits>
#endif
#include "non_abi/nvshmem_build_options.h"
#include "device/nvshmem_device_macros.h"
// This is added so the entrypoint (init_device.cu) can receive the implementations of NVSHMEM
// transfer APIs. transfer_device.cuh internally short-circuits to an empty stub header in
// P2P-only builds.
#if defined(NVSHMEM_ENABLE_ALL_DEVICE_INLINING) || defined(__NVSHMEM_NUMBA_SUPPORT__) || \
    defined(NVSHMEM_BUILD_LTOIR_LIBRARY) || defined(NVSHMEM_BUILD_P2P_ONLY)
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/common/nvshmemi_path_predicates.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "device/logical_endpoint_device.cuh"

#ifdef __CUDA_ARCH__

/* This is signaling function used in barrier algorithm.
nvshmem_<type>_signal function cannot be used in barrier because it uses a
combination of P2P path and IB path depending on how the peer GPU is
connected. In contrast to that, this fuction uses either P2P path (when all GPUs
are NVLink connected) or IB path (when any of the GPU is not NVLink connected).

Using this function in barrier is necessary to ensure any previous RMA
operations are visible. When combination of P2P and IB path are used
as in nvshmem_<type>_signal function, it can lead to race conditions.
For example NVLink writes (of data and signal) can overtake IB writes.
And hence the data may not be visible after the barrier operation.
*/
template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_signal_for_barrier(T *dest, const T value,
                                                                          int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (nvshmemi_use_ldst_path()) {
#if LE_HW_SW_REQUIREMENTS_MET && defined(NVSHMEM_CFT_HANDLES_SUPPORT)
        if (nvshmemi_peer_reachable(peer_base_addr) &&
            !nvshmemi_is_le_supported_and_prioritized(pe)) {
            volatile T *dest_actual =
                (volatile T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
            *dest_actual = value;
        } else if (nvshmemi_is_le_implemented(pe)) {
            // It is more performant to use pointers for loopback to own memory
            if (pe == nvshmemi_device_state_d.mype) {
                *dest = value;
            } else {
                nvshmemi_handle_p<T>((void *)dest, value, pe);
            }

        } else {
            // If handles are not setup yet (e.g., tma_smem not registered during init)
            // fallback to IB
            nvshmemi_transfer_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
        }
#else
        volatile T *dest_actual =
            (volatile T *)((char *)(peer_base_addr) +
                           ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
#endif
    } else {
        nvshmemi_transfer_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
    }
}

__host__ __device__ constexpr bool nvshmemi_check_pow2(const int n) {
    return ((n > 0) && !(n & (n - 1)));
}

#endif /* __CUDA_ARCH__ */
#endif /* NVSHMEMI_DEVICE_COLL_UTILS_H */
