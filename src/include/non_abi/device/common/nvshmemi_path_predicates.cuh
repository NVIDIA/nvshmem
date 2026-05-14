/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_PATH_PREDICATES_CUH_
#define _NVSHMEMI_PATH_PREDICATES_CUH_

#include <cuda_runtime.h>
#include "non_abi/nvshmem_build_options.h"
#include "device/nvshmem_device_macros.h"
#include "device_host/nvshmem_common.cuh"

#ifdef __CUDA_ARCH__

/*
 * Centralized predicates for the dispatch between local (P2P NVLink) and
 * remote-transport code paths. In the case of P2P-only builds, all predicates
 * fold to a literal compile-time constant.
 */

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_use_ldst_atomics_path() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return true;
#else
    return nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS;
#endif
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_use_ldst_path() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return true;
#else
    return nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST;
#endif
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_use_ldst_remote_atomics_path() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return true;
#else
    return nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS;
#endif
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_needs_proxy_cst() {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    return false;
#else
    return nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY;
#endif
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE bool nvshmemi_peer_reachable(const void *peer_base_addr) {
#ifdef NVSHMEM_BUILD_P2P_ONLY
    (void)peer_base_addr;
    return true;
#else
    return peer_base_addr != nullptr;
#endif
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_PATH_PREDICATES_CUH_ */
