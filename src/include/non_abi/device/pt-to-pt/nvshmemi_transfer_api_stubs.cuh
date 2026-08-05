/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Empty inline stubs for every nvshmemi_transfer_* symbol declared in
 * nvshmemi_transfer_api.cuh. Used only when NVSHMEM_BUILD_P2P_ONLY is
 * defined, where every call site dispatches via a P2P-only predicate
 * (see non_abi/device/common/nvshmemi_path_predicates.cuh) that folds to a
 * compile-time constant.
 */

#include <cuda_runtime.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device/nvshmem_device_macros.h"
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/nvshmemi_region_types.h"

#ifndef _NVSHMEMI_TRANSFER_H_
#define _NVSHMEMI_TRANSFER_H_

#ifdef __CUDA_ARCH__

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_rma_p(
    void* /*rptr*/, const T /*value*/, int /*pe*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_transfer_rma_g(
    void* /*rptr*/, int /*pe*/, nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {
    return T{};
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_rma(
    void* /*rptr*/, void* /*lptr*/, size_t /*bytes*/, int /*pe*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_put_signal(
    void* /*rptr*/, void* /*lptr*/, size_t /*bytes*/, void* /*sig_addr*/, uint64_t /*signal*/,
    nvshmemi_amo_t /*sig_op*/, int /*pe*/, bool /*is_nbi*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_rma_nbi(
    void* /*rptr*/, void* /*lptr*/, size_t /*bytes*/, int /*pe*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT,
    const nvshmemi_region_info_t* /*region_info*/ = NULL) {}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_region_end(
    const nvshmemi_region_info_t* /*region_info*/) {}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_transfer_amo_fetch(
    void* /*rptr*/, T /*value*/, T /*compare*/, int /*pe*/, nvshmemi_amo_t /*op*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {
    return T{};
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_amo_nonfetch(
    void* /*rptr*/, T /*value*/, int /*pe*/, nvshmemi_amo_t /*op*/,
    nvshmemx_qp_handle_t /*qp_index*/ = NVSHMEMX_QP_DEFAULT) {}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_quiet(
    bool /*use_membar*/, int /*pe*/ = NVSHMEMX_PE_ALL, nvshmemx_qp_handle_t* /*qp_handle*/ = NULL,
    int /*num_qps*/ = NVSHMEMX_QP_ALL) {}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_fence(
    int /*pe*/ = NVSHMEMX_PE_ALL, nvshmemx_qp_handle_t* /*qp_handle*/ = NULL,
    int /*num_qps*/ = NVSHMEMX_QP_ALL) {}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_enforce_consistency_at_target(
    bool /*use_membar*/) {}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_transfer_syncapi_update_mem() {
    /* In P2P-only builds the wait-until callers replaced the explicit
     * `nvshmemi_transfer_syncapi_update_mem()` stub with `__threadfence()`
     */
    __threadfence();
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_TRANSFER_H_ */
