/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEMI_GDAKI_DEVICE_H_
#define _NVSHMEMI_GDAKI_DEVICE_H_

#include <cuda_runtime.h>
#include <cuda/atomic>
#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_common_gpunetio.h"
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/nvshmem_build_options.h"
#include "utils_device.h"

#ifdef __CUDA_ARCH__
#include "gpunetio/doca_gpunetio_device.h"
/**
 * RMA P base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64) failed");
#endif
template <typename T, bool is_full_warp, bool can_combine_data, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_p_impl(
    void *rptr, const T value, int dst_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma_p_impl not implemented");
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_p(
    void *rptr, const T value, int dst_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma_p not implemented");
}

/**
 * RMA G base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_gdaki_rma_g_impl(
    void *rptr, int dst_pe, int proxy_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma_g_impl not implemented");
    return T();
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_rma_g(void *rptr, int dst_pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma_g not implemented");
    return T();
}

/**
 * RMA NBI base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma_nbi(
    void *rptr, void *lptr, size_t bytes, int dst_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma_nbi not implemented");
}

/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_rma(
    void *rptr, void *lptr, size_t bytes, int dst_pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_rma not implemented");
}

/**
 * AMO non-fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_amo_nonfetch_impl(
    void *rptr, const T value, int pe, nvshmemi_amo_t op,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_amo_nonfetch_impl not implemented");
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_amo_nonfetch(
    void *rptr, const T value, int pe, nvshmemi_amo_t op,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_amo_nonfetch not implemented");
}

/**
 * AMO fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_amo_fetch_impl(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op,
                              nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_amo_fetch_impl not implemented");
    return T();
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T
nvshmemi_gdaki_amo_fetch(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op,
                         nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_amo_fetch not implemented");
    return T();
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 128,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 128) failed");
#endif
template <bool is_nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal_thread_impl(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_put_signal_thread_impl not implemented");
}

/**
 * PUT SIGNAL base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_GPUNETIO_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, bool is_nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal_impl(
    void *req_rptr, void *req_lptr, size_t bytes, void *sig_rptr, uint64_t signal,
    nvshmemi_amo_t sig_op, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_put_signal_impl not implemented");
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT) {
    assert(0 && "nvshmemi_gdaki_put_signal not implemented");
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_qp_quiet(
    bool enforce_cst, int pe_hint, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    assert(0 && "nvshmemi_gdaki_qp_quiet not implemented");
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_fence() {
    assert(0 && "nvshmemi_gdaki_fence not implemented");
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_qp_fence(
    int pe_hint, nvshmemx_qp_handle_t *qp_handle, int num_qps) {
    assert(0 && "nvshmemi_gdaki_qp_fence not implemented");
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_gdaki_enforce_consistency_at_target(
    bool use_membar) {
    assert(0 && "nvshmemi_gdaki_enforce_consistency_at_target not implemented");
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_GDAKI_DEVICE_H_ */
