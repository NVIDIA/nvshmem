/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cuda_runtime.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_constants.h"

#ifndef _NVSHMEMI_TRANSFER_H_
#define _NVSHMEMI_TRANSFER_H_

#ifdef __CUDA_ARCH__

#if defined __clang_llvm_bitcode_lib__
#define NVSHMEMI_TRANSFER_INLINE \
    __attribute__((noinline, section(".text.compute"), not_tail_called))
#define NVSHMEMI_TRANSFER_STATIC
#elif defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_TRANSFER_INLINE inline
#define NVSHMEMI_TRANSFER_STATIC static
#else
#define NVSHMEMI_TRANSFER_INLINE __noinline__
#define NVSHMEMI_TRANSFER_STATIC
#endif

template <typename T>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_rma_p(
    void *rptr, const T value, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <typename T>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ T
nvshmemi_transfer_rma_g(void *rptr, int pe, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_rma(
    void *rptr, void *lptr, size_t bytes, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <threadgroup_t SCOPE>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_addr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi, nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_rma_nbi(
    void *rptr, void *lptr, size_t bytes, int pe,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <typename T>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ T
nvshmemi_transfer_amo_fetch(void *rptr, T value, T compare, int pe, nvshmemi_amo_t op,
                            nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <typename T>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_amo_nonfetch(
    void *rptr, T value, int pe, nvshmemi_amo_t op,
    nvshmemx_qp_handle_t qp_index = NVSHMEMX_QP_DEFAULT);

template <threadgroup_t SCOPE>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_quiet(
    bool use_membar, int pe = NVSHMEMX_PE_ALL, nvshmemx_qp_handle_t *qp_handle = NULL,
    int num_qps = 1);

template <threadgroup_t SCOPE>
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void nvshmemi_transfer_fence(
    int pe = NVSHMEMX_PE_ALL, nvshmemx_qp_handle_t *qp_handle = NULL, int num_qps = 1);
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void
nvshmemi_transfer_enforce_consistency_at_target(bool use_membar);
NVSHMEMI_TRANSFER_STATIC NVSHMEMI_TRANSFER_INLINE __device__ void
nvshmemi_transfer_syncapi_update_mem();
#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_TRANSFER_H_ */
