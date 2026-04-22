/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NVSHMEM_INIT_DEVICE_CUH_
#define _NVSHMEM_INIT_DEVICE_CUH_

#include <stdio.h>
#include <algorithm>
#include <cuda_runtime.h>

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
#if !defined(assert)
#define assert(...)
#endif
#include "nvshmem.h"
#endif

#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmem_version.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/device/nvshmemi_device.h"
#include "internal/common/error_codes_internal.h"
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_types.h"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "device_host_transport/nvshmem_common_ibgda.h"
#endif

#ifdef NVSHMEM_GPUNETIO_SUPPORT
#include "device_host_transport/nvshmem_common_gpunetio.h"
#endif

#if defined(__clang_llvm_bitcode_lib__)
#if defined(__CUDACC__)
// Clang CUDA mode: use __constant__ only (no address_space to avoid LLVM21 conflict)
#ifdef NVSHMEM_IBGDA_SUPPORT
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#endif  // NVSHMEM_IBGDA_SUPPORT
#ifdef NVSHMEM_GPUNETIO_SUPPORT
__constant__
    __attribute__((used)) nvshmemi_gpunetio_device_state_t nvshmemi_gpunetio_device_state_d = {};
#endif  // NVSHMEM_GPUNETIO_SUPPORT
#else   // __CUDACC__
// Plain Clang-to-NVPTX bitcode: use address_space(4) only (no __constant__)
#ifdef NVSHMEM_IBGDA_SUPPORT
__attribute__((address_space(4),
               used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#endif  // NVSHMEM_IBGDA_SUPPORT
#ifdef NVSHMEM_GPUNETIO_SUPPORT
__attribute__((address_space(4),
               used)) nvshmemi_gpunetio_device_state_t nvshmemi_gpunetio_device_state_d = {};
#endif  // NVSHMEM_GPUNETIO_SUPPORT
#endif  // __CUDACC__
#else   // __clang_llvm_bitcode_lib__
// Normal CUDA/nvcc build
#ifdef NVSHMEM_IBGDA_SUPPORT
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#endif  // NVSHMEM_IBGDA_SUPPORT
#ifdef NVSHMEM_GPUNETIO_SUPPORT
__constant__
    __attribute__((used)) nvshmemi_gpunetio_device_state_t nvshmemi_gpunetio_device_state_d;
#endif  // NVSHMEM_GPUNETIO_SUPPORT
#endif  // __clang_llvm_bitcode_lib__

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
#if defined(__CUDACC__)
// Clang CUDA mode: use __constant__ only (no address_space to avoid LLVM21 conflict)
__constant__ __attribute__((used)) nvshmemi_device_host_state_t nvshmemi_device_state_d = {};

__constant__ __attribute__((used)) nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only (no __constant__)
__attribute__((address_space(4), used)) nvshmemi_device_host_state_t nvshmemi_device_state_d = {};

__attribute__((address_space(4), used)) nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#endif
#else
// Normal CUDA/nvcc build
__constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;

__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#endif

#ifdef __CUDA_ARCH__
#ifdef __cplusplus
extern "C" {
#endif
NVSHMEMI_DEVICE_PREFIX void nvshmem_global_exit(int status);
#ifdef __cplusplus
}
#endif

NVSHMEMI_DEVICE_PREFIX void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf(
            "Device side proxy was called, but is not supported under your configuration. "
            "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

#endif