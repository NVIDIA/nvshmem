/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
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
#if defined(__clang_llvm_bitcode_lib__)
#if defined(__CUDACC__)
// Clang CUDA mode: use __constant__ only (no address_space to avoid LLVM21 conflict)
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only (no __constant__)
__attribute__((address_space(4),
               used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#endif
#else
// Normal CUDA/nvcc build
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#endif
#endif

nvshmemi_device_state_t nvshmemi_device_only_state;

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

const nvshmemi_version_t nvshmemi_device_lib_version = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};

extern nvshmemi_selected_device_transport_t nvshmem_selected_device_transport;

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

#ifdef __cplusplus
extern "C" {
int nvshmemi_get_device_state_ptrs(void **dev_state_ptr, void **transport_dev_state_ptr);
}
#endif

static int _nvshmemi_init_device_only_state() {
    int status = 0;
    status = nvshmemi_setup_collective_launch();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "_nvshmemi_init_device_only_state failed\n");
    nvshmemi_device_only_state.is_initialized = true;

out:
    return status;
}

int nvshmemi_check_state_and_init_d() {
    int status = NVSHMEMI_SUCCESS;
    int ret;

    if (nvshmemid_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
        fprintf(stderr, "nvshmem API called before nvshmem_init \n");
        return NVSHMEMI_NOT_INITIALIZED;
    }
    if (nvshmemid_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        /* The fact that we can pass NVSHMEM_THREAD_SERIALIZED
         * here is an implementation detail. It should be fixed
         * if/when NVSHMEM_THREAD_* becomes significant. */
        status = nvshmemid_hostlib_init_attr(NVSHMEM_THREAD_SERIALIZED, &ret, 0, NULL,
                                             nvshmemi_device_lib_version, NULL);
        if (status) {
            fprintf(stderr, "nvshmem initialization failed, exiting \n");
            return NVSHMEMI_NOT_INITIALIZED;
        }

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            fprintf(stderr, "nvshmem cuda device query failed, exiting \n");
            return NVSHMEMI_CUDA_GET_DEVICE_FAILED;
        }

        nvshmemid_hostlib_finalize(NULL, NULL);  // for refcounting
    }

    if (!nvshmemi_device_only_state.is_initialized) {
        status = _nvshmemi_init_device_only_state();
        if (status) {
            fprintf(stderr, "nvshmem device initialization failed, exiting \n");
            return NVSHMEMI_INIT_DEVICE_ONLY_STATE_FAILED;
        }
    }

    return status;
}

int nvshmemi_get_device_state_ptrs(void **dev_state_ptr, void **transport_dev_state_ptr) {
    int status = NVSHMEMI_SUCCESS;
    status = cudaGetSymbolAddress(dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access device state. %d\n", status);
        *dev_state_ptr = NULL;
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    // Get transport device pointer (if required)
    switch (nvshmem_selected_device_transport) {
#ifdef NVSHMEM_IBGDA_SUPPORT
        case NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA:
            status = cudaGetSymbolAddress(transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
            if (status) {
                NVSHMEMI_ERROR_PRINT("Unable to access ibgda device state. %d\n", status);
                *transport_dev_state_ptr = NULL;
                status = NVSHMEMX_ERROR_INTERNAL;
            }
            break;
#endif
        default:
            *transport_dev_state_ptr = NULL;
            break;
    }

out:
    return status;
}

int nvshmemi_init_thread(int requested_thread_support, int *provided_thread_support,
                         unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                         nvshmemi_version_t nvshmem_app_version) {
    int status = 0;

#ifdef _NVSHMEM_DEBUG
    printf("  %-28s %d\n", "DEVICE CUDA API", CUDART_VERSION);
#endif
    status = nvshmemid_hostlib_init_attr(
        requested_thread_support, provided_thread_support, bootstrap_flags, bootstrap_attr,
        nvshmemi_device_lib_version, &nvshmemi_get_device_state_ptrs);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem_internal_init_thread failed \n");

    if (nvshmemid_init_status() > NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        status = _nvshmemi_init_device_only_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem_internal_init_thread failed at init_device_only_state.\n");

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem cuda device query failed, exiting \n");
        }
    }

out:
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
void nvshmemi_finalize() {
    int status;
    void *dev_state_ptr, *transport_dev_state_ptr = NULL;

    status = cudaGetSymbolAddress(&dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemid_hostlib_finalize(NULL, NULL);
        return;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmem_selected_device_transport == NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA) {
        status = cudaGetSymbolAddress(&transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
            nvshmemid_hostlib_finalize(NULL, NULL);
            return;
        }
    }
#endif
    nvshmemid_hostlib_finalize(dev_state_ptr, transport_dev_state_ptr);
}
#ifdef __cplusplus
}
#endif
#endif