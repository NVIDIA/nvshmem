/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>                                 // for std::any_of, std::copy_n
#include <array>                                     // for std::array
#include <cuda.h>                                    // for CUDA_SUCCESS
#include <cuda_runtime.h>                            // for cudaGetErrorString
#include <driver_types.h>                            // for cudaError_t, cud...
#include <limits.h>                                  // for INT_MAX, INT_MIN
#include <stdio.h>                                   // for fprintf, stderr
#include <vector_types.h>                            // for dim3
#include "device/nvshmemx_collective_launch_apis.h"  // for nvshmemx_collect...
#include "internal/device/nvshmemi_device.h"         // for nvshmemi_device_...
#include "non_abi/nvshmemx_error.h"                  // for NVSHMEMI_NE_ERRO...

#define CUDA_RUNTIME_CHECK_GOTO(stmt, res, label)                                 \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            res = result;                                                         \
            goto label;                                                           \
        }                                                                         \
    } while (0)

int nvshmemi_check_state_and_init_d();

inline int nvshmemi_minv(int *vec, int count) {
    int minval = INT_MAX;
    for (int i = 0; i < count; i++) {
        if (vec[i] < minval) {
            minval = vec[i];
        }
    }
    return minval;
}

inline int nvshmemi_maxv(int *vec, int count) {
    int maxval = INT_MIN;
    for (int i = 0; i < count; i++) {
        if (vec[i] > maxval) {
            maxval = vec[i];
        }
    }
    return maxval;
}

static int _nvshmemi_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                                      size_t sharedMem, int *gridsize) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int status = 0;

    int ret = nvshmemi_check_state_and_init_d();
    if (ret) {
        fprintf(stderr, "nvshmemi_check_state_and_init_d() failed");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    multiProcessorCount = nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count;
    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    // XXX: Returns maximum supported grid (including 0) on associated GPU
    *gridsize = maxBlocksSM * multiProcessorCount;  // XXX:caller chooses dimension of grid

out:
    return status;
}

/*
 * Merged attribute buffer is stack-allocated per call. 16 leaves headroom over
 * the size of the cudaLaunchAttributeID enum. Bump if CUDA grows it. The bound
 * is enforced at runtime so an oversized input fails cleanly instead of
 * corrupting the stack.
 */
static constexpr int NVSHMEMI_MAX_MERGED_LAUNCH_ATTRS = 16;

static bool nvshmemi_attr_present(const cudaLaunchAttribute *attrs, int n,
                                  cudaLaunchAttributeID id) {
    return std::any_of(attrs, attrs + n, [id](const cudaLaunchAttribute &a) { return a.id == id; });
}

static int _nvshmemi_collective_launch_attr(const nvshmemx_collective_launch_attr_t *attr,
                                            const void *func, void **args) {
    int multiProcessorCount;
    int maxBlocksSM;
    int gridSize = -1;
    int launchFailed = 1;
    int status = 0;
    cudaLaunchConfig_t cfg{};
    std::array<cudaLaunchAttribute, NVSHMEMI_MAX_MERGED_LAUNCH_ATTRS> merged_attrs{};
    int n_merged = 0;
    cudaStream_t user_stream;

    if (attr == nullptr) {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }

    cfg = attr->cuda_config;
    user_stream = cfg.stream;

    {
        int blockSize = cfg.blockDim.x * cfg.blockDim.y * cfg.blockDim.z;

        int ret = nvshmemi_check_state_and_init_d();
        if (ret) {
            fprintf(stderr, "nvshmemi_check_state_and_init_d() failed");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
        // XXX: Supports the user passing a non-zero grid but of differing size across ranks
        if (cfg.gridDim.x == 0 && cfg.gridDim.y == 0 && cfg.gridDim.z == 0) {
            gridSize = 0;
        } else if (cfg.gridDim.x != 0 && cfg.gridDim.y != 0 && cfg.gridDim.z != 0) {
            gridSize = cfg.gridDim.x * cfg.gridDim.y * cfg.gridDim.z;
        }  // else
           // some but not all grid dim being 0 is illegal
           // XXX: if some ranks pass an illegal grid, others error out

        // get min blocks per SM, error out if 0 for any GPU
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize,
                                                               cfg.dynamicSmemBytes);
        NVSHMEMI_NE_ERROR_JMP(
            status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed aborting job\n");

        multiProcessorCount = nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count;
        if (gridSize == 0) { /*XXX : auto sizing */
            // XXX: Launches maximum supported grid (>0) on associated GPU
            if (maxBlocksSM > 0) { /*Launch will work only if all GPUs can run at least one CTA*/
                launchFailed = 0;
            }
            cfg.gridDim.x = maxBlocksSM * multiProcessorCount;
            cfg.gridDim.y = 1;
            cfg.gridDim.z = 1;
        } else if (gridSize > 0) { /* XXX : legal grid is provided by user*/
            if ((maxBlocksSM > 0) && (gridSize <= maxBlocksSM * multiProcessorCount)) { /*Works*/
                launchFailed = 0;
            }
        }
    }

    /* TODO: make it obvious we aren't going to complete this call from this thread. Possibly global
     * exit? */
    NVSHMEMI_CHECK_ERROR_JMP(launchFailed, status, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                             "One or more PEs cannot launch \n");

    /* Merge user-provided attributes with NVSHMEM's required attributes.
     * User wins on duplicates: if the caller already specified an attribute
     * NVSHMEM would otherwise add, NVSHMEM does not overwrite it. */
    if (cfg.numAttrs > NVSHMEMI_MAX_MERGED_LAUNCH_ATTRS) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "user supplied %d launch attributes; max supported is %d\n",
                           cfg.numAttrs, NVSHMEMI_MAX_MERGED_LAUNCH_ATTRS);
    }
    std::copy_n(cfg.attrs, cfg.numAttrs, merged_attrs.begin());
    n_merged = cfg.numAttrs;
    if (nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch &&
        !nvshmemi_attr_present(merged_attrs.data(), n_merged, cudaLaunchAttributeCooperative)) {
        if (n_merged >= NVSHMEMI_MAX_MERGED_LAUNCH_ATTRS) {
            NVSHMEMI_ERROR_JMP(
                status, NVSHMEMX_ERROR_INTERNAL, out,
                "no room to add cudaLaunchAttributeCooperative to merged attribute list\n");
        }
        merged_attrs[n_merged].id = cudaLaunchAttributeCooperative;
        merged_attrs[n_merged].val.cooperative = 1;
        ++n_merged;
    }

    /* NVSHMEM owns the stream the kernel actually launches on. The user's
     * stream is used as the fence point: an end-event recorded on the internal
     * stream is waited on by the user's stream after the launch. */
    cfg.stream = nvshmemi_device_only_state.claunch_params.stream;
    cfg.attrs = merged_attrs.data();
    cfg.numAttrs = n_merged;

    CUDA_RUNTIME_CHECK_GOTO(
        cudaEventRecord(nvshmemi_device_only_state.claunch_params.begin_event, user_stream), status,
        out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamWaitEvent(nvshmemi_device_only_state.claunch_params.stream,
                            nvshmemi_device_only_state.claunch_params.begin_event, 0),
        status, out);

    status = cudaLaunchKernelExC(&cfg, func, args);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                          "cudaLaunchKernelExC failed\n");

    CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(nvshmemi_device_only_state.claunch_params.end_event,
                                            nvshmemi_device_only_state.claunch_params.stream),
                            status, out);

    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamWaitEvent(user_stream, nvshmemi_device_only_state.claunch_params.end_event, 0),
        status, out);

out:
    return status;
}

int nvshmemi_setup_collective_launch() {
    int leastPriority, greatestPriority, status = 0;
    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&(nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count),
                               cudaDevAttrMultiProcessorCount,
                               nvshmemi_device_only_state.cuda_device_id),
        status, out);

    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&(nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch),
                               cudaDevAttrCooperativeLaunch,
                               nvshmemi_device_only_state.cuda_device_id),
        status, out);

    if (!nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch) {
        NVSHMEMI_WARN_PRINT(
            "Cooperative launch not supported on at least one PE; GPU-side synchronize may cause "
            "hang\n");
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamCreateWithPriority(&nvshmemi_device_only_state.claunch_params.stream,
                                     cudaStreamNonBlocking, greatestPriority),
        status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventCreate(&nvshmemi_device_only_state.claunch_params.begin_event,
                                            cudaEventDisableTiming),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventCreate(&nvshmemi_device_only_state.claunch_params.end_event,
                                            cudaEventDisableTiming),
                            status, out);

out:
    return status;
}

int nvshmemi_teardown_collective_launch() {
    int status = 0;

    if (!nvshmemi_device_only_state.is_initialized) {
        goto out;
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamDestroy(nvshmemi_device_only_state.claunch_params.stream),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(nvshmemi_device_only_state.claunch_params.begin_event),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(nvshmemi_device_only_state.claunch_params.end_event),
                            status, out);

out:
    return status;
}

extern "C" {

int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize) {
    return _nvshmemi_collective_launch_query_gridsize(func, blockDims, args, sharedMem, gridsize);
}

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream) {
    nvshmemx_collective_launch_attr_t a{};
    a.cuda_config.gridDim = gridDims;
    a.cuda_config.blockDim = blockDims;
    a.cuda_config.dynamicSmemBytes = sharedMem;
    a.cuda_config.stream = stream;
    a.cuda_config.attrs = nullptr;
    a.cuda_config.numAttrs = 0;
    return _nvshmemi_collective_launch_attr(&a, func, args);
}

int nvshmemx_collective_launch_attr(const nvshmemx_collective_launch_attr_t *attr, const void *func,
                                    void **args) {
    return _nvshmemi_collective_launch_attr(attr, func, args);
}

}  // extern "C"
