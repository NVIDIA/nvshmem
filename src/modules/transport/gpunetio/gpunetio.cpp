/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "transport_common.h"
#include "gpunetio/host/doca_gpunetio.h"
#include "gpunetio/host/doca_verbs.h"
#include "gpunetio/host/doca_gpunetio_high_level.h"
#include "gpunetio/common/doca_gpunetio_verbs_dev.h"

#include <cassert>
#include <string>
#include <algorithm>

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    NVSHMEMI_WARN_PRINT("GPUNetIO transport not implemented");

    // Call a dummy function to make sure the library is linked
    int status = doca_gpu_create("0000:00:00.0", nullptr);
    if (status != DOCA_SUCCESS) {
        NVSHMEMI_WARN_PRINT("Failed to create GPUNetIO device: %d", status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}
