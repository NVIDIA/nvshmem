/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

static void init_uid() {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

    nvshmemx_get_uniqueid(&id);
    nvshmemx_set_attr_uniqueid_args(0, 1, &id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) return 1;

    init_uid();
    nvshmem_finalize();
    init_uid();

    // Intentionally rely on process-exit cleanup for the final UID bootstrap.
    return 0;
}
