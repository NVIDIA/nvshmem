/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 */

#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define N 2

__device__ size_t dev_strlen(const char *str) {
    size_t i = 0;
    while (str[i] != '\0') {
        ++i;
    }
    return i;
}

__device__ int dev_strncmp(const char *s1, const char *s2, size_t max) {
    for (size_t i = 0; i < max; i++) {
        if (s1[i] != s2[i]) {
            return 1;
        }
        if (s1[i] == '\0') {
            break;
        }
    }

    return 0;
}

__global__ void check_info(void) {
    int major_ver, minor_ver;
    char name[NVSHMEM_MAX_NAME_LEN];

    nvshmem_info_get_version(&major_ver, &minor_ver);
    nvshmem_info_get_name(name);

    assert(dev_strlen(name) <= NVSHMEM_MAX_NAME_LEN);
    assert(major_ver == NVSHMEM_MAJOR_VERSION);
    assert(minor_ver == NVSHMEM_MINOR_VERSION);
    assert(major_ver >= 1);
    assert(minor_ver >= 0);
    assert(dev_strncmp(name, NVSHMEM_VENDOR_STRING, NVSHMEM_MAX_NAME_LEN) == 0);

    printf("%d<%d>: NVSHMEM Device %d.%d -- \"%s\"\n", nvshmem_my_pe(), threadIdx.x, major_ver,
           minor_ver, name);
}

int main(int argc, char **argv) {
    int major_ver, minor_ver, patch_ver;
    char name[NVSHMEM_MAX_NAME_LEN];

    init_wrapper(&argc, &argv);
    nvshmem_barrier_all(); /* Ensure NVSHMEM device init has completed */

    nvshmem_info_get_version(&major_ver, &minor_ver);
    nvshmem_info_get_name(name);

    assert(strlen(name) <= NVSHMEM_MAX_NAME_LEN);
    assert(major_ver == NVSHMEM_MAJOR_VERSION);
    assert(minor_ver == NVSHMEM_MINOR_VERSION);
    assert(major_ver >= 1);
    assert(minor_ver >= 0);
    assert(strncmp(name, NVSHMEM_VENDOR_STRING, NVSHMEM_MAX_NAME_LEN) == 0);

    printf("%d: NVSHMEM Host spec version %d.%d vendor version %d.%d.%d (%d) -- \"%s\"\n",
           nvshmem_my_pe(), major_ver, minor_ver, NVSHMEM_VENDOR_MAJOR_VERSION,
           NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION, NVSHMEM_VENDOR_VERSION,
           name);

    nvshmemx_vendor_get_version_info(&major_ver, &minor_ver, &patch_ver);
    assert(major_ver == NVSHMEM_VENDOR_MAJOR_VERSION);
    assert(minor_ver == NVSHMEM_VENDOR_MINOR_VERSION);
    assert(patch_ver == NVSHMEM_VENDOR_PATCH_VERSION);
    assert(major_ver >= 1);
    assert(minor_ver >= 0);
    assert(patch_ver >= 0);

    printf("%d: NVSHMEM Host Vendor version %d.%d.%d\n", nvshmem_my_pe(), major_ver, minor_ver,
           patch_ver);

    check_info<<<1, N>>>();

    finalize_wrapper();
    return 0;
}
