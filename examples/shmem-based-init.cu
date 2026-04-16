/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "shmem.h"
#include "nvshmem.h"

int main(int c, char *v[]) {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

    shmem_init();
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

    nvshmem_finalize();
    shmem_finalize();
    return 0;
}
