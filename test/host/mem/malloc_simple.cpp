/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <stdlib.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"
#include "utils.h"

#define MAX_SIZE 128 * 1024 * 1024

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    size_t size;
    char *buffer = NULL;
    char size_string[100];

    size = (size_t)MAX_SIZE * 2;
    sprintf(size_string, "%zu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    npes = nvshmem_n_pes();
#endif

    for (size = 1; size <= MAX_SIZE; size *= 2) {
        buffer = (char *)nvshmem_malloc(size);
        if (!buffer) {
            ERROR_PRINT("shmem_malloc failed \n");
            status = -1;
            goto out;
        }

        cudaMemset(buffer, 0, size);

        if (!mype)
            DEBUG_PRINT("[%d of %d] allocated symmetric object: %p size: %zu bytes \n", mype, npes,
                        buffer, size);

        nvshmem_free(buffer);

        if (!mype) DEBUG_PRINT("[%d of %d] free symmetric object: %p \n", mype, npes, buffer);
    }

out:
    finalize_wrapper();
    return status;
}
