/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"
#include "utils.h"

#define MAX_SIZE 100 * 1024 * 1024
#define ITER 20
#define REPEAT 50

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    size_t size;
    char **buffer;
    int iter = ITER;
    int repeat = REPEAT;
    char size_string[100];

    size = (size_t)MAX_SIZE * iter * 2;
    sprintf(size_string, "%zu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        goto out;
    }

    srand(1);
    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();

    buffer = (char **)calloc(iter, sizeof(char *));
    if (!buffer) {
        ERROR_PRINT("malloc failed \n");
        goto out;
    }

    for (int r = 0; r < repeat; r++) {
        uint32_t lsize = r;
        if (!mype) DEBUG_PRINT("[iter %d of %d] allocations: ", r, repeat);
        for (int i = 0; i < iter; i++) {
            lsize = rand_r(&lsize) % (MAX_SIZE - 1) + 1;

            buffer[i] = (char *)nvshmem_malloc(lsize);
            if (!buffer[i]) {
                ERROR_PRINT("shmem_malloc failed \n");
                goto out;
            }

            cudaMemset(buffer[i], 0, lsize);
            if (!mype) DEBUG_PRINT("ptr: %p size: %zuB; ", (void *)buffer[i], lsize);
        }
        if (!mype) DEBUG_PRINT("\n \n");

        if (!mype) DEBUG_PRINT("[%d of %d] freeing all buffers: ", r, repeat);

        for (int i = 0; i < iter; i++) {
            if (!mype) DEBUG_PRINT("ptr: %p; ", (void *)buffer[i]);
            nvshmem_free(buffer[i]);
        }
        if (!mype) DEBUG_PRINT("\n \n");

        if (!mype) DEBUG_PRINT("[iter %d of %d] end of iter \n \n", r, repeat);
    }

    free(buffer);

    finalize_wrapper();

out:
    return status;
}
