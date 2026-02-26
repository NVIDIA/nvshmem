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
#define MIN_SIZE 512
#define MIN_ITER 20
#define MAX_ITER 500
#define REPEAT 20

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    size_t size;
    char **buffer;
    int min_iter = MIN_ITER;
    int max_iter = MAX_ITER;
    int repeat = REPEAT;
    char size_string[100];

    size = (size_t)MAX_SIZE * min_iter * 2;
    sprintf(size_string, "%zu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        goto out;
    }

    srand(1);
    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();

    for (int r = 0; r < repeat; r++) {
        uint32_t lsize = r;
        if (!mype) DEBUG_PRINT("[iter %d of %d] begin \n", r, repeat);
        for (int b = MIN_SIZE; b <= MAX_SIZE; b = b << 1) {
            int iter = min_iter * (MAX_SIZE / b);
            iter = (iter > max_iter) ? max_iter : iter;
            buffer = (char **)calloc(iter, sizeof(char *));
            if (!buffer) {
                ERROR_PRINT("malloc failed \n");
                goto out;
            }

            if (!mype) DEBUG_PRINT("[binsize %d] allocating %d buffers . . . ", b, iter);

            for (int i = 0; i < iter; i++) {
                lsize = rand_r(&lsize) % (b - (b >> 1)) + (b >> 1);

                buffer[i] = (char *)nvshmem_malloc(lsize);
                if (!buffer[i]) {
                    ERROR_PRINT("shmem_malloc failed \n");
                    goto out;
                }

                cudaMemset(buffer[i], 0, lsize);
            }

            if (!mype) DEBUG_PRINT("freeing . . . ");

            // free even buffers
            for (int i = 0; i < iter; i += 2) {
                nvshmem_free(buffer[i]);
            }

            // free odd buffers
            for (int i = 1; i < iter; i += 2) {
                nvshmem_free(buffer[i]);
            }

            if (!mype) DEBUG_PRINT("done \n");

            free(buffer);
        }
        if (!mype) DEBUG_PRINT("[iter %d of %d] end of iter \n \n", r, repeat);
    }

    finalize_wrapper();

out:
    return status;
}
