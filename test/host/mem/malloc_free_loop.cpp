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

#define MAX_SIZE 32 * 1024 * 1024
#define MIN_SIZE 512
#define MIN_ITER 20
#define MAX_ITER 50
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

    for (int b = MIN_SIZE; b <= MAX_SIZE; b = b << 1) {
        int iter, free_start_idx, free_end_idx;
        uint32_t lsize = b;

        iter = min_iter * (MAX_SIZE / b);
        iter = (iter > max_iter) ? max_iter : iter;
        buffer = (char **)calloc(iter, sizeof(char *));
        if (!buffer) {
            ERROR_PRINT("malloc failed \n");
            goto out;
        }

        if (!mype) DEBUG_PRINT("[binsize %d] allocating %d buffers ", b, iter);
        for (int i = 0; i < iter; i++) {
            lsize = rand_r(&lsize) % (b - (b >> 1)) + (b >> 1);

            buffer[i] = (char *)nvshmem_malloc(lsize);
            if (!buffer[i]) {
                ERROR_PRINT("shmem_malloc failed \n");
                goto out;
            }
            cudaMemset(buffer[i], 0, lsize);
        }

        for (int r = 0; r < repeat; r++) {
            // free half of the buffers
            free_start_idx = (iter / 2) * (repeat & 1);
            free_end_idx = free_start_idx + iter / 2 - 1;
            if (!mype)
                DEBUG_PRINT("freeing buffers with index %d to %d \n", free_start_idx, free_end_idx);
            for (int i = free_start_idx; i <= free_end_idx; i++) {
                nvshmem_free(buffer[i]);
            }

            if (!mype)
                DEBUG_PRINT("re-allocating buffers with index %d to %d \n", free_start_idx,
                            free_end_idx);
            for (int i = free_start_idx; i <= free_end_idx; i++) {
                lsize = rand_r(&lsize) % (b - (b >> 1)) + (b >> 1);
                buffer[i] = (char *)nvshmem_malloc(lsize);
                if (!buffer[i]) {
                    ERROR_PRINT("shmem_malloc failed \n");
                    goto out;
                }
                cudaMemset(buffer[i], 0, lsize);
            }

            if (!mype) DEBUG_PRINT("done repetition %d \n", r);
        }
        // free all buffers
        for (int i = 0; i < iter; i++) {
            nvshmem_free(buffer[i]);
        }

        free(buffer);
        if (!mype) DEBUG_PRINT("[binsize %d] done testing \n", b);
    }

    finalize_wrapper();

out:
    return status;
}
