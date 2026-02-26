/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"
#include "utils.h"

#define MAX_SIZE 100 * 1024 * 1024
#define MIN_SIZE 32
#define MIN_ITER 20
#define MAX_ITER 50
#define REPEAT 10
#define MIN_ALIGNMENT 128
#define MAX_ALIGNMENT 1024

bool verify_nvshmem_align(const char *ptr, int align) {
    bool res;
    if (ptr == nullptr) {
        res = false;
    } else {
        auto i = reinterpret_cast<std::uintptr_t>(ptr);

        if (i % align == 0) {
            res = true;
        } else {
            res = false;
        }
    }

    return res;
}

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
        status = -1;
        goto out;
    }

    srand(1);
    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();

    for (int r = 0; r < repeat; r++) {
        uint32_t lsize = r;
        if (!mype) DEBUG_PRINT("[iter %d of %d]  begin\n", r, repeat);
        for (int b = MIN_SIZE; b <= MAX_SIZE; b = b << 1) {
            int iter = min_iter * (MAX_SIZE / b);
            iter = (iter > max_iter) ? max_iter : iter;

            buffer = (char **)calloc(iter, sizeof(char *));
            if (!buffer) {
                ERROR_PRINT("malloc failed \n");
                status = -1;
                goto out;
            }

            if (!mype) DEBUG_PRINT("[binsize %d] ", b);
            std::vector<std::string> combo_list;
            for (int align = MIN_ALIGNMENT; align <= MAX_ALIGNMENT; align *= 2) {
                if (!mype) DEBUG_PRINT("[align %d] allocating %d buffers . . .", align, iter);

                for (int i = 0; i < iter; i++) {
                    lsize = rand_r(&lsize) % (b - (b >> 1)) + (b >> 1);

                    buffer[i] = (char *)nvshmem_align(align, lsize);
                    if (!buffer[i]) {
                        ERROR_PRINT("shmem_malloc failed \n");
                        status = -1;
                        goto out;
                    } else {
                        bool res;
                        res = verify_nvshmem_align(buffer[i], align);
                        if (!res) {
                            std::string tmp = std::to_string(align) + "-" + std::to_string(lsize);
                            combo_list.push_back(tmp);
                        }
                    }

                    cudaMemset(buffer[i], 0, lsize);
                }

                if (!mype) DEBUG_PRINT("freeing buffers . . . ");

                for (int i = 0; i < iter; i++) {
                    nvshmem_free(buffer[i]);
                }

                if (!mype) DEBUG_PRINT("done \n");
            }

            free(buffer);

            if (!combo_list.empty()) {
                ERROR_PRINT("caught error: align-size \n");
                for (auto i : combo_list) ERROR_PRINT("%s\n", i.c_str());
                ERROR_PRINT("verify_nvshmem_align failed \n");
                status = -1;
                goto out;
            }
        }
        if (!mype) DEBUG_PRINT("[iter %d of %d] end \n \n", r, repeat);
    }

out:
    finalize_wrapper();
    return status;
}
