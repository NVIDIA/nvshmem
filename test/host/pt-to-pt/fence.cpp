/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define NUM_ITERS 100

int main(int argc, char *argv[]) {
    int status = 0;
    int num_iters = NUM_ITERS;

    init_wrapper(&argc, &argv);

    while (1) {
        int c;
        c = getopt(argc, argv, "n:h");
        if (c == -1) break;

        switch (c) {
            case 'n':
                num_iters = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-n [No of iterations] \n");
                goto out;
        }
    }

    nvshmem_barrier_all();
    for (int i = 0; i < num_iters; i++) {
        nvshmem_fence();
    }

out:
    nvshmem_barrier_all();
    finalize_wrapper();

    return status;
}
