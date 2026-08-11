/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <cuda.h>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

__global__ void func(int *input, int *output) {
    int bid = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
    int tid = (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int bsize = blockDim.z * blockDim.y * blockDim.x;
    int idx = bid * bsize + tid;
    /*if(tid==0){
       printf("BEFORE : bid %d input %d output %d\n", bid, input[idx], output[idx]);
    }*/
    output[idx] = input[idx];
    /*if(tid==0){
       printf("AFTER : bid %d input %d output %d\n", bid, input[idx], output[idx]);
    }*/
}

int main(int argc, char *argv[]) {
    int status = 0;
    int mype, npes;
    void *din, *dout;
    int *hin, *hout;
    void *args[2];

    int gridsize;
    int blocksize = 256;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    while (1) {
        int c;
        c = getopt(argc, argv, "t:h");
        if (c == -1) {
            break;
        }

        switch (c) {
            case 't':
                blocksize = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-t [THREADS-PER-CTA] \n");
                goto out;
        }
    }

    nvshmem_barrier_all();

    args[0] = &din;
    args[1] = &dout;
    if (NVSHMEMX_SUCCESS != nvshmemx_collective_launch_query_gridsize((const void *)func, blocksize,
                                                                      args, 0, &gridsize)) {
        ERROR_PRINT("SHMEM: [%d of %d] not supported\n", mype, npes);
        status = -1;
        goto out;
    }

    if (mype == 0) {
        DEBUG_PRINT("Grid size (auto) : %d Block size : %d\n", gridsize, blocksize);
    }

    CUDA_CHECK(cudaMalloc(&din, gridsize * blocksize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dout, gridsize * blocksize * sizeof(int)));

    hout = new int[gridsize * blocksize];
    hin = new int[gridsize * blocksize];
    for (int idx = 0; idx < gridsize * blocksize; idx++) {
        hin[idx] = (int)0xDEADBEEF;
        hout[idx] = (int)0xCAFEBABE;
    }
    DEBUG_PRINT("SHMEM: [%d of %d] hin[0] : 0x%x, hout[0] : 0x%x before read from GPU\n", mype,
                npes, hin[0], hout[0]);

    CUDA_CHECK(cudaMemcpy(din, hin, gridsize * blocksize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dout, hout, gridsize * blocksize * sizeof(int), cudaMemcpyHostToDevice));

    if (NVSHMEMX_SUCCESS !=
        nvshmemx_collective_launch((const void *)func, gridsize, blocksize, args, 0, 0)) {
        ERROR_PRINT("SHMEM: [%d of %d] launch failed\n", mype, npes);
        status = -1;
        goto out;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hout, dout, gridsize * blocksize * sizeof(int), cudaMemcpyDeviceToHost));
    DEBUG_PRINT("SHMEM: [%d of %d] hout[0] : 0x%x after read from GPU\n", mype, npes, hout[0]);

    for (int idx = 0; idx < gridsize * blocksize; idx++) {
        if (hout[idx] != (int)0xDEADBEEF) {
            ERROR_PRINT("[%d] collective_launch_choose_grid failed\n", mype);
            status = -1;
            goto out;
        }
    }

out:
    finalize_wrapper();

    return status;
}
