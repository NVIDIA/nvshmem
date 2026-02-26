/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

int main(int c, char *v[]) {
    int mype_node, npes_node;
    int dev_count;

    nvshmem_init();

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(mype_node / npes_per_gpu));

#ifdef _NVSHMEM_DEBUG
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
#endif
    DEBUG_PRINT("[%d of %d] hello shmem world! \n", mype, npes);

    nvshmem_barrier_all();

    nvshmem_finalize();

    return 0;
}
