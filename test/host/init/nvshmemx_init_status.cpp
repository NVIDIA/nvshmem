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

    assert(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED);
    nvshmem_init();
    assert(nvshmemx_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED);
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
    assert(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);

    if (npes_per_gpu > 1) assert(nvshmemx_init_status() >= NVSHMEM_STATUS_LIMITED_MPG);

    nvshmem_finalize();
    assert(nvshmemx_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED);

    nvshmem_init();
    assert(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);
    nvshmem_finalize();

    return 0;
}
