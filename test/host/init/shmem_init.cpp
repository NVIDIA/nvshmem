/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "utils.h"

int main(int c, char *v[]) {
    int nv_npes_node, nv_mype_node;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int dev_count;
    shmem_init();
    DEBUG_PRINT("shmem_init done\n");

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);
#ifdef _NVSHMEM_DEBUG
    int nv_mype, nv_npes;
    nv_mype = nvshmem_my_pe();
    nv_npes = nvshmem_n_pes();
    DEBUG_PRINT("NVSHMEM: [%d of %d] hello nvshmem world! \n", nv_mype, nv_npes);
#endif
    nv_mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nv_npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    DEBUG_PRINT("NVSHMEM TEAM NODE: [%d of %d] hello nvshmem team node world! \n", nv_mype_node,
                nv_npes_node);
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (nv_npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(nv_mype_node / npes_per_gpu));

    nvshmem_barrier_all();
    shmem_barrier_all();

    nvshmem_finalize();

    shmem_finalize();

    return 0;
}
