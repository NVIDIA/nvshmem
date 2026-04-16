/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

int main(int c, char *v[]) {
    int rank, nranks;
    int mype_node, npes_node;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    int dev_count;
    MPI_Init(&c, &v);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
    if (rank == 0) {
        nvshmemx_get_uniqueid(&id);
    }

    MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
    /* Verify if structure is set correctly */
    assert(attr.args.uid_args.id == &id);
    assert(attr.args.uid_args.myrank == rank);
    assert(attr.args.uid_args.nranks == nranks);

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(mype_node / npes_per_gpu));

#ifdef _NVSHMEM_DEBUG
    int mype, npes;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    nvshmem_finalize();

    MPI_Finalize();

    return 0;
}
