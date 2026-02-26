/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

int main(int c, char *v[]) {
    int noderank, rank, nranks, nnoderanks;
    int mype_node, npes_node, npes_per_gpu;
    MPI_Comm mpi_comm, node_comm;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int old_device = 0, device = 43, dev_count;
    int err = 0;

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));

    MPI_Init(&c, &v);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    err = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Comm_split_type: %d\n", err);
        MPI_Abort(MPI_COMM_WORLD, err);
        exit(1);
    }

    MPI_Comm_rank(node_comm, &noderank);
    MPI_Comm_size(node_comm, &nnoderanks);

    printf("MPI: [%d of %d] in world comm, node rank: %d of %d in node comm\n", rank, nranks,
           noderank, nnoderanks);
    printf("attr.args.version: %d\n", attr.args.version);
    printf("attr.version: %d\n", attr.version);

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    attr.args.cuda_device_id = (noderank + 1) % dev_count;
    if (cudaGetDevice(&old_device) == cudaSuccess) {
        fprintf(stderr, "Warning: Device already selected before initialization: %d\n", old_device);
    }
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (cudaGetDevice(&device) != cudaSuccess) {
        fprintf(stderr, "Device not selected after initialization: %d\n", device);
        err = 1;
    }

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    if (mype_node != noderank) {
        fprintf(stderr, "notice: MPI and NVSHMEM node rank mismatch: %d != %d\n", mype_node,
                noderank);
    }
    if (npes_node != nnoderanks) {
        fprintf(stderr, "notice: MPI and NVSHMEM node rank mismatch: %d != %d\n", npes_node,
                nnoderanks);
    }
    if (device != attr.args.cuda_device_id) {
        fprintf(stderr, "Incorrect device selected after initialization: %d\n", device);
        err = 1;
    }
    if (device == attr.args.cuda_device_id) {
        fprintf(stderr, "Correct device selected after initialization: %d\n", device);
    }

#ifdef _NVSHMEM_DEBUG
    int mype, npes;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    nvshmem_finalize();

    MPI_Finalize();

    return err;
}
