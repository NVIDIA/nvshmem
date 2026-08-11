/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

static int init_uid(int rank, int nranks) {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int status = 0;

    if (rank == 0) {
        status = nvshmemx_get_uniqueid(&id);
    }
    if (MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS || status) {
        return 1;
    }
    if (MPI_Bcast(&id, sizeof(id), MPI_UINT8_T, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        return 1;
    }

    status = nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
    if (status) {
        return status;
    }

    return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

static int select_device(int rank) {
    MPI_Comm node_comm;
    int local_rank, dev_count;

    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &node_comm) != MPI_SUCCESS) {
        return 1;
    }
    if (MPI_Comm_rank(node_comm, &local_rank) != MPI_SUCCESS) {
        MPI_Comm_free(&node_comm);
        return 1;
    }
    MPI_Comm_free(&node_comm);

    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return 1;
    }
    return cudaSetDevice(local_rank % dev_count) == cudaSuccess ? 0 : 1;
}

int main(int argc, char *argv[]) {
    int rank, nranks;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (select_device(rank)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (init_uid(rank, nranks)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    nvshmem_finalize();

    MPI_Barrier(MPI_COMM_WORLD);

    if (init_uid(rank, nranks)) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    nvshmem_finalize();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
