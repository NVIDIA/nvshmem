/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nonuniform_teams.cuh"

// Select unequal participation from a uniform MPI job. Optional arguments are
// counts ("auto" or a comma-separated count per node) and PE ordering.
int main(int argc, char **argv) {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Init(&argc, &argv);
    int rank, world_size, local_rank, local_size, node = 0, nodes = 0;
    MPI_Comm local, leaders;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_split(MPI_COMM_WORLD, local_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leaders);
    if (local_rank == 0) {
        MPI_Comm_rank(leaders, &node);
        MPI_Comm_size(leaders, &nodes);
        MPI_Comm_free(&leaders);
    }
    MPI_Bcast(&node, 1, MPI_INT, 0, local);
    MPI_Bcast(&nodes, 1, MPI_INT, 0, local);
    int count = std::min(node + 1, local_size);
    if (argc > 1 && strcmp(argv[1], "auto")) {
        std::stringstream shape(argv[1]);
        std::string item;
        std::vector<int> counts;
        while (std::getline(shape, item, ',')) {
            counts.push_back(std::stoi(item));
        }
        if (static_cast<int>(counts.size()) != nodes || counts[node] < 0 ||
            counts[node] > local_size) {
            fprintf(stderr, "one valid participating count required per node\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        count = counts[node];
    }
    const char *order = argc > 2 ? argv[2] : "block";
    int key = rank;
    if (!strcmp(order, "cyclic")) {
        key = local_rank * nodes + node;
    } else if (!strcmp(order, "permuted")) {
        key = (rank * (world_size - 1) + local_rank * 3) % world_size;
    } else if (strcmp(order, "block")) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_split(MPI_COMM_WORLD, local_rank < count ? 0 : MPI_UNDEFINED, key, &comm);
    MPI_Comm_free(&local);
    if (comm != MPI_COMM_NULL) {
        int devices;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &comm;
        if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (comm != MPI_COMM_NULL) {
        check_nonuniform_teams();
        nvshmem_finalize();
        MPI_Comm_free(&comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
