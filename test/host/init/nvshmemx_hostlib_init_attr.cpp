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
    nvshmemi_version_t nvshmem_version = {
        NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
    int status;
    int mype_node, npes_node;
    int dev_count;
    int use_mpi = 0;

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int init_flags = 0;

#ifdef NVSHMEMTEST_MPI_SUPPORT
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);

    if (use_mpi) {
        int rank, nranks;
        MPI_Init(&c, &v);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        init_flags = NVSHMEMX_INIT_WITH_MPI_COMM;
        attr.mpi_comm = &mpi_comm;
    }
#endif

    assert(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED);
    status = nvshmemx_hostlib_init_attr(init_flags, &attr);
    assert(status == NVSHMEMX_SUCCESS);
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

    nvshmemx_hostlib_finalize();
    assert(nvshmemx_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED);

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif

    return status;
}