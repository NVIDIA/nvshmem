/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#if defined(NVSHMEMTEST_MPI_SUPPORT)
#include <mpi.h>
#endif
#include "nvshmem.h"
#include "nvshmemx.h"

#if !defined(NVSHMEM_CFT_HANDLES_SUPPORT)
#error "putmem_signal_counted requires NVSHMEM_CFT_HANDLES_SUPPORT"
#endif

constexpr size_t kBytes = 256;
constexpr size_t kCounterAlignment = 256;

__global__ void counted_ring(void *destination, uint64_t *counter, int pe) {
    extern __shared__ __align__(16) unsigned char smem[];
    size_t reserve = (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    unsigned char *payload = smem + reserve;
    nvshmemx_give_smem(smem, reserve);

    for (size_t i = threadIdx.x; i < kBytes; i += blockDim.x) payload[i] = (unsigned char)pe;
    __syncthreads();

    int status =
        nvshmemx_putmem_signal_counted_nbi_block(destination, payload, kBytes, counter, pe);
    if (status == NVSHMEMX_SUCCESS) nvshmemx_signal_counted_wait_until(counter, kBytes);
    nvshmemx_release_smem();
}

int main(int argc, char **argv) {
#if defined(NVSHMEMTEST_MPI_SUPPORT)
    MPI_Init(&argc, &argv);
    MPI_Comm node_comm;
    int rank, local_rank, device_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status == cudaSuccess) {
        cuda_status =
            device_count == 0 ? cudaErrorNoDevice : cudaSetDevice(local_rank % device_count);
    }
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "failed to select GPU: %s\n", cudaGetErrorString(cuda_status));
        MPI_Finalize();
        return 1;
    }

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
        MPI_Finalize();
        return 1;
    }
#else
    (void)argc;
    (void)argv;
    nvshmem_init();
#endif
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    void *destination = nvshmem_align(16, kBytes);
    uint64_t *counter = (uint64_t *)nvshmem_align(kCounterAlignment, kCounterAlignment);
    nvshmemx_signal_counted_reset(counter);
    nvshmem_barrier_all();

    size_t reserve = (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    counted_ring<<<1, 128, reserve + kBytes>>>(destination, counter, (mype + 1) % npes);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        std::fprintf(stderr, "counted ring failed: %s\n", cudaGetErrorString(error));
    nvshmem_free(counter);
    nvshmem_free(destination);
    nvshmem_finalize();
#if defined(NVSHMEMTEST_MPI_SUPPORT)
    MPI_Finalize();
#endif
    return 0;
}
