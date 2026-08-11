/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Test: NVSHMEM host APIs compiled without RDC.
 *
 * Verifies that user code can #include <nvshmem.h> in a .cu file compiled
 * without -rdc=true, and use host-side APIs.  No device kernels or device
 * API calls are made.
 *
 * Uses nvshmemx_hostlib_init_attr() because this test is built via
 * nvshmem_add_test_no_device(): it links only libnvshmem_host.so and not
 * libnvshmem_device.a, so nvshmemi_init_thread (which nvshmem_init() expands
 * to) is not available at link time.  Downstream non-RDC users that do link
 * libnvshmem_device.a can call nvshmem_init() directly.
 *
 * This is the use case reported by FlashInfer: the NVSHMEM headers must be
 * parseable by nvcc's device compilation pass even when RDC is disabled.
 */
#include <cstdio>
#include <memory>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#ifdef NVSHMEMTEST_MPI_SUPPORT
#include <mpi.h>
#endif

struct NvshmemHostGuard {
    int status = NVSHMEMX_ERROR_INTERNAL;
    bool use_mpi = false;
    NvshmemHostGuard(int *argc, char ***argv) {
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        int flags = 0;
#ifdef NVSHMEMTEST_MPI_SUPPORT
        const char *mpi_env = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        const char *uid_env = getenv("NVSHMEMTEST_USE_UID_BOOTSTRAP");
        const bool want_mpi = mpi_env && atoi(mpi_env);
        const bool want_uid = uid_env && atoi(uid_env);
        if (want_mpi || want_uid) {
            use_mpi = true;
            MPI_Init(argc, argv);
        }
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (want_uid) {
            /* UID bootstrap: rank 0 generates the id, MPI_Bcast distributes it. */
            int rank = 0, nranks = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nranks);
            if (rank == 0) {
                nvshmemx_get_uniqueid(&id);
            }
            MPI_Bcast(&id, sizeof(id), MPI_UINT8_T, 0, MPI_COMM_WORLD);
            nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
            flags = NVSHMEMX_INIT_WITH_UNIQUEID;
        } else if (want_mpi) {
            attr.mpi_comm = &mpi_comm;
            flags = NVSHMEMX_INIT_WITH_MPI_COMM;
        }
#endif
        status = nvshmemx_hostlib_init_attr(flags, &attr);
    }
    ~NvshmemHostGuard() {
        if (status == NVSHMEMX_SUCCESS) {
            nvshmemx_hostlib_finalize();
        }
#ifdef NVSHMEMTEST_MPI_SUPPORT
        if (use_mpi) {
            MPI_Finalize();
        }
#endif
    }
};

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int main(int argc, char **argv) {
    /* Guard owns MPI + NVSHMEM lifecycle; destructor tears down in correct order. */
    NvshmemHostGuard nvshmem(&argc, &argv);
    if (nvshmem.status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "nvshmemx_hostlib_init_attr failed: %d\n", nvshmem.status);
        return 1;
    }

    /* Set CUDA device based on node-local PE rank */
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node / ((npes_node + dev_count - 1) / dev_count)));

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    printf("[PE %d/%d] init succeeded\n", mype, npes);

    if (npes < 1) {
        fprintf(stderr, "error: expected at least 1 PE, got %d\n", npes);
        return 1;
    }

    /* Allocated after guard — destroyed before finalize (reverse order). */
    auto dest = std::unique_ptr<int, decltype(&nvshmem_free)>((int *)nvshmem_malloc(sizeof(int)),
                                                              nvshmem_free);
    if (!dest) {
        fprintf(stderr, "[PE %d] error: nvshmem_malloc returned NULL\n", mype);
        return 1;
    }

    /* Write from host via cudaMemcpy, barrier, read back */
    int val = mype + 42;
    CUDA_CHECK(cudaMemcpy(dest.get(), &val, sizeof(int), cudaMemcpyHostToDevice));
    nvshmem_barrier_all();

    int received = 0;
    CUDA_CHECK(cudaMemcpy(&received, dest.get(), sizeof(int), cudaMemcpyDeviceToHost));

    if (received != val) {
        fprintf(stderr, "[PE %d] error: expected %d, got %d\n", mype, val, received);
        return 1;
    }

    printf("[PE %d] PASSED\n", mype);
    return 0;
}
