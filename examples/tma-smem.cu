/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

/*
 * Example: TMA Shared Memory Management
 *
 * Demonstrates the nvshmemx_ask_smem / nvshmemx_give_smem APIs for
 * giving shared memory to the NVSHMEM runtime for TMA-based transfers.
 *
 * Usage:
 *   Set NVSHMEM_TMA_POLICY=ENABLE to activate TMA support, then run
 *   with any NVSHMEM-compatible launcher.
 *
 *   export NVSHMEM_TMA_POLICY=ENABLE
 *   nvshmrun -np 2 ./tma-smem
 */

#include <stdio.h>
#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define NUM_ELEMS 1024

/*
 * Kernel that demonstrates giving shared memory to NVSHMEM and then
 * performing a put operation from shared memory. When TMA is enabled and the
 * architecture supports it (SM90+), NVSHMEM uses the provided shared memory as
 * the source buffer for TMA-backed transfers to remote global memory.
 */
__global__ void tma_smem_put_kernel(int *recv_data, int num_elems, int mype, int npes) {
    extern __shared__ char nvshmem_smem[];
    int *payload = (int *)nvshmem_smem;
    int tid = threadIdx.x;

    /* Step 1: Give shared memory to NVSHMEM for TMA-based transfers */
    nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    __syncthreads();

    /* Step 2: Fill shared memory with data to send */
    if (tid < num_elems) {
        payload[tid] = mype * 1000 + tid;
    }
    __syncthreads();

    /* Step 3: Fence to make smem stores visible to the TMA async proxy engine */
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif

    /* Step 4: Put from shared memory to remote PE's global memory */
    int peer = (mype + 1) % npes;
    nvshmemx_putmem_nbi_block(recv_data, nvshmem_smem, (size_t)num_elems * sizeof(int), peer);
    nvshmem_quiet();

    /* Step 5: Release smem registration so subsequent kernels start clean */
    nvshmemx_release_smem();
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    int *recv_data;

#ifdef NVSHMEMTEST_MPI_SUPPORT
    bool use_mpi = false;
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) {
        nvshmemi_init_mpi(&c, &v);
    } else
        nvshmem_init();
#else
    nvshmem_init();
#endif

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(mype_node));

    /* Query how much shared memory NVSHMEM wants (can be done on host) */
    int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
    printf("[PE %d] NVSHMEM recommended shared memory: %d bytes\n", mype, smem_size);

    /* Allocate symmetric memory for receive buffer */
    recv_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
    if (!recv_data) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed for %zu bytes\n", mype,
                sizeof(int) * NUM_ELEMS);
        nvshmem_finalize();
        return 1;
    }

    CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * NUM_ELEMS));

    /* Opt in to > 48 KiB dynamic shared memory if needed */
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(tma_smem_put_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    /* Launch kernel: fill smem, give to NVSHMEM, put from smem to remote gmem */
    tma_smem_put_kernel<<<1, NUM_ELEMS, smem_size>>>(recv_data, NUM_ELEMS, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    /* Barrier ensures all PEs have completed their puts before any PE validates.
     * cudaDeviceSynchronize() only guarantees the LOCAL kernel finished;
     * the sending PE may still be in flight without this barrier. */
    nvshmem_barrier_all();

    /* Validate: recv_data should contain values from the previous PE */
    int *host = new int[NUM_ELEMS];
    CUDA_CHECK(cudaMemcpy(host, recv_data, NUM_ELEMS * sizeof(int), cudaMemcpyDefault));
    int prev_pe = (mype - 1 + npes) % npes;
    bool success = true;
    for (int i = 0; i < NUM_ELEMS; ++i) {
        int expected = prev_pe * 1000 + i;
        if (host[i] != expected) {
            printf("[PE %d] Error at %d: got %d, expected %d\n", mype, i, host[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[PE %d of %d] TMA smem example: PASSED\n", mype, npes);
    } else {
        printf("[PE %d of %d] TMA smem example: FAILED\n", mype, npes);
    }

    delete[] host;
    nvshmem_free(recv_data);
    nvshmem_finalize();

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) nvshmemi_finalize_mpi();
#endif
    return success ? 0 : 1;
}
