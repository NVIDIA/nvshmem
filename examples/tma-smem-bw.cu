/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * See License.txt for license information
 */

/*
 * Bandwidth benchmark: TMA (smem→remote) vs normal put (gmem→remote)
 *
 * Goal: verify that TMA achieves ~100% of normal put bandwidth.
 * In real workloads data is already in smem after computation, so the
 * gmem→smem staging cost is not part of the "put" cost.  Here smem is
 * populated from registers (a simple store loop) to isolate just the
 * smem→remote transfer, matching the real use-case cost model.
 *
 * Set NVSHMEM_TMA_POLICY=ENABLE to activate TMA-backed transfers.
 */

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "bootstrap_helper.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t _r = (stmt);                                                  \
        if (_r != cudaSuccess) {                                                  \
            fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_r));                                       \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

#define THREADS_PER_CTA  128
#define WARMUP_ITERS     10
#define BENCH_ITERS      50

/*
 * TMA kernel: smem is pre-filled from registers (simulating data that
 * already lives in smem after computation), then given to NVSHMEM and
 * put to the remote PE.  This measures pure smem→remote bandwidth.
 */
__global__ void put_tma_kernel(char *dst, size_t bytes_per_cta, int peer) {
    extern __shared__ char smem[];

    /* Fill smem from registers — models data already computed into smem */
    int val = threadIdx.x + blockIdx.x;
    for (int i = threadIdx.x; i < (int)(bytes_per_cta / sizeof(int)); i += blockDim.x)
        ((int *)smem)[i] = val;
    __syncthreads();

    if (threadIdx.x == 0)
        nvshmemx_give_smem(smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    __syncthreads();

    nvshmemx_putmem_nbi_block(dst + (size_t)blockIdx.x * bytes_per_cta,
                               smem, bytes_per_cta, peer);
    __syncthreads();
    if (threadIdx.x == 0) nvshmem_quiet();
}

/*
 * Normal kernel: put from global memory (gmem→remote).
 */
__global__ void put_normal_kernel(const char *src, char *dst,
                                   size_t bytes_per_cta, int peer) {
    nvshmemx_putmem_nbi_block(dst + (size_t)blockIdx.x * bytes_per_cta,
                               src + (size_t)blockIdx.x * bytes_per_cta,
                               bytes_per_cta, peer);
    __syncthreads();
    if (threadIdx.x == 0) nvshmem_quiet();
}

static double measure_bw_tma(int n_ctas, char *dst, int peer,
                              size_t bytes_per_cta, int smem_per_cta) {
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < WARMUP_ITERS; i++)
        put_tma_kernel<<<n_ctas, THREADS_PER_CTA, smem_per_cta>>>(
            dst, bytes_per_cta, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < BENCH_ITERS; i++)
        put_tma_kernel<<<n_ctas, THREADS_PER_CTA, smem_per_cta>>>(
            dst, bytes_per_cta, peer);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    nvshmem_barrier_all();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return (double)n_ctas * bytes_per_cta * BENCH_ITERS / (ms * 1e-3) / 1e9;
}

static double measure_bw_normal(int n_ctas, const char *src, char *dst,
                                 int peer, size_t bytes_per_cta) {
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < WARMUP_ITERS; i++)
        put_normal_kernel<<<n_ctas, THREADS_PER_CTA, 0>>>(
            src, dst, bytes_per_cta, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < BENCH_ITERS; i++)
        put_normal_kernel<<<n_ctas, THREADS_PER_CTA, 0>>>(
            src, dst, bytes_per_cta, peer);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    nvshmem_barrier_all();

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return (double)n_ctas * bytes_per_cta * BENCH_ITERS / (ms * 1e-3) / 1e9;
}

int main(int argc, char *argv[]) {
    int mype, npes, mype_node;

#ifdef NVSHMEMTEST_MPI_SUPPORT
    bool use_mpi = false;
    char *val = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (val) use_mpi = atoi(val);
    if (use_mpi) nvshmemi_init_mpi(&argc, &argv);
    else         nvshmem_init();
#else
    nvshmem_init();
#endif

    mype      = nvshmem_my_pe();
    npes      = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));

    int peer = (mype + 1) % npes;

    int smem_per_cta  = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
    size_t bytes_per_cta = (size_t)smem_per_cta;

    CUDA_CHECK(cudaFuncSetAttribute(put_tma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_per_cta));

    if (mype == 0) {
        printf("bytes_per_cta = %zu KiB,  smem_per_cta = %d KiB,  threads = %d\n",
               bytes_per_cta / 1024, smem_per_cta / 1024, THREADS_PER_CTA);
        printf("TMA kernel: smem filled from registers (models post-compute smem)\n");
        printf("Normal kernel: put directly from global memory\n\n");
        printf("%-8s  %-10s  %-18s  %-16s  %s\n",
               "n_ctas", "total_MiB", "normal_GB/s(pe0)", "tma_GB/s(pe0)", "tma/normal");
        printf("%-8s  %-10s  %-18s  %-16s  %s\n",
               "------", "---------", "----------------", "-------------", "----------");
    }

    for (int n_ctas = 1; n_ctas <= 128; n_ctas *= 2) {
        size_t total_bytes = (size_t)n_ctas * bytes_per_cta;

        char *src = (char *)nvshmem_malloc(total_bytes);
        char *dst = (char *)nvshmem_malloc(total_bytes);
        assert(src && dst);
        CUDA_CHECK(cudaMemset(src, 1, total_bytes));
        CUDA_CHECK(cudaMemset(dst, 0, total_bytes));

        double bw_normal = measure_bw_normal(n_ctas, src, dst, peer, bytes_per_cta);
        double bw_tma    = measure_bw_tma(n_ctas, dst, peer, bytes_per_cta, smem_per_cta);

        double bw_normal_total = 0, bw_tma_total = 0;
        nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, &bw_normal_total, &bw_normal, 1);
        nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, &bw_tma_total,    &bw_tma,    1);

        if (mype == 0)
            printf("%-8d  %-10zu  %-18.2f  %-16.2f  %.0f%%\n",
                   n_ctas, total_bytes / (1024 * 1024),
                   bw_normal, bw_tma,
                   100.0 * bw_tma / bw_normal);

        nvshmem_free(src);
        nvshmem_free(dst);
    }

    nvshmem_finalize();
#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) nvshmemi_finalize_mpi();
#endif
    return 0;
}
