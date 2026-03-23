/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * See License.txt for license information
 */

/*
 * Bandwidth benchmark: TMA (smem→remote) vs normal put (gmem→remote)
 *
 * Sweeps both message size (4K–8M per CTA) and CTA count (1–128).
 * For sizes larger than smem (64K), the TMA kernel loops over smem-sized
 * chunks, issuing each as an NBI put, then calls quiet once at the end.
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
 * TMA kernel: for each smem-sized chunk of bytes_per_cta, fill smem from
 * registers, give to NVSHMEM, and issue an NBI put.  Quiet once at the end.
 * smem_size is always the recommended smem size (64K); bytes_per_cta may be
 * smaller (sub-64K sizes) or a multiple of smem_size (larger sizes).
 */
__global__ void put_tma_kernel(char *dst, size_t bytes_per_cta,
                                int smem_size, int peer) {
    extern __shared__ char smem[];

    int val = threadIdx.x + blockIdx.x;

    /* Register the smem buffer once for this CTA */
    if (threadIdx.x == 0)
        nvshmemx_give_smem(smem, smem_size);
    __syncthreads();

    size_t chunk    = (size_t)smem_size;
    size_t n_chunks = (bytes_per_cta + chunk - 1) / chunk;

    for (size_t c = 0; c < n_chunks; c++) {
        size_t this_bytes = (c < n_chunks - 1) ? chunk
                                               : (bytes_per_cta - c * chunk);

        /* Fill smem from registers */
        for (int i = threadIdx.x; i < (int)(this_bytes / sizeof(int)); i += blockDim.x)
            ((int *)smem)[i] = val;
        __syncthreads();

        /* Caller's responsibility: make smem writes visible to the TMA proxy */
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        /* Blocking put handles: cp.async.bulk + commit + wait_group 0 + __threadfence_system */
        nvshmemx_putmem_block(
            dst + (size_t)blockIdx.x * bytes_per_cta + c * chunk,
            smem, this_bytes, peer);
    }
}

/*
 * Normal kernel: put directly from global memory (gmem→remote).
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
                              size_t bytes_per_cta, int smem_size) {
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int i = 0; i < WARMUP_ITERS; i++)
        put_tma_kernel<<<n_ctas, THREADS_PER_CTA, smem_size>>>(
            dst, bytes_per_cta, smem_size, peer);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < BENCH_ITERS; i++)
        put_tma_kernel<<<n_ctas, THREADS_PER_CTA, smem_size>>>(
            dst, bytes_per_cta, smem_size, peer);
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

    /* Confirm each PE has a distinct ID */
    printf("[PE %d / %d] peer=%d\n", mype, npes, peer);
    fflush(stdout);
    nvshmem_barrier_all();

    int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);

    CUDA_CHECK(cudaFuncSetAttribute(put_tma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size));

    /* Sizes to sweep: 4K to 8M */
    static const size_t sizes[] = {
        4*1024, 8*1024, 16*1024, 32*1024, 64*1024,
        128*1024, 256*1024, 512*1024,
        1*1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024
    };
    static const int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    static const int cta_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    static const int n_cta_counts = (int)(sizeof(cta_counts) / sizeof(cta_counts[0]));

    /* Allocate max needed: 128 CTAs × 8M = 1 GB */
    size_t max_per_cta = sizes[n_sizes - 1];
    int    max_ctas    = cta_counts[n_cta_counts - 1];
    size_t alloc_bytes = (size_t)max_ctas * max_per_cta;

    char *src = (char *)nvshmem_malloc(alloc_bytes);
    char *dst = (char *)nvshmem_malloc(alloc_bytes);
    assert(src && dst);
    CUDA_CHECK(cudaMemset(src, 1, alloc_bytes));
    CUDA_CHECK(cudaMemset(dst, 0, alloc_bytes));

    /* Print header */
    if (mype == 0) {
        printf("  n_ctas\\sz  ");
        for (int s = 0; s < n_sizes; s++) {
            size_t kb = sizes[s] / 1024;
            if (kb < 1024) printf(" %6zuK", kb);
            else           printf(" %6zuM", kb / 1024);
        }
        printf("\n");
        printf("  ----------");
        for (int s = 0; s < n_sizes; s++) printf("  ------");
        printf("\n");
    }

    for (int ci = 0; ci < n_cta_counts; ci++) {
        int n_ctas = cta_counts[ci];

        /* normal row */
        if (mype == 0) printf("  %-10d", n_ctas);
        for (int s = 0; s < n_sizes; s++) {
            size_t bpc = sizes[s];
            double bw = measure_bw_normal(n_ctas, src, dst, peer, bpc);
            if (mype == 0) printf("  %6.1f", bw);
        }
        if (mype == 0) printf("  (normal)\n");

        /* tma row */
        if (mype == 0) printf("  %-10s", "");
        for (int s = 0; s < n_sizes; s++) {
            size_t bpc = sizes[s];
            double bw = measure_bw_tma(n_ctas, dst, peer, bpc, smem_size);
            if (mype == 0) printf("  %6.1f", bw);
        }
        if (mype == 0) printf("  (tma)\n");

        /* ratio row — recompute both */
        if (mype == 0) printf("  %-10s", "");
        for (int s = 0; s < n_sizes; s++) {
            size_t bpc = sizes[s];
            double bw_n = measure_bw_normal(n_ctas, src, dst, peer, bpc);
            double bw_t = measure_bw_tma(n_ctas, dst, peer, bpc, smem_size);
            if (mype == 0) printf("  %5.0f%%", 100.0 * bw_t / bw_n);
        }
        if (mype == 0) printf("  (tma/normal)\n\n");
    }

    nvshmem_free(src);
    nvshmem_free(dst);

    nvshmem_finalize();
#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) nvshmemi_finalize_mpi();
#endif
    return 0;
}
