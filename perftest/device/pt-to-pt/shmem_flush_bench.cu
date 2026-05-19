/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Benchmark: pipelined put bandwidth - nvshmem_quiet() vs
 * nvshmemx_flush(), with source in shared memory.
 *
 * Canonical usage:
 *
 *   __global__ void kernel(...) {
 *       produce(src_smem);                     // fill local smem
 *       nvshmemx_put_nbi_*(dst_remote, src_smem, N, pe);
 *       nvshmemx_flush();  // src_smem reusable
 *   }
 *
 * Two axes:
 *   USE_SMEM (0/1): whether the kernel calls give_smem, opting NVSHMEM
 *     into the TMA path.  With source already in smem:
 *       USE_SMEM=1 -> one cp.async.bulk smem -> remote gmem.  The issuing
 *                     thread returns as soon as the descriptor is enqueued;
 *                     the TMA engine reads smem asynchronously.  This is
 *                     where the flush API should win.
 *       USE_SMEM=0 -> plain st.global path: all 64 threads issue
 *                     st.volatile.global.v4 from smem to remote gmem, in
 *                     parallel.  put_nbi does not return until the stores
 *                     retire, so the source is already consumed by return.
 *   USE_FLUSH (0/1): per-iter sync is nvshmem_quiet() (=0) or
 *     nvshmemx_flush() + one final quiet (=1).
 *
 * Expectation:
 *   TMA + flush should achieve higher pipelined BW than st.global +
 *   flush, because the TMA variant returns from put_nbi before NVLink
 *   is finished, lets the producer start filling next tile, and only waits
 *   for the local engine read when reuse is actually needed.  st.global
 *   serializes the producer on NVLink every iter.
 *
 * Also run with NVSHMEM_REMOTE_TRANSPORT=none so job_connectivity ==
 * NVSHMEMI_JOB_GPU_LDST and nvshmemi_flush's network branch is
 * skipped (no proxy/CQ flush in the hot path).
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

/* smem layout when give_smem is in use: [barrier region 512 B][data].
 * When give_smem is NOT called, the barrier region is unused; we still
 * place the data at the same +512 offset so the two variants have
 * identical source alignment and cache behavior. (Hardcoded to match
 * NVSHMEMI_TMA_BARRIER_REGION_BYTES; the macro itself has a header
 * visibility quirk in this TU.) */
constexpr size_t kSmemDataOffset = 512;

template <bool USE_SMEM, bool USE_FLUSH>
__global__ void pipelined_put_smem_src(double *dst, size_t nelems, int peer, size_t iter) {
    extern __shared__ char nvshmem_smem[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    if constexpr (USE_SMEM) {
        nvshmemx_give_smem(nvshmem_smem, (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
        __syncthreads();
    }

    /* Source lives in smem.  Data starts after the barrier carve-out so the
     * TMA variant has room for its mbarriers; the st.global variant uses the
     * same offset for an apples-to-apples source pointer. */
    double *src_smem = reinterpret_cast<double *>(nvshmem_smem + kSmemDataOffset);

    for (size_t i = 0; i < iter; i++) {
        /* Producer: fill src_smem with iteration-dependent data so the
         * compiler cannot CSE or hoist the put.  Only threads with work
         * participate, but all threads syncthreads after. */
        for (size_t j = tid; j < nelems; j += blockDim.x * blockDim.y * blockDim.z)
            src_smem[j] = (double)(i * 31337 + (int)j);
        __syncthreads();

        /* put_nbi_block.  Dispatch differs per transport:
         *   USE_SMEM=1 (TMA registered) + src is shared: takes the TMA
         *     shared_global path.  Issues ONE cp.async.bulk smem->remote_gmem
         *     from the elected leader and returns.  NVLink is still busy.
         *   USE_SMEM=0: takes the memcpy_threadgroup st.volatile.global.v4
         *     path.  All 64 threads issue stores in parallel; the call
         *     returns only after all have retired. */
        nvshmemx_double_put_nbi_block(dst, src_smem, nelems, peer);

        if constexpr (USE_FLUSH) {
            /* Flush waits only for source-buffer reusability.  For the
             * st.global variant this is a no-op; for TMA it stalls until the
             * engine has drained the source smem.  No membar.sys is issued. */
            nvshmemx_flush();
        } else {
            /* Quiet every iter: __threadfence_system (membar.sys, ~300 ns)
             * plus waits for all prior outbound stores to be remotely
             * visible.  Serializes the pipeline. */
            nvshmem_quiet();
        }
        __syncthreads();
    }

    /* Final quiet to ensure remote visibility for the bench's "after"
     * condition.  The quiet-per-iter variant has already synced each tile;
     * an extra quiet here is a no-op cost-wise. */
    nvshmem_quiet();

    if constexpr (USE_SMEM) nvshmemx_release_smem();
}

static double bw_gbs(size_t bytes_per_iter, size_t iters, float ms) {
    double secs = ms / 1000.0;
    double bytes = (double)bytes_per_iter * (double)iters;
    return bytes / secs / 1.0e9;
}

int main(int argc, char *argv[]) {
    int mype = -1, npes = -1;
    double *dst_d = NULL;
    int max_threads = 0;
    int smem_size = 0;
    int peer = 0;
    float ms_stg_quiet = 0, ms_stg_flush = 0;
    float ms_tma_quiet = 0, ms_tma_flush = 0;
    cudaEvent_t start = NULL, stop = NULL;

    read_args(argc, argv);
    max_threads = (int)threads_per_block;

    init_wrapper(&argc, &argv);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    if (npes != 2) {
        if (mype == 0) fprintf(stderr, "This test requires exactly two processes\n");
        goto finalize;
    }

    dst_d = (double *)nvshmem_malloc(max_size);
    if (!dst_d) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        goto finalize;
    }

    /* Size smem to the device's MAX per-CTA dynamic smem.  We want to run
     * an exhaustive sweep up to smem capacity, so we intentionally go
     * beyond NVSHMEMX_SMEM_RECOMMENDED (= 64 KiB, which doesn't fit a
     * 64 KiB source tile + 512 B barrier reserve).  Clamp the user-supplied
     * max_size to whatever the device allows minus our barrier region. */
    {
        int dev = 0;
        int max_dyn_smem = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&max_dyn_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        smem_size = max_dyn_smem;
        size_t max_data_bytes = (size_t)smem_size - kSmemDataOffset;
        if (max_size > max_data_bytes) {
            if (mype == 0)
                fprintf(stderr,
                        "clamping max_size from %zu to %zu (device max dynamic smem %d, "
                        "barrier reserve %zu)\n",
                        max_size, max_data_bytes, max_dyn_smem, kSmemDataOffset);
            max_size = max_data_bytes & ~(size_t)15; /* keep 16 B aligned */
        }
    }
    CUDA_CHECK(cudaFuncSetAttribute(pipelined_put_smem_src<true, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(pipelined_put_smem_src<true, false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(pipelined_put_smem_src<false, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(pipelined_put_smem_src<false, false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    if (mype == 0) {
        printf("# shmem_flush_bench - pipelined put BW (source in smem)\n");
        printf("#   iters=%zu, threads=%d, CTAs=1\n", (size_t)iters, max_threads);
        printf("#   Columns: aggregate GB/s for each (transport, per-iter-sync) combo.\n");
        printf(
            "#   bytes   st.g+quiet  st.g+flush     TMA+quiet   TMA+flush     "
            "TMA-flush/st.g-flush\n");
    }

    peer = !mype;

    /* Loop: stride by step_factor, then always emit one final point at
     * max_size so the sweep reaches device-smem capacity even when
     * max_size is not a clean power of the step factor. */
    for (size_t size = min_size;; size *= step_factor) {
        if (size > max_size) size = max_size;
        size_t nelems = size / sizeof(double);
        if (nelems == 0) continue;

        auto run = [&](auto kernel_ptr, size_t n) {
            kernel_ptr<<<1, max_threads, smem_size>>>(dst_d, nelems, peer, n);
        };

        /* st.global + quiet */
        if (mype == 0) {
            run(pipelined_put_smem_src<false, false>, warmup_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(start);
            run(pipelined_put_smem_src<false, false>, iters);
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms_stg_quiet, start, stop);
        }
        nvshmem_barrier_all();

        /* st.global + flush */
        if (mype == 0) {
            run(pipelined_put_smem_src<false, true>, warmup_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(start);
            run(pipelined_put_smem_src<false, true>, iters);
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms_stg_flush, start, stop);
        }
        nvshmem_barrier_all();

        /* TMA + quiet */
        if (mype == 0) {
            run(pipelined_put_smem_src<true, false>, warmup_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(start);
            run(pipelined_put_smem_src<true, false>, iters);
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms_tma_quiet, start, stop);
        }
        nvshmem_barrier_all();

        /* TMA + flush */
        if (mype == 0) {
            run(pipelined_put_smem_src<true, true>, warmup_iters);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(start);
            run(pipelined_put_smem_src<true, true>, iters);
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms_tma_flush, start, stop);
        }
        nvshmem_barrier_all();

        if (mype == 0) {
            double g_stg_q = bw_gbs(size, iters, ms_stg_quiet);
            double g_stg_f = bw_gbs(size, iters, ms_stg_flush);
            double g_tma_q = bw_gbs(size, iters, ms_tma_quiet);
            double g_tma_f = bw_gbs(size, iters, ms_tma_flush);
            double speedup = (g_stg_f > 0.0) ? (g_tma_f / g_stg_f) : 0.0;
            printf("  %6zu   %10.3f  %10.3f  %10.3f  %9.3f  %6.2fx\n", size, g_stg_q, g_stg_f,
                   g_tma_q, g_tma_f, speedup);
        }
        if (size == max_size) break;
    }

finalize:
    if (dst_d) nvshmem_free(dst_d);
    finalize_wrapper();
    return 0;
}
