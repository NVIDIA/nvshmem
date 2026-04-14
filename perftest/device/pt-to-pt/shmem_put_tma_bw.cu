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
 * Bandwidth benchmark for nvshmemx_putmem_nbi_block from CTA shared memory
 * to remote global memory, using NVSHMEM's TMA-backed NBI transfer path.
 *
 * Source data is staged in shared memory (registered via nvshmemx_give_smem),
 * submitted in smem-sized chunks via nvshmemx_putmem_nbi_block, and completed
 * with nvshmem_quiet() at the end of each iteration.  Between chunks,
 * cp.async.bulk.wait_group.read 0 ensures smem is safe to reuse without
 * stalling the remote write pipeline.
 *
 * Requires NVSHMEM_TMA_POLICY=ENABLE and sm_90+ hardware.
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

/*
 * Each CTA independently fills its smem, issues NBI TMA puts for all
 * smem-sized chunks of its portion of the transfer, then quiets.
 *
 * give_smem is called once before the iteration loop; the smem address is
 * stable across iterations within a single kernel invocation.
 *
 * cp.async.bulk group state is per-thread: only the thread that issued the
 * TMA op (the warp's elected leader via elect.sync) has a pending group.
 * nvshmem_quiet() is THREAD scope — it drains only the calling thread's
 * pending groups.  Calling it from every thread is safe: non-issuing threads
 * have no pending groups, so their commit_group + wait_group 0 are no-ops.
 * This avoids assuming that elect.sync always picks lane 0.
 */
__global__ void bw_smem_tma(char *dst, size_t bytes, int smem_size, int peer, int iter) {
    extern __shared__ char smem[];

    int tid     = threadIdx.x;
    int bid     = blockIdx.x;
    int nblocks = gridDim.x;

    size_t bytes_per_block = bytes / nblocks;
    char  *block_dst       = dst + (size_t)bid * bytes_per_block;

    /* Register this CTA's smem with NVSHMEM for TMA (once per kernel launch) */
    nvshmemx_give_smem(smem, smem_size);
    __syncthreads();

    size_t chunk    = (size_t)smem_size;
    size_t n_chunks = (bytes_per_block + chunk - 1) / chunk;

    for (int i = 0; i < iter; i++) {
        for (size_t c = 0; c < n_chunks; c++) {
            size_t this_bytes = (c < n_chunks - 1) ? chunk
                                                   : (bytes_per_block - c * chunk);

            /* Fill smem with identifiable data */
            for (int j = tid; j < (int)(this_bytes / sizeof(int)); j += blockDim.x)
                ((int *)smem)[j] = tid;
            __syncthreads();

            /* Make smem stores visible to the TMA async proxy before submitting */
#if __CUDA_ARCH__ >= 900
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif

            /* Submit NBI: cp.async.bulk + commit_group inside, no wait yet */
            nvshmemx_putmem_nbi_block(block_dst + c * chunk, smem, this_bytes, peer);

            /* Between chunks: wait for smem READ to complete before reusing.
             * Called from all threads: non-issuing threads have no pending
             * groups so wait_group.read returns immediately for them. */
            if (c < n_chunks - 1) {
#if __CUDA_ARCH__ >= 900
                asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
                __syncthreads();
            }
        }

        /* All chunks submitted; drain all warps' pending TMA groups.
         * nvshmem_quiet() is THREAD scope; calling from every thread ensures
         * each warp's elected leader drains its own groups, regardless of
         * which lane elect.sync chose.  __syncthreads() orders the fence. */
        nvshmem_quiet();
        __syncthreads();
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    char *dst = NULL;

    read_args(argc, argv);
    int max_blocks  = (int)num_blocks;
    int max_threads = (int)threads_per_block;

    int       array_size;
    void    **h_tables;
    uint64_t *h_size_arr;
    double   *h_bw = NULL;
    float     milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes\n");
        goto finalize;
    }

    {
        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);

        CUDA_CHECK(cudaFuncSetAttribute(bw_smem_tma,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));

        array_size = max_size_log;
        alloc_tables(&h_tables, 2, array_size);
        h_size_arr = (uint64_t *)h_tables[0];
        h_bw       = (double   *)h_tables[1];

        dst = (char *)nvshmem_malloc(max_size);
        if (!dst) {
            fprintf(stderr, "[PE %d] nvshmem_malloc failed for %zu bytes\n", mype, max_size);
            goto finalize;
        }
        CUDA_CHECK(cudaMemset(dst, 0, max_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        int i = 0;
        if (mype == 0) {
            int peer = 1;
            for (size_t size = min_size; size <= max_size; size *= step_factor) {
                h_size_arr[i] = size;

                /* Warmup */
                bw_smem_tma<<<max_blocks, max_threads, smem_size>>>(
                    dst, size, smem_size, peer, (int)warmup_iters);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                /* Timed */
                cudaEventRecord(start);
                bw_smem_tma<<<max_blocks, max_threads, smem_size>>>(
                    dst, size, smem_size, peer, (int)iters);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));

                cudaEventElapsedTime(&milliseconds, start, stop);
                h_bw[i] = (double)size / (milliseconds * (B_TO_GB / ((double)iters * MS_TO_S)));
                nvshmem_barrier_all();

                i++;
            }

            print_table_basic("shmem_put_tma_smem_bw", "None", "size (Bytes)",
                              "BW", "GB/sec", '+', h_size_arr, h_bw, i);
        } else {
            for (size_t size = min_size; size <= max_size; size *= step_factor) {
                nvshmem_barrier_all();
            }
        }

        /* Optional correctness check: set NVSHMEM_PERFTEST_VERIFY=1 to enable.
         * PE 0 sends one smem-sized chunk with a known pattern (element j gets
         * value j % threads_per_block).  PE 1 checks its dst buffer against
         * that pattern and prints PASS or FAIL. */
        if (getenv("NVSHMEM_PERFTEST_VERIFY")) {
            size_t verify_size = (size_t)smem_size;  /* one chunk, no multi-chunk complexity */

            /* PE 1 fills its buffer with a canary so stale data can't mask failures */
            if (mype == 1)
                CUDA_CHECK(cudaMemset(dst, 0xFF, verify_size));
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            /* PE 0 transfers one chunk with the standard fill pattern */
            if (mype == 0) {
                bw_smem_tma<<<1, max_threads, smem_size>>>(
                    dst, verify_size, smem_size, 1 /*peer*/, 1 /*iter*/);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            nvshmem_barrier_all();  /* ensure PE 1 has received the data */

            /* PE 1 copies and checks: element j should equal j % threads_per_block */
            if (mype == 1) {
                int  n_ints = (int)(verify_size / sizeof(int));
                int *h_buf  = (int *)malloc(verify_size);
                if (!h_buf) {
                    fprintf(stderr, "[PE %d] malloc failed for verify buffer\n", mype);
                    goto finalize;
                }
                CUDA_CHECK(cudaMemcpy(h_buf, dst, verify_size, cudaMemcpyDeviceToHost));

                int errors = 0;
                for (int j = 0; j < n_ints; j++) {
                    int expected = j % max_threads;
                    if (h_buf[j] != expected) {
                        if (errors < 5)
                            fprintf(stderr, "[verify] FAIL at int[%d]: got %d, expected %d\n",
                                    j, h_buf[j], expected);
                        errors++;
                    }
                }
                if (errors == 0)
                    printf("[verify] PASS (%zu bytes, pattern j%%threads_per_block)\n",
                           verify_size);
                else
                    printf("[verify] FAIL: %d / %d ints wrong\n", errors, n_ints);
                fflush(stdout);
                free(h_buf);
            }
            nvshmem_barrier_all();
        }
    }

finalize:
    if (dst) nvshmem_free(dst);
    if (h_bw) free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
