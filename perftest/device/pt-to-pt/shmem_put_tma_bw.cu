/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Bandwidth benchmark for NVSHMEM TMA-backed NBI put from shared memory to
 * remote global memory (smem -> remote gmem over NVLink).
 *
 * Models a warp-specialized kernel pattern where many threads fill smem with
 * computed data, then a single elected leader issues a cp.async.bulk TMA put
 * to ship it to the remote GPU.  The fill is amortized (done once before the
 * timed loop) so the benchmark measures the irreducible per-chunk overhead:
 *
 *   fence.proxy.async.shared::cta  -- make smem writes visible to TMA
 *   nvshmemx_putmem_nbi_block      -- TMA reads smem, writes to remote gmem
 *   cp.async.bulk.wait_group.read  -- wait for smem read to finish before reuse
 *
 * Both fence and wait_group.read are required each chunk:
 *   fence:           needed whenever threads have written new data to smem
 *   wait_group.read: needed to ensure TMA has read smem before it can be
 *                    safely overwritten for the next chunk
 *
 * Speed-of-Light (SoL): ~55 GB/s per CTA (1 CTA, GB200 NVLink 5.0).
 * Requires NVSHMEM_TMA_POLICY=ENABLE and sm_90+ hardware.
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

/*
 * Each CTA:
 *   1. Registers its smem with NVSHMEM via give_smem (once per kernel launch).
 *   2. Pre-fills smem (amortized; represents prior compute that produced data).
 *   3. For each iteration and each smem-sized chunk of bytes_per_block:
 *        a. fence.proxy.async.shared::cta  (make smem visible to TMA)
 *        b. nvshmemx_putmem_nbi_block      (TMA: smem -> remote gmem, NBI)
 *        c. cp.async.bulk.wait_group.read 0 (wait for smem read before reuse;
 *                                            skipped after the last chunk)
 *   4. nvshmem_quiet() + __syncthreads() to drain all in-flight remote writes.
 *
 * The elected warp-0 leader (via elect.sync + __shfl_sync) issues all
 * cp.async.bulk ops.  nvshmem_quiet() is called from all threads so the
 * elected leader drains its own pending groups regardless of which lane
 * elect.sync chose.
 */
__global__ void bw_smem_tma(char *dst, size_t bytes, int smem_size, int peer, int iter) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    size_t block_id = (size_t)bid;
    size_t nblocks_sz = (size_t)nblocks;
    size_t base_bytes_per_block = bytes / nblocks_sz;
    size_t extra_blocks = bytes % nblocks_sz;
    size_t block_offset =
        block_id * base_bytes_per_block + (block_id < extra_blocks ? block_id : extra_blocks);
    size_t bytes_per_block = base_bytes_per_block + (block_id < extra_blocks ? 1 : 0);
    char *block_dst = dst + block_offset;

    /* Register smem with NVSHMEM for TMA (once per kernel launch) */
    nvshmemx_give_smem(smem, smem_size);
    __syncthreads();

    /* Pre-fill smem (amortized; simulates prior compute filling smem) */
    size_t chunk = (size_t)smem_size;
    for (int j = tid; j < (int)(chunk / sizeof(*smem)); j += blockDim.x) smem[j] = tid;
    __syncthreads();

    size_t n_chunks = (bytes_per_block + chunk - 1) / chunk;

    for (int i = 0; i < iter; i++) {
        for (size_t c = 0; c < n_chunks; c++) {
            /* Clamp last chunk to the actual remaining bytes so the BW
             * measurement is accurate when bytes_per_block < smem_size. */
            size_t this_bytes =
                ((c + 1) * chunk <= bytes_per_block) ? chunk : (bytes_per_block - c * chunk);

            /* fence: makes smem writes visible to the TMA async proxy.
             * Required each chunk because in the real workload compute writes
             * new data to smem each iteration.  One fence per collective is
             * sufficient if smem is not modified between consecutive puts. */
#if __CUDA_ARCH__ >= 900
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif

            /* NBI put: elected leader issues cp.async.bulk from smem to
             * remote gmem; remote write proceeds asynchronously. */
            nvshmemx_putmem_nbi_block(block_dst + c * chunk, smem, this_bytes, peer);

            /* wait_group.read: waits for TMA to finish reading from smem so
             * smem can be safely overwritten for the next chunk.  The remote
             * write is still in flight.  Skipped after the last chunk. */
            if (c < n_chunks - 1) {
#if __CUDA_ARCH__ >= 900
                asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
                __syncthreads();
            }
        }

        /* Drain all in-flight remote writes.  Called from all threads so the
         * elected leader (whichever lane elect.sync chose) drains its own
         * pending cp.async.bulk groups. */
        nvshmem_quiet();
        __syncthreads();
    }
    nvshmemx_release_smem();
}

int main(int argc, char *argv[]) {
    int mype, npes;
    char *dst = NULL;

    read_args(argc, argv);
    int max_blocks = (int)num_blocks;
    int max_threads = (int)threads_per_block;

    int array_size;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw = NULL;
    float milliseconds;
    int exit_code = 0;
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

        CUDA_CHECK(cudaFuncSetAttribute(bw_smem_tma, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));

        array_size = max_size_log;
        alloc_tables(&h_tables, 2, array_size);
        h_size_arr = (uint64_t *)h_tables[0];
        h_bw = (double *)h_tables[1];

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
                bw_smem_tma<<<max_blocks, max_threads, smem_size>>>(dst, size, smem_size, peer,
                                                                    (int)warmup_iters);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                /* Timed */
                cudaEventRecord(start);
                bw_smem_tma<<<max_blocks, max_threads, smem_size>>>(dst, size, smem_size, peer,
                                                                    (int)iters);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));

                cudaEventElapsedTime(&milliseconds, start, stop);
                h_bw[i] = (double)size / (milliseconds * (B_TO_GB / ((double)iters * MS_TO_S)));
                nvshmem_barrier_all();

                i++;
            }

            print_basic_table("shmem_put_tma_smem_bw", "None", "BW", "GB/sec", '+', h_size_arr,
                              h_bw, i);
        } else {
            for (size_t size = min_size; size <= max_size; size *= step_factor)
                nvshmem_barrier_all();
        }

        /* Optional correctness check: set NVSHMEM_PERFTEST_VERIFY=1 to enable.
         * PE 0 sends one smem-sized chunk (pre-filled: element j = j % threads).
         * PE 1 checks the received buffer and reports PASS or FAIL. */
        if (getenv("NVSHMEM_PERFTEST_VERIFY")) {
            size_t verify_size = (size_t)smem_size < max_size ? (size_t)smem_size : max_size;
            verify_size -= verify_size % sizeof(int);

            if (verify_size == 0) {
                exit_code = 1;
                if (mype == 0) {
                    printf("[verify] SKIP: max_size %zu is too small for int-pattern check\n",
                           max_size);
                }
            } else {
                if (mype == 1) CUDA_CHECK(cudaMemset(dst, 0xFF, verify_size));
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_barrier_all();

                if (mype == 0) {
                    bw_smem_tma<<<1, max_threads, smem_size>>>(dst, verify_size, smem_size,
                                                               1 /*peer*/, 1 /*iter*/);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                nvshmem_barrier_all();

                if (mype == 1) {
                    int n_ints = (int)(verify_size / sizeof(int));
                    int *h_buf = (int *)malloc(verify_size);
                    if (!h_buf) {
                        fprintf(stderr, "[PE %d] malloc failed for verify buffer\n", mype);
                        exit_code = 1;
                    } else {
                        CUDA_CHECK(cudaMemcpy(h_buf, dst, verify_size, cudaMemcpyDeviceToHost));

                        int errors = 0;
                        for (int j = 0; j < n_ints; j++) {
                            int expected = j % max_threads;
                            if (h_buf[j] != expected) {
                                if (errors < 5)
                                    fprintf(stderr,
                                            "[verify] FAIL at int[%d]: got %d, expected %d\n", j,
                                            h_buf[j], expected);
                                errors++;
                            }
                        }
                        if (errors == 0)
                            printf("[verify] PASS (%zu bytes, pattern j%%threads_per_block)\n",
                                   verify_size);
                        else {
                            printf("[verify] FAIL: %d / %d ints wrong\n", errors, n_ints);
                            exit_code = 1;
                        }
                        fflush(stdout);
                        free(h_buf);
                    }
                }
                nvshmem_barrier_all();
            }
        }
    }

finalize:
    if (dst) nvshmem_free(dst);
    if (h_bw) free_tables(h_tables, 2);
    finalize_wrapper();

    return exit_code;
}
