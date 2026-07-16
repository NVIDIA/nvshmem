/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

enum class SMEMToggle { DISABLE, ENABLE };

template <SMEMToggle SMEM_MODE>
class smem_registration_guard {
   public:
    __device__ smem_registration_guard(void *smem, int smem_size) {
        if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
            nvshmemx_give_smem(smem, smem_size);
            __syncthreads();
        } else {
            (void)smem;
            (void)smem_size;
        }
    }

    __device__ ~smem_registration_guard() {
        if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
            __syncthreads();
            nvshmemx_release_smem();
        }
    }
};

/* SMEMToggle::ENABLE opts this benchmark in to NVSHMEM's TMA path by registering
 * shared memory at kernel entry and releasing it at kernel exit (and requires
 * a matching dynamic smem allocation at launch).  SMEMToggle::DISABLE runs the
 * original benchmark with no smem involvement; TMA stays off even if
 * NVSHMEM_TMA_POLICY=ENABLE/FORCE, because the dispatch is gated on
 * give_smem registration.  Selected at runtime via --use_smem (default: 1). */
/* These kernels are launched with 1D grids and blocks.  Keep generic index
 * flattening if that changes. */
template <SMEMToggle SMEM_MODE>
__global__ void bw_block(double *data_d, volatile unsigned int *counter_d, int len, int pe,
                         int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int counter;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_put_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1);
        }
        __syncthreads();
    }

    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1);
        nvshmem_quiet();
    }
    __syncthreads();
}

template <SMEMToggle SMEM_MODE>
__global__ void bw_warp(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter,
                        int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int counter;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nwarps_per_block = blockDim.x * blockDim.y * blockDim.z / warpSize;
    int warpid = tid / warpSize;
    size_t put_size_per_block = len / nblocks;
    size_t put_size_per_warp = put_size_per_block / nwarps_per_block;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_put_nbi_warp(
            data_d + (bid * put_size_per_block + warpid * put_size_per_warp),
            data_d + (bid * put_size_per_block + warpid * put_size_per_warp), put_size_per_warp,
            peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1);
        }
        __syncthreads();
    }

    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1);
        nvshmem_quiet();
    }
    __syncthreads();
}

template <SMEMToggle SMEM_MODE>
__global__ void bw_thread(double *data_d, volatile unsigned int *counter_d, int len, int pe,
                          int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int counter;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads_per_block = blockDim.x;
    size_t put_size_per_block = len / nblocks;
    size_t put_size_per_thread = put_size_per_block / nthreads_per_block;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmem_double_put_nbi(data_d + (bid * put_size_per_block + tid * put_size_per_thread),
                               data_d + (bid * put_size_per_block + tid * put_size_per_thread),
                               put_size_per_thread, peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1);
        }
        __syncthreads();
    }

    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1);
        nvshmem_quiet();
    }
    __syncthreads();
}

typedef void (*bw_fn_t)(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter,
                        int smem_size);

static SMEMToggle parse_smem_enabled() {
    return use_smem ? SMEMToggle::ENABLE : SMEMToggle::DISABLE;
}

template <SMEMToggle SMEM_MODE>
static bool configure_bw_mode(bw_fn_t *bw_fn, int *smem_size) {
    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            *bw_fn = bw_thread<SMEM_MODE>;
            DEBUG_PRINT("Using thread-scope put (smem=%d)\n",
                        (int)(SMEM_MODE == SMEMToggle::ENABLE));
            break;
        case NVSHMEM_WARP:
            *bw_fn = bw_warp<SMEM_MODE>;
            DEBUG_PRINT("Using warp-scope put (smem=%d)\n", (int)(SMEM_MODE == SMEMToggle::ENABLE));
            break;
        case NVSHMEM_BLOCK:
        case NVSHMEM_ALL_SCOPES:
            *bw_fn = bw_block<SMEM_MODE>;
            DEBUG_PRINT("Using block-scope put (smem=%d)\n",
                        (int)(SMEM_MODE == SMEMToggle::ENABLE));
            break;
        default:
            fprintf(stderr, "Invalid threadgroup scope: %s\n", threadgroup_scope.name.c_str());
            return false;
    }

    if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
        /* If smem opt-in: request dynamic smem of size RECOMMENDED so the kernel
         * can call give_smem and the put API can take the TMA path. */
        *smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        CUDA_CHECK(cudaFuncSetAttribute(bw_block<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(bw_warp<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(bw_thread<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
    } else {
        /* If opted out: launch with smem_size=0 and leave extern __shared__ unused. */
        *smem_size = 0;
    }

    return true;
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL;
    unsigned int *counter_d;

    read_args(argc, argv);
    int max_blocks = num_blocks, max_threads = threads_per_block;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw = NULL, *h_bw_total = NULL;
    perf_stats_t *h_bw_stats = NULL;
    double *d_bw = NULL, *d_bw_sum = NULL;

    bw_fn_t bw_fn = NULL;
    /* Opt this benchmark into NVSHMEM's TMA path by registering smem at kernel
     * boundaries.  Controlled by --use_smem so users can compare TMA-staged
     * vs baseline P2P stores without rebuilding or flipping NVSHMEM_TMA_POLICY. */
    const SMEMToggle smem_mode = parse_smem_enabled();
    int smem_size = 0;
    int iter = iters;
    int skip = warmup_iters;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    switch (smem_mode) {
        case SMEMToggle::ENABLE:
            if (!configure_bw_mode<SMEMToggle::ENABLE>(&bw_fn, &smem_size)) goto finalize;
            break;
        case SMEMToggle::DISABLE:
            if (!configure_bw_mode<SMEMToggle::DISABLE>(&bw_fn, &smem_size)) goto finalize;
            break;
    }

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];
    h_bw_stats = (perf_stats_t *)calloc(array_size, sizeof(perf_stats_t));
    if (!h_bw_stats) goto finalize;

    if (bidirectional) {
        h_bw_total = (double *)malloc(sizeof(double) * array_size);

        if (!h_bw_total) {
            fprintf(stderr, "Error: Unable to malloc on the host.\n");
            exit(1);
        }

        memset(h_bw_total, 0, sizeof(double) * array_size);

        d_bw = (double *)nvshmem_malloc(sizeof(double));
        d_bw_sum = (double *)nvshmem_malloc(sizeof(double));
    }

    if (use_mmap) {
        data_d = (double *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (double *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (bidirectional || mype == 0) {
        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            bw_fn<<<max_blocks, max_threads, smem_size>>>(data_d, counter_d, size / sizeof(double),
                                                          mype, skip, smem_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
                cudaEventRecord(start);
                bw_fn<<<max_blocks, max_threads, smem_size>>>(
                    data_d, counter_d, size / sizeof(double), mype, iter, smem_size);
                cudaEventRecord(stop);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventSynchronize(stop));
                cudaEventElapsedTime(&milliseconds, start, stop);
                h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
                nvshmem_barrier_all();
                if (bidirectional) {
                    CUDA_CHECK(cudaMemcpy(d_bw, &h_bw[i], sizeof(double), cudaMemcpyDefault));
                    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, d_bw_sum, d_bw, 1);
                    CUDA_CHECK(
                        cudaMemcpy(&h_bw_total[i], d_bw_sum, sizeof(double), cudaMemcpyDefault));
                }
                if (!mype) perf_stats_add(h_bw_stats[i], bidirectional ? h_bw_total[i] : h_bw[i]);
            }

            i++;
        }
    } else {
        for (int size = min_size; size <= max_size; size *= step_factor) {
            for (size_t repetition = 0; repetition < repetitions; repetition++)
                nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        double *p_h_bw_tmp = bidirectional ? h_bw_total : h_bw;
        const char *const test_name = bidirectional ? "shmem_put_bw_bidi" : "shmem_put_bw_uni";
        print_basic_table(test_name, "None", "BW", "GB/sec", '+', h_size_arr, p_h_bw_tmp, i,
                          h_bw_stats);
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }

    if (d_bw) nvshmem_free(d_bw);
    free(h_bw_stats);
    if (d_bw_sum) nvshmem_free(d_bw_sum);

    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
