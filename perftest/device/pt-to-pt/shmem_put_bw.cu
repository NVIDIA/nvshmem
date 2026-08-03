/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdio>
#include <cassert>
#include <vector>
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
                         int npes, int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int counter;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = pe ^ (npes / 2);
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
__global__ void bw_warp(double *data_d, volatile unsigned int *counter_d, int len, int pe, int npes,
                        int iter, int smem_size) {
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

    peer = pe ^ (npes / 2);
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
                          int npes, int iter, int smem_size) {
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

    peer = pe ^ (npes / 2);
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

typedef void (*bw_fn_t)(double *data_d, volatile unsigned int *counter_d, int len, int pe, int npes,
                        int iter, int smem_size);

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
        *smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        CUDA_CHECK(cudaFuncSetAttribute(bw_block<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(bw_warp<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(bw_thread<SMEM_MODE>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
    } else {
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
    void **h_tables = NULL;
    uint64_t *h_size_arr;
    double *h_bw = NULL;
    /* Per-PE BW gather: each PE stages its h_bw[i] in d_bw_local and writes
     * it into d_bw_all[mype] on PE 0 every iteration. PE 0 then prints one
     * row per size with N/2 columns (one per sender pair). */
    double *d_bw_local = NULL, *d_bw_all = NULL;

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

    if (npes < 2 || (npes & (npes - 1)) != 0) {
        fprintf(stderr, "This test requires a power-of-two number of processes (>= 2)\n");
        goto finalize;
    }

    print_device_uuid_and_peer(mype, mype ^ (npes / 2));

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

    d_bw_local = (double *)nvshmem_malloc(sizeof(double));
    d_bw_all = (double *)nvshmem_malloc(npes * sizeof(double));

    /* Per-pair BW table header (PE 0 only). */
    if (mype == 0) {
        const char *test_name = bidirectional ? "shmem_put_bw_bidi" : "shmem_put_bw_uni";
        std::fprintf(stdout, "\n%s (GB/s)\n", test_name);
        std::fprintf(stdout, "%14s", "size (B)");
        for (int s = 0; s < npes / 2; s++)
            std::fprintf(stdout, "  PE %d -> PE %d", s, s ^ (npes / 2));
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
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

    {
        std::array<cudaLaunchAttribute, 1> user_attrs{};
        int n_user_attrs = 0;
#if CUDART_VERSION >= 13000
        if (use_nucs) {
            user_attrs[n_user_attrs].id = cudaLaunchAttributeNvlinkUtilCentricScheduling;
            user_attrs[n_user_attrs].val.nvlinkUtilCentricScheduling = 1;
            ++n_user_attrs;
        }
#endif

        nvshmemx_collective_launch_attr_t cl_attr{};
        cl_attr.cuda_config.gridDim = dim3(max_blocks);
        cl_attr.cuda_config.blockDim = dim3(max_threads);
        cl_attr.cuda_config.dynamicSmemBytes = smem_size;
        cl_attr.cuda_config.stream = 0;
        cl_attr.cuda_config.attrs = (n_user_attrs > 0) ? user_attrs.data() : nullptr;
        cl_attr.cuda_config.numAttrs = n_user_attrs;

        const int is_sender = (bidirectional || mype < npes / 2);

        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            h_size_arr[i] = size;
            int len = static_cast<int>(size / sizeof(double));
            int iter_warmup = is_sender ? skip : 0;
            int iter_timed = is_sender ? iter : 0;
            int status;

            /* warmup */
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            {
                void *args[] = {&data_d, &counter_d, &len, &mype, &npes, &iter_warmup, &smem_size};
                status = nvshmemx_collective_launch_attr(&cl_attr, (const void *)bw_fn, args);
                if (status != 0) {
                    fprintf(stderr, "nvshmemx_collective_launch_attr (warmup) failed: %d\n",
                            status);
                    goto finalize;
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                /* timed run */
                CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
                {
                    void *args[] = {&data_d, &counter_d,  &len,      &mype,
                                    &npes,   &iter_timed, &smem_size};
                    if (is_sender) cudaEventRecord(start);
                    status = nvshmemx_collective_launch_attr(&cl_attr, (const void *)bw_fn, args);
                    if (is_sender) cudaEventRecord(stop);
                    if (status != 0) {
                        fprintf(stderr, "nvshmemx_collective_launch_attr (timed) failed: %d\n",
                                status);
                        goto finalize;
                    }
                }

                if (is_sender) {
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
                } else {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    h_bw[i] = 0.0;
                }
                nvshmem_barrier_all();

                /* Gather every PE's h_bw[i] onto PE 0 and emit the row.
                 * In bidirectional mode each pair contributes two flows
                 * (sender → peer and peer → sender), so the pair-column
                 * value is the sum of the two; otherwise only the
                 * sender's BW is meaningful (receivers stored 0.0). */
                CUDA_CHECK(cudaMemcpy(d_bw_local, &h_bw[i], sizeof(double), cudaMemcpyDefault));
                nvshmem_double_put(d_bw_all + mype, d_bw_local, 1, 0);
                nvshmem_barrier_all();

                if (mype == 0) {
                    std::vector<double> h_bw_all(npes, 0.0);
                    CUDA_CHECK(cudaMemcpy(h_bw_all.data(), d_bw_all, npes * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    std::fprintf(stdout, "%14lu", (unsigned long)h_size_arr[i]);
                    for (int s = 0; s < npes / 2; s++) {
                        double bw =
                            bidirectional ? (h_bw_all[s] + h_bw_all[s + npes / 2]) : h_bw_all[s];
                        std::fprintf(stdout, "%14.2f", bw);
                    }
                    std::fprintf(stdout, "\n");
                    std::fflush(stdout);
                }
            }

            i++;
        }
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }

    if (d_bw_local) nvshmem_free(d_bw_local);
    if (d_bw_all) nvshmem_free(d_bw_all);

    if (h_tables) free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
