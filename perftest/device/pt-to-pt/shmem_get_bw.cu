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

/* counter_d[0] counts CTA arrivals across all enabled barriers, while
 * counter_d[1] records the most recently released barrier epoch. */
template <bool CALL_QUIET, bool SYNC_AFTER>
__device__ __forceinline__ void inter_cta_barrier(volatile unsigned int *counter_d,
                                                  unsigned int barrier_epoch) {
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);

    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * barrier_epoch - 1)) {
            if constexpr (CALL_QUIET) nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != barrier_epoch);
    }
    if constexpr (SYNC_AFTER) __syncthreads();
}

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

/* SMEMToggle::ENABLE opts get bandwidth kernels into NVSHMEM's TMA-capable path by
 * registering dynamic shared memory at kernel entry. CFT handles are limited to
 * warp/block scope, but thread scope still uses this registration for TMA-only paths. */
template <SMEMToggle SMEM_MODE, bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER>
__global__ void bw_block(double *data_d, volatile unsigned int *counter_d, size_t len, int pe,
                         int npes, int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int barrier_epoch = 0;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = pe ^ (npes / 2);
    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false, true>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true, false>(counter_d, ++barrier_epoch);
    }
}

template <SMEMToggle SMEM_MODE, bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER>
__global__ void bw_warp(double *data_d, volatile unsigned int *counter_d, size_t len, int pe,
                        int npes, int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int barrier_epoch = 0;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nwarps_per_block = blockDim.x * blockDim.y * blockDim.z / warpSize;
    int warpid = tid / warpSize;
    size_t get_size_per_block = len / nblocks;
    size_t get_size_per_warp = get_size_per_block / nwarps_per_block;

    peer = pe ^ (npes / 2);
    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_warp(
            data_d + (bid * get_size_per_block + warpid * get_size_per_warp),
            data_d + (bid * get_size_per_block + warpid * get_size_per_warp), get_size_per_warp,
            peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false, true>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true, false>(counter_d, ++barrier_epoch);
    }
}

template <SMEMToggle SMEM_MODE, bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER>
__global__ void bw_thread(double *data_d, volatile unsigned int *counter_d, size_t len, int pe,
                          int npes, int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    smem_registration_guard<SMEM_MODE> smem_guard(nvshmem_smem, smem_size);
    int i, peer;
    unsigned int barrier_epoch = 0;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t get_size_per_block = len / nblocks;
    size_t get_size_per_thread = get_size_per_block / nthreads_per_block;

    peer = pe ^ (npes / 2);
    for (i = 0; i < iter; i++) {
        nvshmem_double_get_nbi(data_d + (bid * get_size_per_block + tid * get_size_per_thread),
                               data_d + (bid * get_size_per_block + tid * get_size_per_thread),
                               get_size_per_thread, peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false, true>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true, false>(counter_d, ++barrier_epoch);
    }
}

typedef void (*bw_fn_t)(double *data_d, volatile unsigned int *counter_d, size_t len, int pe,
                        int npes, int iter, int smem_size);

static SMEMToggle parse_smem_enabled() {
    return use_smem ? SMEMToggle::ENABLE : SMEMToggle::DISABLE;
}

template <SMEMToggle SMEM_MODE, bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER>
static bool configure_bw_variant(bw_fn_t *bw_fn, int *smem_size) {
    *smem_size = 0;

    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            *bw_fn = bw_thread<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>;
            DEBUG_PRINT(
                "Using thread-scope get (smem=%d, iteration_barrier=%d, "
                "final_barrier=%d)\n",
                (int)(SMEM_MODE == SMEMToggle::ENABLE), (int)USE_ITERATION_BARRIER,
                (int)USE_FINAL_BARRIER);
            if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
                *smem_size = NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED;
                CUDA_CHECK(cudaFuncSetAttribute(
                    bw_thread<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
            }
            break;
        case NVSHMEM_WARP:
            *bw_fn = bw_warp<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>;
            DEBUG_PRINT("Using warp-scope get (smem=%d, iteration_barrier=%d, final_barrier=%d)\n",
                        (int)(SMEM_MODE == SMEMToggle::ENABLE), (int)USE_ITERATION_BARRIER,
                        (int)USE_FINAL_BARRIER);
            if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
                *smem_size = NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED;
                CUDA_CHECK(cudaFuncSetAttribute(
                    bw_warp<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
            }
            break;
        case NVSHMEM_BLOCK:
        case NVSHMEM_ALL_SCOPES:
            *bw_fn = bw_block<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>;
            DEBUG_PRINT("Using block-scope get (smem=%d, iteration_barrier=%d, final_barrier=%d)\n",
                        (int)(SMEM_MODE == SMEMToggle::ENABLE), (int)USE_ITERATION_BARRIER,
                        (int)USE_FINAL_BARRIER);
            if constexpr (SMEM_MODE == SMEMToggle::ENABLE) {
                *smem_size = NVSHMEM_PERF_SMEM_SIZE_RECOMMENDED;
                CUDA_CHECK(cudaFuncSetAttribute(
                    bw_block<SMEM_MODE, USE_ITERATION_BARRIER, USE_FINAL_BARRIER>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, *smem_size));
            }
            break;
        default:
            fprintf(stderr, "Invalid threadgroup scope: %s\n", threadgroup_scope.name.c_str());
            return false;
    }

    return true;
}

template <SMEMToggle SMEM_MODE>
static bool configure_bw_mode(bw_fn_t *bw_fn, int *smem_size) {
    if (use_iteration_barrier) {
        if (use_final_barrier) {
            return configure_bw_variant<SMEM_MODE, true, true>(bw_fn, smem_size);
        }
        return configure_bw_variant<SMEM_MODE, true, false>(bw_fn, smem_size);
    }

    if (use_final_barrier) {
        return configure_bw_variant<SMEM_MODE, false, true>(bw_fn, smem_size);
    }
    return configure_bw_variant<SMEM_MODE, false, false>(bw_fn, smem_size);
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL;
    unsigned int *counter_d;
    double *d_bw_local = NULL, *d_bw_all = NULL;

    read_args(argc, argv);
    int max_blocks = num_blocks, max_threads = threads_per_block;

    int array_size, i;
    void **h_tables = NULL;
    uint64_t *h_size_arr;
    double *h_bw = NULL;

    bw_fn_t bw_fn = NULL;
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

    d_bw_local = (double *)nvshmem_malloc(sizeof(double));
    d_bw_all = (double *)nvshmem_malloc(npes * sizeof(double));

    /* Print table header on PE 0 */
    if (mype == 0) {
        std::fprintf(stdout, "\nshmem_get_bw (GB/s)\n");
        std::fprintf(stdout, "%14s", "size (B)");
        for (int s = 0; s < npes / 2; s++)
            std::fprintf(stdout, "  PE %d <- PE %d", s, s ^ (npes / 2));
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
    }

    {
        std::vector<double> h_bw_all(npes, 0.0);

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

        const int is_sender = (mype < npes / 2);

        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            h_size_arr[i] = size;
            size_t len = size / sizeof(double);
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

                /* Gather all BW values to PE 0 */
                CUDA_CHECK(
                    cudaMemcpy(d_bw_local, &h_bw[i], sizeof(double), cudaMemcpyHostToDevice));
                nvshmem_double_put(d_bw_all + mype, d_bw_local, 1, 0);
                nvshmem_barrier_all();

                if (mype == 0) {
                    CUDA_CHECK(cudaMemcpy(h_bw_all.data(), d_bw_all, npes * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    std::fprintf(stdout, "%14lu", (unsigned long)size);
                    for (int s = 0; s < npes / 2; s++) std::fprintf(stdout, "%14.2f", h_bw_all[s]);
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
