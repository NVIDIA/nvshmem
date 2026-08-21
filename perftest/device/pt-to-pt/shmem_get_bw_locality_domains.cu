/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

namespace {

constexpr int kValidationThreads = 256;
constexpr int kValidationMaxBlocks = 4096;

uint64_t validation_pattern(int source_pe, int locality_domain, size_t message_size,
                            size_t repetition) {
    uint64_t pattern = 0x9e3779b97f4a7c15ULL;
    pattern = (pattern ^ static_cast<uint64_t>(source_pe + 1)) * 0xbf58476d1ce4e5b9ULL;
    pattern = (pattern ^ static_cast<uint64_t>(locality_domain + 1)) * 0x94d049bb133111ebULL;
    pattern = (pattern ^ static_cast<uint64_t>(message_size)) * 0xbf58476d1ce4e5b9ULL;
    pattern = (pattern ^ static_cast<uint64_t>(repetition + 1)) * 0x94d049bb133111ebULL;
    return pattern | 1ULL;
}

}  // namespace

__global__ void fill_validation_pattern(uint64_t *data, size_t nelems, uint64_t pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t index = tid; index < nelems; index += stride) {
        data[index] = pattern;
    }
}

__global__ void validate_pattern(const uint64_t *data, size_t nelems, uint64_t expected,
                                 unsigned long long *error_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t index = tid; index < nelems; index += stride) {
        if (data[index] != expected) {
            atomicAdd(error_count, 1ULL);
        }
    }
}

/* counter_d[0] counts CTA arrivals across all enabled barriers, while
 * counter_d[1] records the most recently released barrier epoch. */
template <bool CALL_QUIET>
__device__ __forceinline__ void inter_cta_barrier(volatile unsigned int *counter_d,
                                                  unsigned int barrier_epoch) {
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);

    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * barrier_epoch - 1)) {
            if constexpr (CALL_QUIET) {
                nvshmem_quiet();
            }
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != barrier_epoch);
        if constexpr (CALL_QUIET) {
            nvshmem_quiet();
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void cta_quiet() {
    __syncthreads();
    if (!threadIdx.x) {
        nvshmem_quiet();
    }
    __syncthreads();
}

template <bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER, bool USE_FINAL_QUIET_ONLY>
__global__ void bw_block(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                         int iter) {
    int i;
    unsigned int barrier_epoch = 0;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_QUIET_ONLY) {
        cta_quiet();
    } else if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true>(counter_d, ++barrier_epoch);
    }
}

template <bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER, bool USE_FINAL_QUIET_ONLY>
__global__ void bw_warp(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                        int iter) {
    int i;
    unsigned int barrier_epoch = 0;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nwarps_per_block = blockDim.x * blockDim.y * blockDim.z / warpSize;
    int warpid = tid / warpSize;
    size_t get_size_per_block = len / nblocks;
    size_t get_size_per_warp = get_size_per_block / nwarps_per_block;

    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_warp(
            data_d + (bid * get_size_per_block + warpid * get_size_per_warp),
            data_d + (bid * get_size_per_block + warpid * get_size_per_warp), get_size_per_warp,
            peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_QUIET_ONLY) {
        cta_quiet();
    } else if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true>(counter_d, ++barrier_epoch);
    }
}

template <bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER, bool USE_FINAL_QUIET_ONLY>
__global__ void bw_thread(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                          int iter) {
    int i;
    unsigned int barrier_epoch = 0;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t get_size_per_block = len / nblocks;
    size_t get_size_per_thread = get_size_per_block / nthreads_per_block;

    for (i = 0; i < iter; i++) {
        nvshmem_double_get_nbi(data_d + (bid * get_size_per_block + tid * get_size_per_thread),
                               data_d + (bid * get_size_per_block + tid * get_size_per_thread),
                               get_size_per_thread, peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_QUIET_ONLY) {
        cta_quiet();
    } else if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true>(counter_d, ++barrier_epoch);
    }
}

/* TMA-enabled block-scope global-to-global get. Shared memory is registered as
 * NVSHMEM scratch space only; source and destination remain in localized global
 * allocations. NVSHMEM falls back to its non-TMA path when routing is unavailable. */
template <bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER, bool USE_FINAL_QUIET_ONLY>
__global__ void bw_block_tma(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                             int iter, int smem_size) {
    extern __shared__ char nvshmem_smem[];
    int i;
    unsigned int barrier_epoch = 0;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    nvshmemx_give_smem(nvshmem_smem, smem_size);
    __syncthreads();

    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        if constexpr (USE_ITERATION_BARRIER) {
            inter_cta_barrier<false>(counter_d, ++barrier_epoch);
        }
    }

    if constexpr (USE_FINAL_QUIET_ONLY) {
        cta_quiet();
    } else if constexpr (USE_FINAL_BARRIER) {
        inter_cta_barrier<true>(counter_d, ++barrier_epoch);
    }
    if constexpr (!USE_FINAL_BARRIER && !USE_FINAL_QUIET_ONLY) {
        __syncthreads();
    }
    nvshmemx_release_smem();
}

typedef void (*bw_fn_t)(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                        int iter);
typedef void (*bw_tma_fn_t)(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                            int iter, int smem_size);

template <bool USE_ITERATION_BARRIER, bool USE_FINAL_BARRIER, bool USE_FINAL_QUIET_ONLY>
static bool configure_bw_variant(bw_fn_t *bw_fn, bw_tma_fn_t *bw_tma_fn) {
    *bw_tma_fn = bw_block_tma<USE_ITERATION_BARRIER, USE_FINAL_BARRIER, USE_FINAL_QUIET_ONLY>;

    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            *bw_fn = bw_thread<USE_ITERATION_BARRIER, USE_FINAL_BARRIER, USE_FINAL_QUIET_ONLY>;
            DEBUG_PRINT("Using thread-scope get (iteration_barrier=%d, final_barrier=%d)\n",
                        (int)USE_ITERATION_BARRIER, (int)USE_FINAL_BARRIER);
            break;
        case NVSHMEM_WARP:
            *bw_fn = bw_warp<USE_ITERATION_BARRIER, USE_FINAL_BARRIER, USE_FINAL_QUIET_ONLY>;
            DEBUG_PRINT("Using warp-scope get (iteration_barrier=%d, final_barrier=%d)\n",
                        (int)USE_ITERATION_BARRIER, (int)USE_FINAL_BARRIER);
            break;
        case NVSHMEM_BLOCK:
        case NVSHMEM_ALL_SCOPES:
            *bw_fn = bw_block<USE_ITERATION_BARRIER, USE_FINAL_BARRIER, USE_FINAL_QUIET_ONLY>;
            DEBUG_PRINT("Using block-scope get (iteration_barrier=%d, final_barrier=%d)\n",
                        (int)USE_ITERATION_BARRIER, (int)USE_FINAL_BARRIER);
            break;
        default:
            fprintf(stderr, "Invalid threadgroup scope: %s\n", threadgroup_scope.name.c_str());
            return false;
    }

    return true;
}

static bool configure_bw_mode(bw_fn_t *bw_fn, bw_tma_fn_t *bw_tma_fn) {
    if (use_final_quiet_only) {
        return configure_bw_variant<false, false, true>(bw_fn, bw_tma_fn);
    }
    if (use_iteration_barrier) {
        if (use_final_barrier) {
            return configure_bw_variant<true, true, false>(bw_fn, bw_tma_fn);
        }
        return configure_bw_variant<true, false, false>(bw_fn, bw_tma_fn);
    }

    if (use_final_barrier) {
        return configure_bw_variant<false, true, false>(bw_fn, bw_tma_fn);
    }
    return configure_bw_variant<false, false, false>(bw_fn, bw_tma_fn);
}

int main(int argc, char *argv[]) {
    int mype, npes;
    int num_locality_domains = 0;
    int exit_status = 1;

    /* Per-memory-node state */
    std::vector<double *> data_d;
    std::vector<void *> buf_addrs;
    std::vector<CUmemGenericAllocationHandle> alloc_handles;
    std::vector<unsigned int *> counter_d_arr;
    std::vector<CUgreenCtx> green_ctxs;
    std::vector<CUstream> gc_streams;
    size_t alloc_size = 0;

    /* Timing: per-domain completion events live in green contexts (sync only).
       The common start and per-domain stop events live in the primary context
       for cuEventElapsedTime. */
    CUstream timing_stream = nullptr;
    CUevent timing_start = nullptr;
    std::vector<CUstream> timing_streams;
    std::vector<CUevent> timing_stop_events;
    std::vector<CUevent> gc_done_events;

    /* Bandwidth gathering buffers */
    double *d_bw_local = NULL, *d_bw_all = NULL;
    int *d_validation_status = NULL;
    unsigned long long *d_validation_errors = NULL;
    int return_code = 0;

    read_args(argc, argv);
    int max_threads = threads_per_block;

    int array_size, i;
    void **h_tables = NULL;
    uint64_t *h_size_arr;
    double *h_bw = NULL;

    std::vector<std::vector<perf_stats_t>> bw_stats_per_pair_per_size;
    std::vector<perf_stats_t> bw_avg_stats_per_size;

    bw_fn_t bw_fn = NULL;
    bw_tma_fn_t bw_tma_fn = NULL;
    bool use_tma = false;
    int smem_size = 0;
    int min_partition_sms = 0;
    int max_partition_sms = 0;
    int blocks_per_domain = 0;
    int iter = iters;
    int skip = warmup_iters;

    float milliseconds;

    /* Reset all per-node inter-block counters to zero. */
    auto reset_counters = [&]() {
        for (auto *ptr : counter_d_arr) {
            if (ptr) {
                CUDA_CHECK(cudaMemset(ptr, 0, sizeof(unsigned int) * 2));
            }
        }
    };

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes < 2 || (npes & (npes - 1)) != 0) {
        fprintf(stderr, "This test requires a power-of-two number of processes (>= 2)\n");
        return_code = 1;
        goto finalize;
    }

    /* ------------------------------------------------------------------ */
    /* Query device info and locality domains                              */
    /* ------------------------------------------------------------------ */
    {
        int dev;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::array<char, 40> uuid_str{};
        const auto &bytes = prop.uuid.bytes;
        std::snprintf(uuid_str.data(), uuid_str.size(),
                      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      static_cast<unsigned char>(bytes[0]), static_cast<unsigned char>(bytes[1]),
                      static_cast<unsigned char>(bytes[2]), static_cast<unsigned char>(bytes[3]),
                      static_cast<unsigned char>(bytes[4]), static_cast<unsigned char>(bytes[5]),
                      static_cast<unsigned char>(bytes[6]), static_cast<unsigned char>(bytes[7]),
                      static_cast<unsigned char>(bytes[8]), static_cast<unsigned char>(bytes[9]),
                      static_cast<unsigned char>(bytes[10]), static_cast<unsigned char>(bytes[11]),
                      static_cast<unsigned char>(bytes[12]), static_cast<unsigned char>(bytes[13]),
                      static_cast<unsigned char>(bytes[14]), static_cast<unsigned char>(bytes[15]));

        CUdevice cu_dev;
        CU_CHECK(cuDeviceGet(&cu_dev, dev));
        CU_CHECK(cuDeviceGetAttribute(&num_locality_domains,
                                      CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_dev));

        int peer = mype ^ (npes / 2);
        std::fprintf(stdout, "PE %d: GPU %d, UUID: GPU-%s, peer: %d, locality_domains: %d\n", mype,
                     dev, uuid_str.data(), peer, num_locality_domains);
        std::fflush(stdout);
    }

    if (num_locality_domains < 2) {
        fprintf(stderr, "PE %d: need >= 2 locality domains for this test, found %d\n", mype,
                num_locality_domains);
        return_code = 1;
        goto finalize;
    }
    if (min_size < num_locality_domains * sizeof(double)) {
        fprintf(stderr, "PE %d: min_size needs to be at least %zu for this test, found %zu\n", mype,
                (size_t)num_locality_domains * sizeof(double), min_size);
        return_code = 1;
        goto finalize;
    }

    /* ------------------------------------------------------------------ */
    /* Create green contexts and streams (one per locality domain)         */
    /* ------------------------------------------------------------------ */
    {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));

        CUdevResource smResource;
        CU_CHECK(cuDeviceGetDevResource(dev, &smResource, CU_DEV_RESOURCE_TYPE_SM));

        std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(num_locality_domains);
        for (int n = 0; n < num_locality_domains; n++) {
            params[n].smCount = 0;
            params[n].coscheduledSmCount = 2;
            params[n].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
            params[n].localityDomainId = n;
        }

        std::vector<CUdevResource> nodeSmResources(num_locality_domains);
        CUdevResource remainder;
        CU_CHECK(cuDevSmResourceSplit(nodeSmResources.data(), num_locality_domains, &smResource,
                                      &remainder, 0, params.data()));

        /* Smallest/largest SM partition across locality domains.  The inter-block
           barrier requires the whole grid to be co-resident, so the launch grid
           is later clamped to (occupancy * smallest-SMs-in-partition). */
        for (int n = 0; n < num_locality_domains; n++) {
            int sms = (int)nodeSmResources[n].sm.smCount;
            if (n == 0 || sms < min_partition_sms) {
                min_partition_sms = sms;
            }
            if (n == 0 || sms > max_partition_sms) {
                max_partition_sms = sms;
            }
        }
        if (mype == 0) {
            std::fprintf(stdout,
                         "SM partitions per locality domain: min=%d, max=%d (of %d domains)\n",
                         min_partition_sms, max_partition_sms, num_locality_domains);
            std::fflush(stdout);
        }

        green_ctxs.resize(num_locality_domains);
        gc_streams.resize(num_locality_domains);
        for (int n = 0; n < num_locality_domains; n++) {
            CUdevResourceDesc desc;
            CU_CHECK(cuDevResourceGenerateDesc(&desc, &nodeSmResources[n], 1));
            CU_CHECK(cuGreenCtxCreate(&green_ctxs[n], desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
            CU_CHECK(
                cuGreenCtxStreamCreate(&gc_streams[n], green_ctxs[n], CU_STREAM_NON_BLOCKING, 0));
        }
    }

    /* ------------------------------------------------------------------ */
    /* Timing infrastructure                                               */
    /* ------------------------------------------------------------------ */
    {
        CU_CHECK(cuStreamCreate(&timing_stream, CU_STREAM_NON_BLOCKING));
        CU_CHECK(cuEventCreate(&timing_start, CU_EVENT_DEFAULT));

        timing_streams.resize(num_locality_domains);
        timing_stop_events.resize(num_locality_domains);
        gc_done_events.resize(num_locality_domains);
        for (int n = 0; n < num_locality_domains; n++) {
            CU_CHECK(cuStreamCreate(&timing_streams[n], CU_STREAM_NON_BLOCKING));
            CU_CHECK(cuEventCreate(&timing_stop_events[n], CU_EVENT_DEFAULT));

            CUcontext gc_ctx;
            CU_CHECK(cuCtxFromGreenCtx(&gc_ctx, green_ctxs[n]));
            CU_CHECK(cuCtxPushCurrent(gc_ctx));
            CU_CHECK(cuEventCreate(&gc_done_events[n], CU_EVENT_DISABLE_TIMING));
            CU_CHECK(cuCtxPopCurrent(NULL));
        }
    }

    if (!configure_bw_mode(&bw_fn, &bw_tma_fn)) {
        return_code = 1;
        goto finalize;
    }

    /* Register scratch smem for NVSHMEM's block-scope global-to-global TMA path.
       Requires NVSHMEM_TMA_POLICY=ENABLE, sm_90+, and at least two full warps;
       other routing failures fall back to P2P loads inside NVSHMEM. */
    use_tma = use_smem && (threadgroup_scope.type == NVSHMEM_BLOCK ||
                           threadgroup_scope.type == NVSHMEM_ALL_SCOPES);
    if (use_tma) {
        if (max_threads < 64) {
            fprintf(stderr,
                    "Localized global-to-global TMA requires at least 64 threads per CTA "
                    "(requested %d)\n",
                    max_threads);
            return_code = 1;
            goto finalize;
        }
        smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        CUDA_CHECK(cudaFuncSetAttribute(bw_tma_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
        DEBUG_PRINT("Using block-scope global-to-global TMA get (smem_size=%d)\n", smem_size);
    }

    /* ------------------------------------------------------------------ */
    /* Clamp the launch grid to the co-resident capacity of one SM         */
    /* partition.  An enabled global-counter inter-block barrier deadlocks  */
    /* if the grid has more CTAs than can be simultaneously resident.  We   */
    /* cannot use a cooperative launch here                                 */
    /* (nvshmemx_collective_launch ignores the user stream and so cannot    */
    /* target the per-domain green-context streams), so we guarantee co-     */
    /* residence ourselves: max CTAs = occupancy * SMs-in-partition.        */
    /* ------------------------------------------------------------------ */
    {
        /* C/n CTAs per domain (n * blocks_per_domain == C). */
        blocks_per_domain = num_blocks / num_locality_domains;
        if (blocks_per_domain < 1) {
            blocks_per_domain = 1;
        }

        if (use_iteration_barrier || use_final_barrier) {
            int occupancy = 0;
            if (use_tma) {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occupancy, bw_tma_fn, max_threads, (size_t)smem_size));
            } else {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, bw_fn,
                                                                         max_threads, 0));
            }

            int max_coresident = occupancy * min_partition_sms;
            if (max_coresident < 1) {
                max_coresident = 1;
            }

            if (blocks_per_domain > max_coresident) {
                if (mype == 0) {
                    fprintf(stderr,
                            "WARNING: %d CTAs/domain exceeds the co-resident capacity of an SM "
                            "partition (%d blocks/SM * %d SMs = %d); clamping to %d to avoid an "
                            "inter-block-barrier deadlock.\n",
                            blocks_per_domain, occupancy, min_partition_sms, max_coresident,
                            max_coresident);
                }
                blocks_per_domain = max_coresident;
            }
            DEBUG_PRINT("Grid clamp: occupancy=%d, partition_sms=%d, blocks_per_domain=%d\n",
                        occupancy, min_partition_sms, blocks_per_domain);
        } else {
            DEBUG_PRINT(
                "Grid clamp disabled because both inter-CTA barriers are disabled "
                "(blocks_per_domain=%d)\n",
                blocks_per_domain);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Allocate localized memory per locality domain and register with NVSHMEM */
    /* ------------------------------------------------------------------ */
    {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));

        CUmemAllocationProp gran_prop = {};
        gran_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        gran_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        gran_prop.location.localized.deviceId = (unsigned char)dev;
        gran_prop.location.localized.localityDomainId = 0;
        size_t granularity;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &gran_prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        alloc_size = ((max_size + granularity - 1) / granularity) * granularity;
        alloc_size = pad_up(alloc_size);
        DEBUG_PRINT("Localized alloc granularity: %zu, alloc_size: %zu\n", granularity, alloc_size);

        data_d.resize(num_locality_domains, nullptr);
        buf_addrs.resize(num_locality_domains, nullptr);
        alloc_handles.resize(num_locality_domains);

        for (int n = 0; n < num_locality_domains; n++) {
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
            prop.location.localized.deviceId = (unsigned char)dev;
            prop.location.localized.localityDomainId = (unsigned char)n;
            prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);

            CU_CHECK(cuMemCreate(&alloc_handles[n], alloc_size, &prop, 0));
            CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&buf_addrs[n], alloc_size, 0, 0, 0));
            CU_CHECK(cuMemMap((CUdeviceptr)buf_addrs[n], alloc_size, 0, alloc_handles[n], 0));

            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
            access.location.localized.deviceId = (unsigned char)dev;
            access.location.localized.localityDomainId = (unsigned char)n;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess((CUdeviceptr)buf_addrs[n], alloc_size, &access, 1));

            data_d[n] = (double *)nvshmemx_buffer_register_symmetric(buf_addrs[n], alloc_size, 0);
            if (!data_d[n]) {
                fprintf(stderr,
                        "PE %d: nvshmemx_buffer_register_symmetric failed "
                        "for locality domain %d\n",
                        mype, n);
                return_code = 1;
                goto finalize;
            }
            CUDA_CHECK(cudaMemset(data_d[n], 0, alloc_size));
            DEBUG_PRINT("PE %d: node %d  buf_addr=%p  sym_ptr=%p\n", mype, n, buf_addrs[n],
                        (void *)data_d[n]);
        }
    }

    counter_d_arr.resize(num_locality_domains, nullptr);
    if (use_iteration_barrier || use_final_barrier) {
        for (int n = 0; n < num_locality_domains; n++) {
            CUDA_CHECK(cudaMalloc((void **)&counter_d_arr[n], sizeof(unsigned int) * 2));
            CUDA_CHECK(cudaMemset(counter_d_arr[n], 0, sizeof(unsigned int) * 2));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    d_bw_local = (double *)nvshmem_malloc(sizeof(double));
    d_bw_all = (double *)nvshmem_malloc(npes * sizeof(double));
    d_validation_status = (int *)nvshmem_malloc(2 * sizeof(int));
    CUDA_CHECK(
        cudaMalloc(&d_validation_errors, num_locality_domains * sizeof(*d_validation_errors)));
    if (!d_bw_local || !d_bw_all || !d_validation_status) {
        fprintf(stderr, "PE %d: failed to allocate benchmark result buffers\n", mype);
        return_code = 1;
        goto finalize;
    }

    if (mype == 0) {
        bw_stats_per_pair_per_size.assign(array_size,
                                          std::vector<perf_stats_t>(std::max(1, npes / 2)));
        bw_avg_stats_per_size.resize(array_size);
    }

    /* ------------------------------------------------------------------ */
    /* Benchmark loop                                                      */
    /* ------------------------------------------------------------------ */
    {
        int peer = mype ^ (npes / 2);
        std::vector<double> h_bw_all(npes, 0.0);
        std::vector<float> elapsed_ms(num_locality_domains, 0.0f);
        std::vector<unsigned long long> validation_errors(num_locality_domains, 0);

        if (blocks_per_domain * num_locality_domains != num_blocks && mype == 0) {
            fprintf(stderr,
                    "WARNING: -c %zu not divisible by num_locality_domains %d. "
                    "Using %d blocks per domain (%d total)\n",
                    num_blocks, num_locality_domains, blocks_per_domain,
                    blocks_per_domain * num_locality_domains);
        }

        /* The TMA kernel differs only by registering NVSHMEM scratch smem; both
           paths pass localized global memory as the get source and destination. */
        auto launch_kernel = [&](int n, size_t kern_len, int kern_peer, int kern_iter) {
            if (use_tma) {
                bw_tma_fn<<<blocks_per_domain, max_threads, smem_size,
                            (cudaStream_t)gc_streams[n]>>>(data_d[n], counter_d_arr[n], kern_len,
                                                           kern_peer, kern_iter, smem_size);
            } else {
                bw_fn<<<blocks_per_domain, max_threads, 0, (cudaStream_t)gc_streams[n]>>>(
                    data_d[n], counter_d_arr[n], kern_len, kern_peer, kern_iter);
            }
        };

        bool warned_split = false;
        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            /* size/n bytes per domain (total == size). */
            size_t bytes_per_domain = size / num_locality_domains;
            size_t len = bytes_per_domain / sizeof(double);
            size_t bytes_per_block = (len / blocks_per_domain) * sizeof(double);
            size_t transferred_elements = (len / blocks_per_domain) * blocks_per_domain;
            h_size_arr[i] = size;

            if (!warned_split && mype == 0 &&
                (size % num_locality_domains != 0 || bytes_per_domain % sizeof(double) != 0 ||
                 (use_tma && bytes_per_block % 16 != 0))) {
                fprintf(stderr,
                        "WARNING: size %zu does not split cleanly across %d domains "
                        "(per-domain %zu B, per-CTA %zu B); reported BW reflects the %zu B "
                        "actually moved%s\n",
                        size, num_locality_domains, bytes_per_domain, bytes_per_block,
                        (size_t)num_locality_domains * bytes_per_domain,
                        (use_tma && bytes_per_block % 16 != 0)
                            ? " and TMA falls back to P2P (per-CTA size must be a multiple of 16B)."
                            : ".");
                warned_split = true;
            }

            if (mype < npes / 2) {
                /* warmup */
                reset_counters();
                for (int n = 0; n < num_locality_domains; n++) {
                    launch_kernel(n, len, peer, skip);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());
            }

            for (size_t repetition = 0; repetition < repetitions; repetition++) {
                int validation_blocks = static_cast<int>(std::min<size_t>(
                    kValidationMaxBlocks, (len + kValidationThreads - 1) / kValidationThreads));

                /* A get reads from the upper-half peer into the lower-half PE's
                   corresponding localized buffer.  Initialize the source with a
                   domain-specific pattern and clear the destination. */
                if (mype >= npes / 2) {
                    for (int n = 0; n < num_locality_domains; n++) {
                        fill_validation_pattern<<<validation_blocks, kValidationThreads, 0,
                                                  (cudaStream_t)gc_streams[n]>>>(
                            reinterpret_cast<uint64_t *>(data_d[n]), len,
                            validation_pattern(mype, n, size, repetition));
                    }
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    for (int n = 0; n < num_locality_domains; n++) {
                        CUDA_CHECK(cudaMemsetAsync(data_d[n], 0, bytes_per_domain,
                                                   (cudaStream_t)gc_streams[n]));
                    }
                }
                for (int n = 0; n < num_locality_domains; n++) {
                    CU_CHECK(cuStreamSynchronize(gc_streams[n]));
                }
                nvshmem_barrier_all();

                if (mype < npes / 2) {
                    /* timed run */
                    reset_counters();
                    std::fill(elapsed_ms.begin(), elapsed_ms.end(), 0.0f);

                    CU_CHECK(cuEventRecord(timing_start, timing_stream));
                    for (int n = 0; n < num_locality_domains; n++) {
                        CU_CHECK(cuStreamWaitEvent(gc_streams[n], timing_start, 0));
                        launch_kernel(n, len, peer, iter);
                        CU_CHECK(cuEventRecord(gc_done_events[n], gc_streams[n]));
                        CU_CHECK(cuStreamWaitEvent(timing_streams[n], gc_done_events[n], 0));
                        CU_CHECK(cuEventRecord(timing_stop_events[n], timing_streams[n]));
                    }
                    CUDA_CHECK(cudaGetLastError());

                    for (int n = 0; n < num_locality_domains; n++) {
                        CU_CHECK(cuEventSynchronize(timing_stop_events[n]));
                        CU_CHECK(cuEventElapsedTime(&elapsed_ms[n], timing_start,
                                                    timing_stop_events[n]));
                    }

                    milliseconds = *std::max_element(elapsed_ms.begin(), elapsed_ms.end());
                    /* Aggregate BW uses the actual bytes moved and the slowest domain time. */
                    h_bw[i] = ((double)num_locality_domains * bytes_per_domain) /
                              (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
                } else {
                    h_bw[i] = 0.0;
                }

                /* Validate only after every timed requester stream has completed. */
                nvshmem_barrier_all();
                int local_validation_failed = 0;
                std::fill(validation_errors.begin(), validation_errors.end(), 0);
                if (mype < npes / 2 && transferred_elements != 0) {
                    int transferred_validation_blocks = static_cast<int>(std::min<size_t>(
                        kValidationMaxBlocks,
                        (transferred_elements + kValidationThreads - 1) / kValidationThreads));
                    for (int n = 0; n < num_locality_domains; n++) {
                        CUDA_CHECK(cudaMemsetAsync(d_validation_errors + n, 0,
                                                   sizeof(*d_validation_errors),
                                                   (cudaStream_t)gc_streams[n]));
                        validate_pattern<<<transferred_validation_blocks, kValidationThreads, 0,
                                           (cudaStream_t)gc_streams[n]>>>(
                            reinterpret_cast<const uint64_t *>(data_d[n]), transferred_elements,
                            validation_pattern(peer, n, size, repetition), d_validation_errors + n);
                        CUDA_CHECK(cudaGetLastError());
                        CUDA_CHECK(cudaMemcpyAsync(&validation_errors[n], d_validation_errors + n,
                                                   sizeof(validation_errors[n]),
                                                   cudaMemcpyDeviceToHost,
                                                   (cudaStream_t)gc_streams[n]));
                    }
                    for (int n = 0; n < num_locality_domains; n++) {
                        CU_CHECK(cuStreamSynchronize(gc_streams[n]));
                        if (validation_errors[n] != 0) {
                            fprintf(stderr,
                                    "PE %d: get validation failed for source PE %d, domain %d, "
                                    "size %zu, repetition %zu: %llu mismatched elements\n",
                                    mype, peer, n, size, repetition, validation_errors[n]);
                            local_validation_failed = 1;
                        }
                    }
                }

                CUDA_CHECK(cudaMemcpy(d_validation_status, &local_validation_failed, sizeof(int),
                                      cudaMemcpyHostToDevice));
                nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, d_validation_status + 1,
                                       d_validation_status, 1);
                int global_validation_failed = 0;
                CUDA_CHECK(cudaMemcpy(&global_validation_failed, d_validation_status + 1,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                if (global_validation_failed) {
                    return_code = 1;
                }

                /* Gather all BW values to PE 0 */
                CUDA_CHECK(
                    cudaMemcpy(d_bw_local, &h_bw[i], sizeof(double), cudaMemcpyHostToDevice));
                nvshmem_double_put(d_bw_all + mype, d_bw_local, 1, 0);
                nvshmem_barrier_all();

                if (mype == 0) {
                    CUDA_CHECK(cudaMemcpy(h_bw_all.data(), d_bw_all, npes * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    double bw_sum = 0.0;
                    for (int s = 0; s < npes / 2; s++) {
                        perf_stats_add(bw_stats_per_pair_per_size[i][s], h_bw_all[s]);
                        bw_sum += h_bw_all[s];
                    }
                    perf_stats_add(bw_avg_stats_per_size[i], bw_sum / (npes / 2));
                }
            }

            i++;
        }
    }

    exit_status = 0;

    if (mype == 0) {
        const char *test_name =
            use_tma ? "shmem_get_bw_locality_domains_tma" : "shmem_get_bw_locality_domains";
        const int num_pairs = std::max(1, npes / 2);

        std::vector<double> bw_avg(i, 0.0);
        for (int j = 0; j < i; j++) {
            bw_avg[j] = bw_avg_stats_per_size[j].mean;
        }
        print_basic_table(test_name, "None", "BW", "GB/sec", '+', h_size_arr, bw_avg.data(), i,
                          bw_avg_stats_per_size.data());

        if (npes > 2) {
            std::vector<double> bw_pair(i, 0.0);
            std::vector<perf_stats_t> stats(i);
            for (int s = 0; s < num_pairs; s++) {
                for (int j = 0; j < i; j++) {
                    bw_pair[j] = bw_stats_per_pair_per_size[j][s].mean;
                    stats[j] = bw_stats_per_pair_per_size[j][s];
                }
                char subjob[32];
                std::snprintf(subjob, sizeof(subjob), "PE%d_from_PE%d", s, s ^ (npes / 2));
                print_basic_table(test_name, subjob, "BW", "GB/sec", '+', h_size_arr,
                                  bw_pair.data(), i, stats.data());
            }
        }
    }

finalize:

    for (size_t n = 0; n < data_d.size(); n++) {
        if (data_d[n]) {
            nvshmemx_buffer_unregister_symmetric(data_d[n], alloc_size);
        }
        if (n < buf_addrs.size() && buf_addrs[n]) {
            cuMemUnmap((CUdeviceptr)buf_addrs[n], alloc_size);
            cuMemAddressFree((CUdeviceptr)buf_addrs[n], alloc_size);
        }
        if (n < alloc_handles.size()) {
            cuMemRelease(alloc_handles[n]);
        }
    }

    for (auto *ptr : counter_d_arr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    for (auto evt : gc_done_events) {
        cuEventDestroy(evt);
    }
    for (auto evt : timing_stop_events) {
        cuEventDestroy(evt);
    }
    if (timing_start) {
        cuEventDestroy(timing_start);
    }

    for (auto stream : timing_streams) {
        cuStreamDestroy(stream);
    }
    for (auto stream : gc_streams) {
        cuStreamDestroy(stream);
    }
    for (auto ctx : green_ctxs) {
        cuGreenCtxDestroy(ctx);
    }
    if (timing_stream) {
        cuStreamDestroy(timing_stream);
    }

    if (d_bw_local) {
        nvshmem_free(d_bw_local);
    }
    if (d_bw_all) {
        nvshmem_free(d_bw_all);
    }
    if (d_validation_status) {
        nvshmem_free(d_validation_status);
    }
    if (d_validation_errors) {
        cudaFree(d_validation_errors);
    }

    if (h_tables) {
        free_tables(h_tables, 2);
    }
    finalize_wrapper();

    return exit_status != 0 ? exit_status : return_code;
}
