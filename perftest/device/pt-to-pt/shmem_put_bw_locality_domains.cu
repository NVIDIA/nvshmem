/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

__global__ void bw_block(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                         int iter) {
    int i;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

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

__global__ void bw_warp(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                        int iter) {
    int i;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nwarps_per_block = blockDim.x * blockDim.y * blockDim.z / warpSize;
    int warpid = tid / warpSize;
    size_t put_size_per_block = len / nblocks;
    size_t put_size_per_warp = put_size_per_block / nwarps_per_block;

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

__global__ void bw_thread(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                          int iter) {
    int i;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t put_size_per_block = len / nblocks;
    size_t put_size_per_thread = put_size_per_block / nthreads_per_block;

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

typedef void (*bw_fn_t)(double *data_d, volatile unsigned int *counter_d, size_t len, int peer,
                        int iter);

int main(int argc, char *argv[]) {
    int mype, npes;
    int num_locality_domains = 0;

    /* Per-memory-node state */
    std::vector<double *> data_d;
    std::vector<void *> buf_addrs;
    std::vector<CUmemGenericAllocationHandle> alloc_handles;
    std::vector<unsigned int *> counter_d_arr;
    std::vector<CUgreenCtx> green_ctxs;
    std::vector<CUstream> gc_streams;
    size_t alloc_size = 0;

    /* Timing: per-node completion events live in green contexts (sync only).
       The common start and per-node stop events live in the primary context
       for cuEventElapsedTime. */
    CUstream timing_stream = nullptr;
    CUevent timing_start = nullptr;
    std::vector<CUstream> timing_streams;
    std::vector<CUevent> timing_stop_events;
    std::vector<CUevent> gc_done_events;

    /* Bandwidth gathering buffers */
    double *d_bw_local = NULL, *d_bw_all = NULL;

    read_args(argc, argv);
    int max_threads = threads_per_block;

    int array_size, i;
    void **h_tables = NULL;
    uint64_t *h_size_arr;
    double *h_bw = NULL;

    bw_fn_t bw_fn = bw_block;
    int min_partition_sms = 0;
    int max_partition_sms = 0;
    int blocks_per_domain = 0;
    int iter = iters;
    int skip = warmup_iters;

    float milliseconds;

    /* Reset all per-node inter-block counters to zero. */
    auto reset_counters = [&]() {
        for (auto *ptr : counter_d_arr) CUDA_CHECK(cudaMemset(ptr, 0, sizeof(unsigned int) * 2));
    };

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes < 2 || (npes & (npes - 1)) != 0) {
        fprintf(stderr, "This test requires a power-of-two number of processes (>= 2)\n");
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
        goto finalize;
    }
    if (min_size < num_locality_domains * sizeof(double)) {
        fprintf(stderr, "PE %d: min_size needs to be at least %zu for this test, found %zu\n", mype,
                (size_t)num_locality_domains * sizeof(double), min_size);
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
            if (n == 0 || sms < min_partition_sms) min_partition_sms = sms;
            if (n == 0 || sms > max_partition_sms) max_partition_sms = sms;
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

    switch (threadgroup_scope.type) {
        case NVSHMEM_THREAD:
            bw_fn = bw_thread;
            DEBUG_PRINT("Using thread-scope put\n");
            break;
        case NVSHMEM_WARP:
            bw_fn = bw_warp;
            DEBUG_PRINT("Using warp-scope put\n");
            break;
        case NVSHMEM_BLOCK:
        case NVSHMEM_ALL_SCOPES:
            bw_fn = bw_block;
            DEBUG_PRINT("Using block-scope put\n");
            break;
        default:
            fprintf(stderr, "Invalid threadgroup scope: %s\n", threadgroup_scope.name.c_str());
            goto finalize;
    }

    /* ------------------------------------------------------------------ */
    /* Clamp the launch grid to the co-resident capacity of one SM         */
    /* partition.  The kernels use a global-counter inter-block barrier     */
    /* every iteration, which deadlocks if the grid has more CTAs than can  */
    /* be simultaneously resident.  We cannot use a cooperative launch here */
    /* (nvshmemx_collective_launch ignores the user stream and so cannot    */
    /* target the per-domain green-context streams), so we guarantee co-     */
    /* residence ourselves: max CTAs = occupancy * SMs-in-partition.        */
    /* ------------------------------------------------------------------ */
    {
        /* C/n CTAs per domain (n * blocks_per_domain == C). */
        blocks_per_domain = num_blocks / num_locality_domains;
        if (blocks_per_domain < 1) blocks_per_domain = 1;

        int occupancy = 0;
        CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, bw_fn, max_threads, 0));

        int max_coresident = occupancy * min_partition_sms;
        if (max_coresident < 1) max_coresident = 1;

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
        DEBUG_PRINT("Grid clamp: occupancy=%d, partition_sms=%d, blocks_per_domain=%d\n", occupancy,
                    min_partition_sms, blocks_per_domain);
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
                goto finalize;
            }
            CUDA_CHECK(cudaMemset(data_d[n], 0, alloc_size));
            DEBUG_PRINT("PE %d: node %d  buf_addr=%p  sym_ptr=%p\n", mype, n, buf_addrs[n],
                        (void *)data_d[n]);
        }
    }

    counter_d_arr.resize(num_locality_domains, nullptr);
    for (int n = 0; n < num_locality_domains; n++) {
        CUDA_CHECK(cudaMalloc((void **)&counter_d_arr[n], sizeof(unsigned int) * 2));
        CUDA_CHECK(cudaMemset(counter_d_arr[n], 0, sizeof(unsigned int) * 2));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    d_bw_local = (double *)nvshmem_malloc(sizeof(double));
    d_bw_all = (double *)nvshmem_malloc(npes * sizeof(double));

    /* Print table header on PE 0 */
    if (mype == 0) {
        std::fprintf(stdout, "\nshmem_put_bw_locality_domains (GB/s)\n");
        std::fprintf(stdout, "%14s", "size (B)");
        for (int s = 0; s < npes / 2; s++)
            std::fprintf(stdout, "  PE %d -> PE %d", s, s ^ (npes / 2));
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
    }

    /* ------------------------------------------------------------------ */
    /* Benchmark loop                                                      */
    /* ------------------------------------------------------------------ */
    {
        int peer = mype ^ (npes / 2);
        std::vector<double> h_bw_all(npes, 0.0);
        std::vector<float> elapsed_ms(num_locality_domains, 0.0f);

        if (blocks_per_domain * num_locality_domains != num_blocks && mype == 0) {
            fprintf(stderr,
                    "WARNING: -c %d not divisible by num_locality_domains %d. "
                    "Using %d blocks per domain (%d total)\n",
                    num_blocks, num_locality_domains, blocks_per_domain,
                    blocks_per_domain * num_locality_domains);
        }

        auto launch_kernel = [&](int n, size_t kern_len, int kern_peer, int kern_iter) {
            bw_fn<<<blocks_per_domain, max_threads, 0, (cudaStream_t)gc_streams[n]>>>(
                data_d[n], counter_d_arr[n], kern_len, kern_peer, kern_iter);
        };

        bool warned_split = false;
        i = 0;
        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            /* size/n bytes per domain (total == size). */
            size_t bytes_per_domain = size / num_locality_domains;
            size_t len = bytes_per_domain / sizeof(double);
            size_t bytes_per_block = (len / blocks_per_domain) * sizeof(double);
            h_size_arr[i] = size;

            if (!warned_split && mype == 0 &&
                (size % num_locality_domains != 0 || bytes_per_domain % sizeof(double) != 0)) {
                fprintf(stderr,
                        "WARNING: size %zu does not split cleanly across %d domains "
                        "(per-domain %zu B, per-CTA %zu B); reported BW reflects the %zu B "
                        "actually moved.\n",
                        size, num_locality_domains, bytes_per_domain, bytes_per_block,
                        (size_t)num_locality_domains * bytes_per_domain);
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
            }

            for (size_t repetition = 0; repetition < repetitions; repetition++) {
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

    /* Clean up localized buffers */
    for (size_t n = 0; n < data_d.size(); n++) {
        if (data_d[n]) nvshmemx_buffer_unregister_symmetric(data_d[n], alloc_size);
        if (n < buf_addrs.size() && buf_addrs[n]) {
            cuMemUnmap((CUdeviceptr)buf_addrs[n], alloc_size);
            cuMemAddressFree((CUdeviceptr)buf_addrs[n], alloc_size);
        }
        if (n < alloc_handles.size()) cuMemRelease(alloc_handles[n]);
    }

    for (auto *ptr : counter_d_arr)
        if (ptr) cudaFree(ptr);

    for (auto evt : gc_done_events) cuEventDestroy(evt);
    for (auto evt : timing_stop_events) cuEventDestroy(evt);
    if (timing_start) cuEventDestroy(timing_start);

    for (auto stream : timing_streams) cuStreamDestroy(stream);
    for (auto stream : gc_streams) cuStreamDestroy(stream);
    for (auto ctx : green_ctxs) cuGreenCtxDestroy(ctx);
    if (timing_stream) cuStreamDestroy(timing_stream);

    if (d_bw_local) nvshmem_free(d_bw_local);
    if (d_bw_all) nvshmem_free(d_bw_all);

    if (h_tables) free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
