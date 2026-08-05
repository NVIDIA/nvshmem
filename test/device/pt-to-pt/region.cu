/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

constexpr int NUM_BLOCKS = 2;
constexpr int THREADS_PER_BLOCK = 64;
constexpr int ELEMS_PER_BLOCK = 64;
constexpr int ELEMS_PER_KERNEL = NUM_BLOCKS * ELEMS_PER_BLOCK;
constexpr int NUM_KERNELS = 2;
constexpr int TOTAL_ELEMS = NUM_KERNELS * ELEMS_PER_KERNEL;
constexpr int WARP_CHUNK = 31;
constexpr int PATTERN_SCALE = 100000;
constexpr int SCOPE_TEST_ELEMS = THREADS_PER_BLOCK;

__device__ int flat_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ int flat_block_size() { return blockDim.x * blockDim.y * blockDim.z; }

__device__ void record_error(int *status, int code) { atomicCAS(status, 0, code); }

__global__ void region_block_umbrella_kernel(int *source, int *recv, int *get_recv, int *status,
                                             int base, int mype, int npes) {
    int tid = flat_thread_id();
    int block_base = base + blockIdx.x * ELEMS_PER_BLOCK;
    int peer = (mype + 1) % npes;

    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;
    nvshmemx_region_handle_t region = 0;
    int rc = nvshmemx_region_start_block(&region, &attrs);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 1);
    }

    int active = 0;
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, &active);
    if (rc != NVSHMEMX_SUCCESS || active != 1) {
        record_error(status, 2);
    }
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_BATCH_RMA, &active);
    if (rc != NVSHMEMX_SUCCESS || active != 1) {
        record_error(status, 5);
    }

    if (tid == 0) {
        nvshmem_putmem_nbi(recv + block_base, source + block_base, sizeof(int), peer);
    }

    if (tid < warpSize) {
        nvshmemx_putmem_nbi_warp(recv + block_base + 1, source + block_base + 1,
                                 WARP_CHUNK * sizeof(int), peer);
    }

    nvshmemx_putmem_nbi_block(recv + block_base + 1 + WARP_CHUNK,
                              source + block_base + 1 + WARP_CHUNK,
                              (ELEMS_PER_BLOCK - 1 - WARP_CHUNK) * sizeof(int), peer);

    if (tid == 0) {
        nvshmem_getmem_nbi(get_recv + block_base, source + block_base, sizeof(int), peer);
    }

    if (tid < warpSize) {
        nvshmemx_getmem_nbi_warp(get_recv + block_base + 1, source + block_base + 1,
                                 WARP_CHUNK * sizeof(int), peer);
    }

    nvshmemx_getmem_nbi_block(get_recv + block_base + 1 + WARP_CHUNK,
                              source + block_base + 1 + WARP_CHUNK,
                              (ELEMS_PER_BLOCK - 1 - WARP_CHUNK) * sizeof(int), peer);

    if (tid == 0) {
        if (blockIdx.x == 0) {
            nvshmem_fence();
        } else {
            nvshmem_quiet();
        }
    }
    __syncthreads();
    rc = nvshmemx_region_stop_block(region);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 3);
    }

    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, &active);
    if (rc != NVSHMEMX_SUCCESS || active != 0) {
        record_error(status, 4);
    }

    nvshmem_quiet();
}

__global__ void region_invalid_nested_kernel(int *status) {
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;
    nvshmemx_region_handle_t block_region = 0;
    nvshmemx_region_handle_t nested_region = 0;

    int rc = nvshmemx_region_start_block(&block_region, &attrs);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 10);
    }

    rc = nvshmemx_region_start_block(&nested_region, &attrs);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 11);
    }

    rc = nvshmemx_region_stop_block(block_region);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 12);
    }
}

__global__ void region_no_hint_kernel(int *source, int *recv, int *get_recv, int *status, int mype,
                                      int npes) {
    int tid = flat_thread_id();
    int peer = (mype + 1) % npes;
    nvshmemx_region_handle_t region = 0;

    int rc = nvshmemx_region_start_block(&region, NULL);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 20);
    }

    int active = 0;
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, &active);
    if (rc != NVSHMEMX_SUCCESS || active != 1) {
        record_error(status, 21);
    }
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_BATCH_RMA, &active);
    if (rc != NVSHMEMX_SUCCESS || active != 0) {
        record_error(status, 22);
    }

    nvshmem_putmem_nbi(recv + tid, source + tid, sizeof(int), peer);
    nvshmem_getmem_nbi(get_recv + tid, source + tid, sizeof(int), peer);

    rc = nvshmemx_region_stop_block(region + 1);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 23);
    }
    rc = nvshmemx_region_stop_block(region);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 24);
    }
    nvshmem_quiet();
}

__global__ void region_invalid_hint_kernel(int *status) {
    constexpr uint32_t unknown_hint = 1u << 31;
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = unknown_hint;
    nvshmemx_region_handle_t region = 0;
    int active = 0;

    int rc = nvshmemx_region_start_block(&region, &attrs);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 30);
    }
    rc = nvshmemx_region_start_block(NULL, NULL);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 31);
    }
    rc = nvshmemx_region_is_active(unknown_hint, &active);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 32);
    }
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, NULL);
    if (rc != NVSHMEMX_ERROR_INVALID_VALUE) {
        record_error(status, 33);
    }
}

__global__ void region_block_stop_only_kernel(int *source, int *recv, int *get_recv, int *status,
                                              int mype, int npes) {
    int peer = (mype + 1) % npes;
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;
    nvshmemx_region_handle_t region = 0;

    int rc = nvshmemx_region_start_block(&region, &attrs);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 40);
    }

    nvshmemx_putmem_nbi_block(recv, source, SCOPE_TEST_ELEMS * sizeof(int), peer);
    nvshmemx_getmem_nbi_block(get_recv, source, SCOPE_TEST_ELEMS * sizeof(int), peer);

    rc = nvshmemx_region_stop_block(region);
    if (rc != NVSHMEMX_SUCCESS) {
        record_error(status, 41);
    }
    nvshmem_quiet();
}

static int check_device_status(int *status_d, const char *name) {
    int status = 0;
    CUDA_CHECK(cudaMemcpy(&status, status_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (status != 0) {
        printf("[PE %d] FAIL: %s device status %d\n", nvshmem_my_pe(), name, status);
        return 1;
    }
    return 0;
}

static int verify_recv_data(const char *name, int *recv, int expected_pe, int nelems) {
    std::vector<int> host(nelems);
    CUDA_CHECK(cudaMemcpy(host.data(), recv, sizeof(int) * nelems, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
        int expected = expected_pe * PATTERN_SCALE + i;
        if (host[i] != expected) {
            printf("[PE %d] FAIL: %s[%d] = %d, expected %d\n", nvshmem_my_pe(), name, i, host[i],
                   expected);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int status = 0;

    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int prev_pe = (mype - 1 + npes) % npes;
    int next_pe = (mype + 1) % npes;
    std::vector<int> source_data(TOTAL_ELEMS);

    if (npes < 2) {
        printf("[PE %d] SKIP: region device test requires at least 2 PEs\n", mype);
        finalize_wrapper();
        return EXIT_SUCCESS;
    }

    int *source = static_cast<int *>(nvshmem_malloc(sizeof(int) * TOTAL_ELEMS));
    int *recv = static_cast<int *>(nvshmem_malloc(sizeof(int) * TOTAL_ELEMS));
    int *get_recv = static_cast<int *>(nvshmem_malloc(sizeof(int) * TOTAL_ELEMS));
    int *status_d = NULL;
    cudaStream_t streams[NUM_KERNELS] = {NULL, NULL};
    if (source == NULL || recv == NULL || get_recv == NULL) {
        printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
        status = 1;
        goto out;
    }

    CUDA_CHECK(cudaMalloc(&status_d, sizeof(int)));
    for (int i = 0; i < TOTAL_ELEMS; i++) {
        source_data[i] = mype * PATTERN_SCALE + i;
    }
    CUDA_CHECK(
        cudaMemcpy(source, source_data.data(), sizeof(int) * TOTAL_ELEMS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(status_d, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * TOTAL_ELEMS));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * TOTAL_ELEMS));

    nvshmem_barrier_all();
    region_invalid_nested_kernel<<<1, THREADS_PER_BLOCK>>>(status_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    status |= check_device_status(status_d, "nested-region rejection");

    CUDA_CHECK(cudaMemset(status_d, 0, sizeof(int)));
    region_invalid_hint_kernel<<<1, THREADS_PER_BLOCK>>>(status_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    status |= check_device_status(status_d, "invalid region arguments");
    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking));

    CUDA_CHECK(cudaMemset(status_d, 0, sizeof(int)));
    nvshmem_barrier_all();
    region_block_umbrella_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[0]>>>(
        source, recv, get_recv, status_d, 0, mype, npes);
    region_block_umbrella_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[1]>>>(
        source, recv, get_recv, status_d, ELEMS_PER_KERNEL, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    nvshmem_barrier_all();

    status |= check_device_status(status_d, "block umbrella");
    status |= verify_recv_data("recv", recv, prev_pe, TOTAL_ELEMS);
    status |= verify_recv_data("get_recv", get_recv, next_pe, TOTAL_ELEMS);

    CUDA_CHECK(cudaMemset(status_d, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * SCOPE_TEST_ELEMS));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * SCOPE_TEST_ELEMS));
    nvshmem_barrier_all();
    region_no_hint_kernel<<<1, THREADS_PER_BLOCK>>>(source, recv, get_recv, status_d, mype, npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    status |= check_device_status(status_d, "region without batching hint");
    status |= verify_recv_data("no-hint recv", recv, prev_pe, SCOPE_TEST_ELEMS);
    status |= verify_recv_data("no-hint get_recv", get_recv, next_pe, SCOPE_TEST_ELEMS);

    CUDA_CHECK(cudaMemset(status_d, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * SCOPE_TEST_ELEMS));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * SCOPE_TEST_ELEMS));
    nvshmem_barrier_all();
    region_block_stop_only_kernel<<<1, THREADS_PER_BLOCK>>>(source, recv, get_recv, status_d, mype,
                                                            npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    status |= check_device_status(status_d, "block stop-only");
    status |= verify_recv_data("block recv", recv, prev_pe, SCOPE_TEST_ELEMS);
    status |= verify_recv_data("block get_recv", get_recv, next_pe, SCOPE_TEST_ELEMS);

    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

out:
    if (status_d) {
        CUDA_CHECK(cudaFree(status_d));
    }
    if (source) {
        nvshmem_free(source);
    }
    if (recv) {
        nvshmem_free(recv);
    }
    if (get_recv) {
        nvshmem_free(get_recv);
    }
    finalize_wrapper();

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
