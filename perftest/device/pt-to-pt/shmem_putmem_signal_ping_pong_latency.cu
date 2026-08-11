/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>
#include "utils.h"

static constexpr unsigned char kPayloadByte = 0x5a;
static constexpr unsigned char kDestinationSentinel = 0xa5;

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void ping_pong(int *data_d, uint64_t *flag_d, uint64_t *ack_d, int len, int pe,
                          int iter) {
    extern __shared__ __align__(16) unsigned char smem[];
    nvshmemx_give_smem(smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    int peer = !pe;
    size_t bytes = static_cast<size_t>(len) * sizeof(*data_d);

    for (int i = 0; i < iter; i++) {
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (!pe) {
            nvshmemx_putmem_signal_nbi_block(data_d, data_d, bytes, flag_d, expected,
                                             NVSHMEM_SIGNAL_SET, peer);
            if (threadIdx.x == 0) {
                nvshmem_uint64_wait_until(ack_d, NVSHMEM_CMP_EQ, expected);
            }
            __syncthreads();
        } else {
            if (threadIdx.x == 0) {
                nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, expected);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                nvshmemx_signal_op(ack_d, expected, NVSHMEM_SIGNAL_SET, peer);
            }
            __syncthreads();
        }
    }
    nvshmemx_release_smem();
}

void test_ping_pong(void **arglist, cudaStream_t stream) {
    int status = nvshmemx_collective_launch((const void *)ping_pong, 1, threads_per_block, arglist,
                                            nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED), stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaGetLastError());
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

static void initialize_case(int *data, uint64_t *flag, uint64_t *ack, int *status, size_t bytes,
                            int mype) {
    unsigned char value = mype == 0 ? kPayloadByte : kDestinationSentinel;
    if (use_egm) {
        memset(data, value, bytes);
        memset(flag, 0, sizeof(*flag));
    } else {
        CUDA_CHECK(cudaMemset(data, value, bytes));
        CUDA_CHECK(cudaMemset(flag, 0, sizeof(*flag)));
    }
    CUDA_CHECK(cudaMemset(ack, 0, sizeof(*ack)));
    CUDA_CHECK(cudaMemset(status, 0, 2 * sizeof(*status)));
}

static int validate_case(int *data, uint64_t *flag, uint64_t *ack, int *status, size_t bytes,
                         int iterations, int mype) {
    int local_errors = 0;
    if (mype == 1) {
        std::vector<unsigned char> received(bytes);
        if (use_egm) {
            memcpy(received.data(), data, bytes);
        } else {
            CUDA_CHECK(cudaMemcpy(received.data(), data, bytes, cudaMemcpyDeviceToHost));
        }
        for (unsigned char value : received) {
            local_errors += value != kPayloadByte;
        }
        uint64_t observed = 0;
        CUDA_CHECK(cudaMemcpy(&observed, flag, sizeof(observed), cudaMemcpyDeviceToHost));
        local_errors += observed != static_cast<uint64_t>(iterations);
    } else {
        uint64_t observed = 0;
        CUDA_CHECK(cudaMemcpy(&observed, ack, sizeof(observed), cudaMemcpyDeviceToHost));
        local_errors += observed != static_cast<uint64_t>(iterations);
    }

    CUDA_CHECK(cudaMemcpy(&status[0], &local_errors, sizeof(local_errors), cudaMemcpyHostToDevice));
    int rc = nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, &status[1], &status[0], 1);
    if (rc != NVSHMEMX_SUCCESS) {
        return rc;
    }
    int global_errors = 0;
    CUDA_CHECK(
        cudaMemcpy(&global_errors, &status[1], sizeof(global_errors), cudaMemcpyDeviceToHost));
    if (mype == 0 && global_errors != 0) {
        fprintf(stderr, "putmem-signal payload validation failed for %zu bytes (%d errors)\n",
                bytes, global_errors);
    }
    return global_errors;
}

int main(int argc, char *argv[]) {
    int mype, npes;
    uint64_t *flag_d = NULL;
    uint64_t *ack_d = NULL;
    int *data_d = NULL;
    int *status_d = NULL;
    int exit_code = 0;
    cudaStream_t stream;

    read_args(argc, argv);
    int iter = iters;
    int skip = warmup_iters;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;
    perf_stats_t *h_lat_stats = NULL;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        fprintf(stderr, "Putmem-signal ping-pong does not support cubin mode\n");
        finalize_wrapper();
        return EXIT_FAILURE;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        flag_d = (uint64_t *)allocate_mmap_buffer(sizeof(*flag_d), mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (int *)nvshmem_malloc(max_size);
        flag_d = (uint64_t *)nvshmem_malloc(sizeof(*flag_d));
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, mype == 0 ? kPayloadByte : kDestinationSentinel, max_size));
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(*flag_d)));
    }
    status_d = (int *)nvshmem_malloc(2 * sizeof(*status_d));
    ack_d = (uint64_t *)nvshmem_malloc(sizeof(*ack_d));
    CUDA_CHECK(cudaMemset(ack_d, 0, sizeof(*ack_d)));
    CUDA_CHECK(cudaMemset(status_d, 0, 2 * sizeof(*status_d)));

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];
    h_lat_stats = (perf_stats_t *)calloc(array_size, sizeof(perf_stats_t));
    if (!h_lat_stats) {
        fprintf(stderr, "Failed to allocate latency statistics\n");
        nvshmem_global_exit(EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFuncSetAttribute(ping_pong, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED)));

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    i = 0;
    for (size_t size = min_size; size <= max_size; size *= step_factor) {
        if (size == 0 || (size & 15) != 0) {
            continue;
        }
        int nelems = size / sizeof(int);
        h_size_arr[i] = size;
        void *args_1[] = {&data_d, &flag_d, &ack_d, &nelems, &mype, &skip};
        void *args_2[] = {&data_d, &flag_d, &ack_d, &nelems, &mype, &iter};
        initialize_case(data_d, flag_d, ack_d, status_d, size, mype);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        test_ping_pong(args_1, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (size_t repetition = 0; repetition < repetitions; repetition++) {
            initialize_case(data_d, flag_d, ack_d, status_d, size, mype);
            nvshmem_barrier_all();

            cudaEventRecord(start, stream);
            test_ping_pong(args_2, stream);
            cudaEventRecord(stop, stream);

            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);
            if (validate_case(data_d, flag_d, ack_d, status_d, size, iter, mype) != 0) {
                exit_code = 1;
                goto finalize;
            }
            h_lat[i] = (milliseconds * 1000) / iter;
            if (mype == 0) {
                perf_stats_add(h_lat_stats[i], h_lat[i]);
            }
            nvshmem_barrier_all();
        }
        i++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        print_basic_table("shmem_putmem_signal_ping_lat", "None", "latency", "us", '-', h_size_arr,
                          h_lat, i, h_lat_stats);
    }
finalize:

    if (status_d) {
        nvshmem_free(status_d);
    }
    if (ack_d) {
        nvshmem_free(ack_d);
    }
    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    if (flag_d) {
        if (use_mmap) {
            free_mmap_buffer(flag_d);
        } else {
            nvshmem_free(flag_d);
        }
    }
    free(h_lat_stats);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return exit_code;
}
