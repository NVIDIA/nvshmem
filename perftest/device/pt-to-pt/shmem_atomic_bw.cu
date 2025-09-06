/*
 * Copyright (c) 2021, NVIDIA CORPORATION   All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto   Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#define CUMODULE_NAME "shmem_atomic_bw.cubin"

#include "atomic_bw_common.h"

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

DEFINE_ATOMIC_BW_FN_NO_ARG(inc);
DEFINE_ATOMIC_BW_FN_NO_ARG(fetch_inc);

DEFINE_ATOMIC_BW_FN_ONE_ARG(add, 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_add, 1);

DEFINE_ATOMIC_BW_FN_ONE_ARG(and, (*(data_d + idx) << (i + 1)));
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_and, (*(data_d + idx) << (i + 1)));

DEFINE_ATOMIC_BW_FN_ONE_ARG(or, (*(data_d + idx) << i));
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_or, (*(data_d + idx) << i));

DEFINE_ATOMIC_BW_FN_ONE_ARG(xor, 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_xor, 1);

DEFINE_ATOMIC_BW_FN_ONE_ARG(swap, i + 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(set, i + 1);

DEFINE_ATOMIC_BW_FN_TWO_ARG(compare_swap, i, i + 1);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    int mype, npes;
    int size;
    int nelems;
    uint64_t *data_d = NULL;
    uint64_t set_value;
    unsigned int *counter_d;
    read_args(argc, argv);

    int max_blocks = num_blocks, max_threads = threads_per_block;
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;
    char perf_table_name[30];

    int iter = iters;
    int skip = warmup_iters;

    float milliseconds;
    cudaEvent_t start, stop;

    void *args_skip[] = {(void *)&data_d, (void *)&counter_d, (void *)&(nelems), (void *)&mype,
                         (void *)&skip};
    void *args_iter[] = {(void *)&data_d, (void *)&counter_d, (void *)&(nelems), (void *)&mype,
                         (void *)&iter};

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes   \n");
        goto finalize;
    }

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    if (use_mmap) {
        data_d = (uint64_t *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (uint64_t *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    strncpy(perf_table_name, ("shmem_atomic_" + test_amo.name).c_str(), 30);

    i = 0;
    if (mype == 0) {
        for (size = min_size; size <= max_size; size *= step_factor) {
            int blocks = max_blocks, threads = max_threads;
            nelems = size / sizeof(uint64_t);
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            /* Do warmup round for NIC cache. */
            switch (test_amo.type) {
                case AMO_INC: {
                    CALL_ATOMIC_BW_KERNEL(inc, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_SET: {
                    CALL_ATOMIC_BW_KERNEL(set, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_ADD: {
                    CALL_ATOMIC_BW_KERNEL(add, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_AND: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    CALL_ATOMIC_BW_KERNEL(and, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_OR: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    CALL_ATOMIC_BW_KERNEL(or, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_XOR: {
                    set_value = 1;
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    CALL_ATOMIC_BW_KERNEL(xor, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_FETCH_INC: {
                    CALL_ATOMIC_BW_KERNEL(fetch_inc, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                case AMO_FETCH_ADD: {
                    CALL_ATOMIC_BW_KERNEL(fetch_add, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                case AMO_FETCH_AND: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    CALL_ATOMIC_BW_KERNEL(fetch_and, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                case AMO_FETCH_OR: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    CALL_ATOMIC_BW_KERNEL(fetch_or, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                case AMO_FETCH_XOR: {
                    for (size_t j = 0; j < nelems; j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    CALL_ATOMIC_BW_KERNEL(fetch_xor, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                case AMO_SWAP: {
                    CALL_ATOMIC_BW_KERNEL(swap, blocks, threads, data_d, counter_d, nelems, mype,
                                          skip, args_skip)
                    break;
                }
                case AMO_COMPARE_SWAP: {
                    CALL_ATOMIC_BW_KERNEL(compare_swap, blocks, threads, data_d, counter_d, nelems,
                                          mype, skip, args_skip)
                    break;
                }
                default: {
                    /* Should be unreachable */
                    fprintf(stderr, "Error, unsupported Atomic op %d.\n", test_amo.type);
                    goto finalize;
                }
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            /* reset values in code. */
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            switch (test_amo.type) {
                case AMO_AND: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    break;
                }
                case AMO_OR: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    break;
                }
                case AMO_XOR: {
                    set_value = 1;
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    break;
                }
                case AMO_FETCH_AND: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    break;
                }
                case AMO_FETCH_OR: {
                    if (use_egm) {
                        memset(data_d, 0xFF, size);
                    } else {
                        CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    }
                    break;
                }
                case AMO_FETCH_XOR: {
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    break;
                }
                default: { break; }
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            cudaEventRecord(start);
            switch (test_amo.type) {
                case AMO_INC: {
                    CALL_ATOMIC_BW_KERNEL(inc, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_SET: {
                    CALL_ATOMIC_BW_KERNEL(set, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_ADD: {
                    CALL_ATOMIC_BW_KERNEL(add, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_AND: {
                    CALL_ATOMIC_BW_KERNEL(and, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_OR: {
                    CALL_ATOMIC_BW_KERNEL(or, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_XOR: {
                    CALL_ATOMIC_BW_KERNEL(xor, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_FETCH_INC: {
                    CALL_ATOMIC_BW_KERNEL(fetch_inc, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                case AMO_FETCH_ADD: {
                    CALL_ATOMIC_BW_KERNEL(fetch_add, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                case AMO_FETCH_AND: {
                    CALL_ATOMIC_BW_KERNEL(fetch_and, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                case AMO_FETCH_OR: {
                    CALL_ATOMIC_BW_KERNEL(fetch_or, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                case AMO_FETCH_XOR: {
                    CALL_ATOMIC_BW_KERNEL(fetch_xor, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                case AMO_SWAP: {
                    CALL_ATOMIC_BW_KERNEL(swap, blocks, threads, data_d, counter_d, nelems, mype,
                                          iter, args_iter)
                    break;
                }
                case AMO_COMPARE_SWAP: {
                    CALL_ATOMIC_BW_KERNEL(compare_swap, blocks, threads, data_d, counter_d, nelems,
                                          mype, iter, args_iter)
                    break;
                }
                default: {
                    /* Should be unreachable */
                    fprintf(stderr, "Error, unsupported Atomic op %d.\n", test_amo.type);
                    goto finalize;
                }
            }
            cudaEventRecord(stop);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);

            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size = min_size; size <= max_size; size *= step_factor) {
            nvshmem_barrier_all();
            nvshmem_barrier_all();
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_basic(perf_table_name, "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                          h_bw, i);
    }

finalize:

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
