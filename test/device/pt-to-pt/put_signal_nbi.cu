/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "put_signal_nbi.cubin"
#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include <string.h>
#include "utils.h"

__device__ int errors_d;

#define MSG_SZ 10

#define TEST_NVSHMEM_PUT_CUBIN(SC_SUFFIX)                                                    \
    void *args_put_##SC_SUFFIX[] = {(void *)&target, (void *)&source, (void *)&sig_addr,     \
                                    (void *)&me, (void *)&npes};                             \
    CUfunction test_put_cubin_##SC_SUFFIX;                                                   \
    init_test_case_kernel(&test_put_cubin_##SC_SUFFIX,                                       \
                          NVSHMEMI_TEST_STRINGIFY(test_put_signal_nbi##SC_SUFFIX##_kernel)); \
    CU_CHECK(cuLaunchKernel(test_put_cubin_##SC_SUFFIX, 1, 1, 1, num_threads, 1, 1, 0, 0,    \
                            args_put_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define TEST_PUT_SIGNAL_NBI_SCOPE_KERNEL(SCOPE, SC_SUFFIX, SC_PREFIX)                             \
    __global__ void test_put_signal_nbi##SC_SUFFIX##_kernel(                                      \
        long *target, long *source, uint64_t *sig_addr, int me, int npes) {                       \
        int i;                                                                                    \
        int myIdx = nvshmtest_thread_id_in_##SCOPE();                                             \
        int groupSize = nvshmtest_##SCOPE##_size();                                               \
                                                                                                  \
        for (i = myIdx; i < MSG_SZ; i += groupSize) source[i] = i;                                \
                                                                                                  \
        if (me == 0) {                                                                            \
            for (i = 0; i < npes; i++) {                                                          \
                nvshmem##SC_PREFIX##_long_put_signal_nbi##SC_SUFFIX(                              \
                    target, source, MSG_SZ, sig_addr, 1, NVSHMEM_SIGNAL_SET, i);                  \
            }                                                                                     \
        }                                                                                         \
        if (!myIdx) nvshmem_uint64_wait_until(sig_addr, NVSHMEM_CMP_EQ, 1);                       \
        nvshmtest_##SCOPE##_sync();                                                               \
                                                                                                  \
        if (!myIdx) {                                                                             \
            for (i = 0; i < MSG_SZ; i++) {                                                        \
                if (target[i] != source[i]) {                                                     \
                    printf(                                                                       \
                        "%10d: target[%d] = %ld not matching %ld for "                            \
                        "test_put_signal_nbi" #SC_SUFFIX " with NVSHMEM_SIGNAL_SET\n",            \
                        me, i, target[i], source[i]);                                             \
                    errors_d++;                                                                   \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        nvshmtest_##SCOPE##_sync();                                                               \
                                                                                                  \
        nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();                                            \
                                                                                                  \
        for (i = myIdx; i < MSG_SZ; i += groupSize) target[i] = 0;                                \
                                                                                                  \
        *sig_addr = 0;                                                                            \
                                                                                                  \
        nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();                                            \
        if (me == 0) {                                                                            \
            for (i = 0; i < npes; i++) {                                                          \
                nvshmem##SC_PREFIX##_long_put_signal_nbi##SC_SUFFIX(                              \
                    target, source, MSG_SZ, sig_addr, i + 1, NVSHMEM_SIGNAL_ADD, i);              \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        if (!myIdx) nvshmem_uint64_wait_until(sig_addr, NVSHMEM_CMP_EQ, me + 1);                  \
        nvshmtest_##SCOPE##_sync();                                                               \
                                                                                                  \
        if (!myIdx) {                                                                             \
            for (i = myIdx; i < MSG_SZ; i += groupSize) {                                         \
                if (target[i] != source[i]) {                                                     \
                    printf(                                                                       \
                        "%10d: target[%d] = %ld not matching %ld for "                            \
                        "test_put_signal_nbi##SC_SUFFIX with NVSHMEM_SIGNAL_ADD\n",               \
                        me, i, target[i], source[i]);                                             \
                    errors_d++;                                                                   \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        nvshmtest_##SCOPE##_sync();                                                               \
        /* Test putmem_signal_nbi */                                                              \
        nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();                                            \
                                                                                                  \
        for (i = myIdx; i < MSG_SZ; i += groupSize) target[i] = 0;                                \
                                                                                                  \
        *sig_addr = 0;                                                                            \
                                                                                                  \
        nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();                                            \
        if (me == 0) {                                                                            \
            for (i = 0; i < npes; i++) {                                                          \
                nvshmem##SC_PREFIX##_putmem_signal_nbi##SC_SUFFIX(                                \
                    (void *)target, (const void *)source, MSG_SZ * sizeof(long), sig_addr, i + 1, \
                    NVSHMEM_SIGNAL_ADD, i);                                                       \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        if (!myIdx) nvshmem_uint64_wait_until(sig_addr, NVSHMEM_CMP_EQ, me + 1);                  \
        nvshmtest_##SCOPE##_sync();                                                               \
                                                                                                  \
        if (!myIdx) {                                                                             \
            for (i = myIdx; i < MSG_SZ; i += groupSize) {                                         \
                if (target[i] != source[i]) {                                                     \
                    printf(                                                                       \
                        "%10d: target[%d] = %ld not matching %ld for "                            \
                        "test_putmem_signal_nbi##SC_SUFFIX with NVSHMEM_SIGNAL_ADD\n",            \
                        me, i, target[i], source[i]);                                             \
                    errors_d++;                                                                   \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

TEST_PUT_SIGNAL_NBI_SCOPE_KERNEL(thread, , )
TEST_PUT_SIGNAL_NBI_SCOPE_KERNEL(warp, _warp, x)
TEST_PUT_SIGNAL_NBI_SCOPE_KERNEL(block, _block, x)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    long *source;
    long *target;
    int me, npes;
    int errors = 0;
    int num_threads;
    uint64_t *sig_addr;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    me = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    cudaMemcpyToSymbol(errors_d, &errors, sizeof(int), 0);
    if (use_mmap) {
        sig_addr = (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t), _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        sig_addr = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    }
    if (!sig_addr) {
        fprintf(stderr, "failed to allocate sig_addr pointer\n");
        return -1;
    }

    if (use_mmap) {
        source = (long *)allocate_mmap_buffer(sizeof(long) * MSG_SZ, _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        source = (long *)nvshmem_malloc(sizeof(long) * MSG_SZ);
    }
    if (!source) {
        fprintf(stderr, "Failed to allocate source pointer\n");
        return -1;
    }

    if (use_mmap) {
        target =
            (long *)allocate_mmap_buffer(sizeof(long) * MSG_SZ, _mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        target = (long *)nvshmem_calloc(MSG_SZ, sizeof(long));
    }
    if (!target) {
        fprintf(stderr, "failed to allocate target pointer\n");
        return -1;
    }

    /* test put_signal_nbi */
    nvshmem_barrier_all();
    num_threads = 1;
    if (use_cubin) {
        TEST_NVSHMEM_PUT_CUBIN();
    } else {
        test_put_signal_nbi_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    }
    // test_put_signal_nbi_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    cudaDeviceSynchronize();

    /* test put_signal_warp */
    nvshmem_barrier_all();
    num_threads = 32;
    if (use_cubin) {
        TEST_NVSHMEM_PUT_CUBIN(_warp);
    } else {
        test_put_signal_nbi_warp_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    }
    // test_put_signal_nbi_warp_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    cudaDeviceSynchronize();

    /* test put_signal_block */
    nvshmem_barrier_all();
    num_threads = 1024;
    if (use_cubin) {
        num_threads = 256;
        TEST_NVSHMEM_PUT_CUBIN(_block);
    } else {
        test_put_signal_nbi_block_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    }
    // test_put_signal_nbi_block_kernel<<<1, num_threads>>>(target, source, sig_addr, me, npes);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&errors, errors_d, sizeof(int), 0);

    if (use_mmap) {
        free_mmap_buffer(sig_addr);
        free_mmap_buffer(source);
        free_mmap_buffer(target);
    } else {
        nvshmem_free(sig_addr);
        nvshmem_free(source);
        nvshmem_free(target);
    }
    finalize_wrapper();

    return errors;
}
