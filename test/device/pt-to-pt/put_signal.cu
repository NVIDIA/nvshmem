/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Validate shmem_put_signal operation through blocking API
 * It mimics the broadcast operation through a ring-based algorithm
 * using shmem_put_signal.
 */
#define CUMODULE_NAME "put_signal.cubin"
#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "nvshmem.h"
#include "nvshmemx.h"

__device__ int errors_d;

#define MSG_SZ 16

#define TEST_NVSHMEM_PUT_CUBIN(FUNC, SC_SUFFIX)                                                \
    size_t cubin_dynamic_smem_size = 0;                                                        \
    void *args_##FUNC##_##SC_SUFFIX[] = {(void *)&target,                                      \
                                         (void *)&source,                                      \
                                         (void *)&sig_addr,                                    \
                                         (void *)&me,                                          \
                                         (void *)&npes,                                        \
                                         (void *)&op,                                          \
                                         (void *)&cubin_dynamic_smem_size};                    \
    CUfunction test_##FUNC##_cubin_##SC_SUFFIX;                                                \
    init_test_case_kernel(&test_##FUNC##_cubin_##SC_SUFFIX,                                    \
                          NVSHMEMI_TEST_STRINGIFY(test_##FUNC##_signal##SC_SUFFIX##_kernel));  \
    CU_CHECK(cuLaunchKernel(test_##FUNC##_cubin_##SC_SUFFIX, 1, 1, 1, num_threads, 1, 1, 0, 0, \
                            args_##FUNC##_##SC_SUFFIX, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DEFINE_SIGNAL_PUT_SIZE(TYPE, FUNC, SCOPE, SC_SUFFIX, SC_PREFIX)                            \
    __global__ void test_##FUNC##_signal##SC_SUFFIX##_kernel(void *target, void *source,           \
                                                             uint64_t *sig_addr, int me, int npes, \
                                                             int op, size_t dynamic_smem_size) {   \
        int i;                                                                                     \
        int dest = (me + 1) % npes;                                                                \
        int myIdx = nvshmtest_thread_id_in_##SCOPE();                                              \
        int groupSize = nvshmtest_##SCOPE##_size();                                                \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                 \
                                                                                                   \
        *sig_addr = 0;                                                                             \
                                                                                                   \
        for (i = myIdx; i < MSG_SZ; i += groupSize) {                                              \
            ((TYPE *)source)[i] = i;                                                               \
            ((TYPE *)target)[i] = 0;                                                               \
        }                                                                                          \
        nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX();                                             \
                                                                                                   \
        if (me == 0) {                                                                             \
            nvshmem##SC_PREFIX##_##FUNC##_signal##SC_SUFFIX((TYPE *)target, (TYPE *)source,        \
                                                            MSG_SZ, sig_addr, 1, op, dest);        \
            if (!myIdx) nvshmem_uint64_wait_until(sig_addr, NVSHMEM_CMP_EQ, 1);                    \
        } else {                                                                                   \
            if (!myIdx) nvshmem_uint64_wait_until(sig_addr, NVSHMEM_CMP_EQ, 1);                    \
            nvshmem##SC_PREFIX##_##FUNC##_signal##SC_SUFFIX((TYPE *)target, (TYPE *)target,        \
                                                            MSG_SZ, sig_addr, 1, op, dest);        \
        }                                                                                          \
        if (!myIdx) {                                                                              \
            for (i = 0; i < MSG_SZ; i++) {                                                         \
                if (((TYPE *)target)[i] != ((TYPE *)source)[i]) {                                  \
                    printf("%10d: target[%d] = not matching op: %d for " #SCOPE " scope\n", me, i, \
                           op);                                                                    \
                    errors_d++;                                                                    \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        nvshmtest_##SCOPE##_sync();                                                                \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                              \
    }                                                                                              \
                                                                                                   \
    static int run_##FUNC##_##SC_SUFFIX##_test(int me, int npes) {                                 \
        uint64_t *sig_addr = NULL;                                                                 \
        TYPE *target, *source;                                                                     \
        int errors = 0;                                                                            \
        int num_threads;                                                                           \
                                                                                                   \
        cudaMemcpyToSymbol(errors_d, &errors, sizeof(int), 0);                                     \
        source = (TYPE *)nvshmem_calloc(MSG_SZ, sizeof(TYPE));                                     \
        if (!source) {                                                                             \
            fprintf(stderr, "Failed to allocate source pointer\n");                                \
            return -1;                                                                             \
        }                                                                                          \
                                                                                                   \
        target = (TYPE *)nvshmem_calloc(MSG_SZ, sizeof(TYPE));                                     \
        if (!target) {                                                                             \
            fprintf(stderr, "failed to allocate target pointer\n");                                \
            return -1;                                                                             \
        }                                                                                          \
                                                                                                   \
        sig_addr = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));                                \
        if (!sig_addr) {                                                                           \
            fprintf(stderr, "failed to allocate sig_addr pointer\n");                              \
            return -1;                                                                             \
        }                                                                                          \
                                                                                                   \
        if (strcmp(#SCOPE, "thread") == 0)                                                         \
            num_threads = 1;                                                                       \
        else if (strcmp(#SCOPE, "warp") == 0)                                                      \
            num_threads = 32;                                                                      \
        else                                                                                       \
            num_threads = use_cubin ? 256 : 1024;                                                  \
        nvshmem_barrier_all();                                                                     \
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_##FUNC##_signal##SC_SUFFIX##_kernel,                \
                                          _dynamic_smem_size);                                     \
        if (use_cubin) {                                                                           \
            int op = NVSHMEM_SIGNAL_SET;                                                           \
            TEST_NVSHMEM_PUT_CUBIN(FUNC, SC_SUFFIX);                                               \
        } else {                                                                                   \
            test_##FUNC##_signal##SC_SUFFIX##_kernel<<<1, num_threads, _dynamic_smem_size>>>(      \
                target, source, sig_addr, me, npes, NVSHMEM_SIGNAL_SET, _dynamic_smem_size);       \
        }                                                                                          \
        cudaDeviceSynchronize();                                                                   \
                                                                                                   \
        cudaMemcpyFromSymbol(&errors, errors_d, sizeof(int), 0);                                   \
        if (errors) {                                                                              \
            ERROR_PRINT("Failed in " #FUNC " with %d errors", errors);                             \
            goto out;                                                                              \
        }                                                                                          \
                                                                                                   \
        nvshmem_barrier_all();                                                                     \
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_##FUNC##_signal##SC_SUFFIX##_kernel,                \
                                          _dynamic_smem_size);                                     \
        if (use_cubin) {                                                                           \
            int op = NVSHMEM_SIGNAL_ADD;                                                           \
            TEST_NVSHMEM_PUT_CUBIN(FUNC, SC_SUFFIX);                                               \
        } else {                                                                                   \
            test_##FUNC##_signal##SC_SUFFIX##_kernel<<<1, num_threads, _dynamic_smem_size>>>(      \
                target, source, sig_addr, me, npes, NVSHMEM_SIGNAL_ADD, _dynamic_smem_size);       \
        }                                                                                          \
        cudaDeviceSynchronize();                                                                   \
                                                                                                   \
        cudaMemcpyFromSymbol(&errors, errors_d, sizeof(int), 0);                                   \
                                                                                                   \
        if (errors) {                                                                              \
            ERROR_PRINT("Failed in " #FUNC " with %d errors", errors);                             \
            goto out;                                                                              \
        }                                                                                          \
                                                                                                   \
    out:                                                                                           \
        nvshmem_free(source);                                                                      \
        nvshmem_free(target);                                                                      \
        nvshmem_free(sig_addr);                                                                    \
                                                                                                   \
        return errors;                                                                             \
    }

DEFINE_SIGNAL_PUT_SIZE(long, long_put, thread, , )
DEFINE_SIGNAL_PUT_SIZE(long, long_put, warp, _warp, x)
DEFINE_SIGNAL_PUT_SIZE(long, long_put, block, _block, x)
DEFINE_SIGNAL_PUT_SIZE(int8_t, put8, warp, _warp, x)
DEFINE_SIGNAL_PUT_SIZE(int32_t, put32, block, _block, x)
DEFINE_SIGNAL_PUT_SIZE(int64_t, put64, thread, , )
DEFINE_SIGNAL_PUT_SIZE(char, putmem, block, _block, x)
#undef DEFINE_SIGNAL_PUT_SIZE

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char *argv[]) {
    int me, npes;
    int errors;

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }
    me = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    /*TODO Add an int4 test.*/
    errors = run_long_put__test(me, npes);
    errors += run_long_put__warp_test(me, npes);
    errors += run_long_put__block_test(me, npes);
    errors += run_put8__warp_test(me, npes);
    errors += run_put32__block_test(me, npes);
    errors += run_put64__test(me, npes);
    errors += run_putmem__block_test(me, npes);

    finalize_wrapper();
    return errors;
}
