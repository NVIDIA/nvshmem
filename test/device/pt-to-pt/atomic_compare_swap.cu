/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "atomic_compare_swap.cubin"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <assert.h>

#include "utils.h"

__device__ int error_d;

enum op { ATOMIC_COMPARE_SWAP = 0 };

#define REPT_MACRO_FOR_TYPES(MACRO_NAME, OP)       \
    MACRO_NAME(OP, int, int);                      \
    MACRO_NAME(OP, long, long);                    \
    MACRO_NAME(OP, unsigned int, uint);            \
    MACRO_NAME(OP, unsigned long, ulong);          \
    MACRO_NAME(OP, unsigned long long, ulonglong); \
    MACRO_NAME(OP, int32_t, int32);                \
    MACRO_NAME(OP, uint32_t, uint32);              \
    MACRO_NAME(OP, uint64_t, uint64);              \
    MACRO_NAME(OP, size_t, size);

#define TEST_NVSHMEM_ATOMIC_COMP_CUBIN(TYPENAME, TYPE, OP)                                   \
    void *args_##TYPENAME##_comp_##OP[] = {(void *)&remote, (void *)&old_value,              \
                                           (void *)&new_value, (void *)&match,               \
                                           (void *)&_dynamic_smem_size};                     \
    CUfunction test_##TYPENAME##_comp_##OP##_cubin;                                          \
    init_test_case_kernel(&test_##TYPENAME##_comp_##OP##_cubin,                              \
                          NVSHMEMI_TEST_STRINGIFY(test_nvshmem_##TYPENAME##_##OP##_kernel)); \
    CU_CHECK(cuLaunchKernel(test_##TYPENAME##_comp_##OP##_cubin, 1, 1, 1, 1, 1, 1,           \
                            _dynamic_smem_size, 0, args_##TYPENAME##_comp_##OP, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define TEST_NVSHMEM_ATOMIC_COMPARE_SWAP_KERNEL(OP, TYPE, TYPENAME)                                \
    __global__ void test_nvshmem_##TYPENAME##_##OP##_kernel(                                       \
        TYPE *remote, TYPE old_value, TYPE new_value, bool match, size_t dynamic_smem_size) {      \
        TYPE old;                                                                                  \
        const int mype = nvshmem_my_pe();                                                          \
        const int npes = nvshmem_n_pes();                                                          \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                                 \
        nvshmem_barrier_all();                                                                     \
        switch (OP) {                                                                              \
            case ATOMIC_COMPARE_SWAP:                                                              \
                old = nvshmem_##TYPENAME##_atomic_compare_swap(remote, old_value, new_value,       \
                                                               (mype + 1) % npes);                 \
                break;                                                                             \
            default:                                                                               \
                printf("invalid operation (%d)\n", OP);                                            \
                assert(0);                                                                         \
        }                                                                                          \
        nvshmem_barrier_all();                                                                     \
        if (*remote != new_value && match) {                                                       \
            printf("PE %i observed error with TEST_NVSHMEM_COMPARE_SWAP(%s, %s)\n", mype, #OP,     \
                   #TYPE);                                                                         \
            error_d = 1;                                                                           \
        } else if (!match && *remote == new_value) {                                               \
            printf("PE %i observed bad write with TEST_NVSHMEM_COMPARE_SWAP(%s, %s)\n", mype, #OP, \
                   #TYPE);                                                                         \
            error_d = 1;                                                                           \
        }                                                                                          \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                              \
        if (match && old != old_value) {                                                           \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, #TYPE);          \
            error_d = 1;                                                                           \
        } else if (!match && old == old_value) {                                                   \
            printf("PE %i error unexpected consistent value of old (%s, %s)\n", mype, #OP, #TYPE); \
            error_d = 1;                                                                           \
        }                                                                                          \
    }
REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_COMPARE_SWAP_KERNEL, ATOMIC_COMPARE_SWAP)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define TEST_NVSHMEM_ATOMIC_COMPARE_SWAP(OP, TYPE, TYPENAME)                                \
    do {                                                                                    \
        TYPE *remote;                                                                       \
        if (use_mmap) {                                                                     \
            remote = (TYPE *)allocate_mmap_buffer(sizeof(TYPE), _mem_handle_type, use_egm); \
            DEBUG_PRINT("Allocating mmaped buffer\n");                                      \
        } else {                                                                            \
            remote = (TYPE *)nvshmem_malloc(sizeof(TYPE));                                  \
        }                                                                                   \
        TYPE remote_h;                                                                      \
        TYPE new_value;                                                                     \
        TYPE old_value;                                                                     \
        bool match;                                                                         \
        remote_h = (TYPE)0x0123456789ABCDEFull;                                             \
        old_value = (TYPE)0x0123456789ABCDEFull;                                            \
        new_value = (TYPE)0x1234567812345678ull;                                            \
        match = true;                                                                       \
        CUDA_CHECK(cudaMemcpy(remote, &remote_h, sizeof(TYPE), cudaMemcpyHostToDevice));    \
        nvshmem_barrier_all();                                                              \
        if (use_cubin) {                                                                    \
            TEST_NVSHMEM_ATOMIC_COMP_CUBIN(TYPENAME, TYPE, OP);                             \
        } else {                                                                            \
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_nvshmem_##TYPENAME##_##OP##_kernel,      \
                                              _dynamic_smem_size);                          \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1, _dynamic_smem_size>>>(          \
                remote, old_value, new_value, match, _dynamic_smem_size);                   \
        }                                                                                   \
        cudaDeviceSynchronize();                                                            \
        old_value = (TYPE)0x23456789ABCDEF0Full;                                            \
        new_value = (TYPE)0x2345678923456789ull;                                            \
        match = false;                                                                      \
        nvshmem_barrier_all();                                                              \
        if (use_cubin) {                                                                    \
            TEST_NVSHMEM_ATOMIC_COMP_CUBIN(TYPENAME, TYPE, OP);                             \
        } else {                                                                            \
            CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(test_nvshmem_##TYPENAME##_##OP##_kernel,      \
                                              _dynamic_smem_size);                          \
            test_nvshmem_##TYPENAME##_##OP##_kernel<<<1, 1, _dynamic_smem_size>>>(          \
                remote, old_value, new_value, match, _dynamic_smem_size);                   \
        }                                                                                   \
        cudaDeviceSynchronize();                                                            \
    } while (0)

int main(int argc, char *argv[]) {
    int ret_val = 0, zero = 0;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    cudaMemcpyToSymbol(error_d, &zero, sizeof(int), 0);
    cudaDeviceSynchronize();

    REPT_MACRO_FOR_TYPES(TEST_NVSHMEM_ATOMIC_COMPARE_SWAP, ATOMIC_COMPARE_SWAP)

    cudaMemcpyFromSymbol(&ret_val, error_d, sizeof(int), 0);

    finalize_wrapper();
    return ret_val;
}
