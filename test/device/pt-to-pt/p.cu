/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "p.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define THREADS 512

__device__ void rma_inline_wrapper(char *src, char *dest, int len, int pe) {
    nvshmem_char_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned char *src, unsigned char *dest, int len, int pe) {
    nvshmem_uchar_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(short *src, short *dest, int len, int pe) {
    nvshmem_short_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned short *src, unsigned short *dest, int len, int pe) {
    nvshmem_ushort_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(int *src, int *dest, int len, int pe) {
    nvshmem_int_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned int *src, unsigned int *dest, int len, int pe) {
    nvshmem_uint_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(long long int *src, long long int *dest, int len, int pe) {
    nvshmem_longlong_p(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned long long int *src, unsigned long long int *dest,
                                   int len, int pe) {
    nvshmem_ulonglong_p(dest, *src, pe);
}

#define TEST_NVSHMEM_ALL_CUBIN(TYPENAME)                                                   \
    size_t cubin_dynamic_smem_size = 0;                                                    \
    void *args_all_##TYPENAME[] = {                                                        \
        (void *)&src_, (void *)&dest_, (void *)&len,                                       \
        (void *)&mype, (void *)&npes,  (void *)&cubin_dynamic_smem_size};                  \
    CUfunction test_all_##TYPENAME##_cubin;                                                \
    if (typeid(T) == typeid(int)) {                                                        \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_int));                      \
    }                                                                                      \
    if (typeid(T) == typeid(unsigned int)) {                                               \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_uint));                     \
    }                                                                                      \
    if (typeid(T) == typeid(long long int)) {                                              \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_longlong));                 \
    }                                                                                      \
    if (typeid(T) == typeid(unsigned long long int)) {                                     \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_ulonglong));                \
    }                                                                                      \
    if (typeid(T) == typeid(char)) {                                                       \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_char));                     \
    }                                                                                      \
    if (typeid(T) == typeid(unsigned char)) {                                              \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_uchar));                    \
    }                                                                                      \
    if (typeid(T) == typeid(short)) {                                                      \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_short));                    \
    }                                                                                      \
    if (typeid(T) == typeid(unsigned short)) {                                             \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_ushort));                   \
    }                                                                                      \
    CU_CHECK(cuLaunchKernel(test_all_##TYPENAME##_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, \
                            args_all_##TYPENAME, NULL));

#define TEST_NVSHMEM_RING_CUBIN(TYPENAME)                                                          \
    size_t cubin_dynamic_smem_size = 0;                                                            \
    void *args_ring_##TYPENAME[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe,  \
                                    (void *)&cubin_dynamic_smem_size};                             \
    CUfunction test_ring_##TYPENAME##_cubin;                                                       \
    if (typeid(T) == typeid(int)) {                                                                \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin, NVSHMEMI_TEST_STRINGIFY(ring_int));   \
    }                                                                                              \
    if (typeid(T) == typeid(unsigned int)) {                                                       \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin, NVSHMEMI_TEST_STRINGIFY(ring_uint));  \
    }                                                                                              \
    if (typeid(T) == typeid(long long int)) {                                                      \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin,                                       \
                              NVSHMEMI_TEST_STRINGIFY(ring_longlong));                             \
    }                                                                                              \
    if (typeid(T) == typeid(unsigned long long int)) {                                             \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin,                                       \
                              NVSHMEMI_TEST_STRINGIFY(ring_ulonglong));                            \
    }                                                                                              \
    if (typeid(T) == typeid(char)) {                                                               \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin, NVSHMEMI_TEST_STRINGIFY(ring_char));  \
    }                                                                                              \
    if (typeid(T) == typeid(unsigned char)) {                                                      \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin, NVSHMEMI_TEST_STRINGIFY(ring_uchar)); \
    }                                                                                              \
    if (typeid(T) == typeid(short)) {                                                              \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin, NVSHMEMI_TEST_STRINGIFY(ring_short)); \
    }                                                                                              \
    if (typeid(T) == typeid(unsigned short)) {                                                     \
        init_test_case_kernel(&test_ring_##TYPENAME##_cubin,                                       \
                              NVSHMEMI_TEST_STRINGIFY(ring_ushort));                               \
    }                                                                                              \
    CU_CHECK(cuLaunchKernel(test_ring_##TYPENAME##_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm,        \
                            args_ring_##TYPENAME, NULL));

#if defined(NVSHMEM_HOSTLIB_ONLY)
extern "C" {

#define DEFINE_Group(TYPENAME, TYPE)                                                           \
    __global__ void alltoall_##TYPENAME(TYPE *src, TYPE *dest, size_t len, int mype, int npes, \
                                        size_t dynamic_smem_size) {                            \
        int tid = threadIdx.x;                                                                 \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                             \
        for (int i = 0; i < npes; i++) {                                                       \
            for (int j = tid; j < len; j += THREADS) {                                         \
                rma_inline_wrapper(src + i * len + j, dest + mype * len + j, len, i);          \
            }                                                                                  \
            __syncthreads();                                                                   \
        }                                                                                      \
        if (!tid) nvshmem_quiet();                                                             \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                          \
    }                                                                                          \
    __global__ void ring_##TYPENAME(TYPE *src, TYPE *dest, size_t len, int nextpe,             \
                                    size_t dynamic_smem_size) {                                \
        int tid = threadIdx.x;                                                                 \
        NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                                             \
        for (int j = tid; j < len; j += THREADS) {                                             \
            rma_inline_wrapper(src + j, dest + j, len, nextpe);                                \
        }                                                                                      \
        __syncthreads();                                                                       \
        if (!tid) nvshmem_quiet();                                                             \
        NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);                                          \
    }

DEFINE_Group(int, int);
DEFINE_Group(uint, unsigned int);
DEFINE_Group(longlong, long long int);
DEFINE_Group(ulonglong, unsigned long long int);
DEFINE_Group(char, char);
DEFINE_Group(uchar, unsigned char);
DEFINE_Group(short, short);
DEFINE_Group(ushort, unsigned short);
}
#endif

template <typename T>
__global__ void alltoall(T *src, T *dest, size_t len, int mype, int npes,
                         size_t dynamic_smem_size) {
    int tid = threadIdx.x;

    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    for (int i = 0; i < npes; i++) {
        for (int j = tid; j < len; j += THREADS) {
            rma_inline_wrapper(src + i * len + j, dest + mype * len + j, len, i);
        }
        __syncthreads();
    }

    if (!tid) {
        nvshmem_quiet();
    }
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

template <typename T>
__global__ void ring(T *src, T *dest, int len, int nextpe, size_t dynamic_smem_size) {
    int tid = threadIdx.x;

    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    for (int j = tid; j < len; j += THREADS) {
        rma_inline_wrapper(src + j, dest + j, len, nextpe);
    }
    __syncthreads();

    if (!tid) {
        nvshmem_quiet();
    }
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

template <typename T>
void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(alltoall<T>, _dynamic_smem_size);
    if (use_cubin) {
        TEST_NVSHMEM_ALL_CUBIN(T);
    } else {
        alltoall<T><<<1, THREADS, _dynamic_smem_size, cstrm>>>(src_, dest_, len, mype, npes,
                                                               _dynamic_smem_size);
    }
}

template <typename T>
void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(ring<T>, _dynamic_smem_size);
    if (use_cubin) {
        TEST_NVSHMEM_RING_CUBIN(T);
    } else {
        ring<T><<<1, THREADS, _dynamic_smem_size, cstrm>>>(src_, dest_, len, nextpe,
                                                           _dynamic_smem_size);
    }
}

int main(int c, char *v[]) {
    int status = 0;
    int max_msg_size = 8 * 1024;
    int iters = 50;

    status = setup(1, 1, max_msg_size, iters);
    if (status) {
        goto out;
    }

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    status = test<char>(launch_alltoall<char>, launch_ring<char>);
    if (status) {
        goto out;
    }

    status = test<unsigned char>(launch_alltoall<unsigned char>, launch_ring<unsigned char>);
    if (status) {
        goto out;
    }

    status = test<short>(launch_alltoall<short>, launch_ring<short>);
    if (status) {
        goto out;
    }

    status = test<unsigned short>(launch_alltoall<unsigned short>, launch_ring<unsigned short>);
    if (status) {
        goto out;
    }

    status = test<int>(launch_alltoall<int>, launch_ring<int>);
    if (status) {
        goto out;
    }

    status = test<unsigned int>(launch_alltoall<unsigned int>, launch_ring<unsigned int>);
    if (status) {
        goto out;
    }

    status = test<long long int>(launch_alltoall<long long int>, launch_ring<long long int>);
    if (status) {
        goto out;
    }

    status = test<unsigned long long int>(launch_alltoall<unsigned long long int>,
                                          launch_ring<unsigned long long int>);
    if (status) {
        goto out;
    }

    cleanup();

out:
    return status;
}
