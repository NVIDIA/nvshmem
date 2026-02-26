/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "swap.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define THREADS 512

__device__ void rma_inline_wrapper(int *src, int *dest, int pe) {
    int value = nvshmem_int_atomic_swap(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned int *src, unsigned int *dest, int pe) {
    unsigned int value = nvshmem_uint_atomic_swap(dest, *src, pe);
}
__device__ void rma_inline_wrapper(long long int *src, long long int *dest, int pe) {
    long long int value = nvshmem_longlong_atomic_swap(dest, *src, pe);
}
__device__ void rma_inline_wrapper(unsigned long long int *src, unsigned long long int *dest,
                                   int pe) {
    unsigned long long int value = nvshmem_ulonglong_atomic_swap(dest, *src, pe);
}

#define TEST_NVSHMEM_ALL_CUBIN(TYPENAME)                                                       \
    void *args_all_##TYPENAME[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype, \
                                   (void *)&npes};                                             \
    CUfunction test_all_##TYPENAME##_cubin;                                                    \
    if (typeid(T) == typeid(int)) {                                                            \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                    \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_int));                          \
    }                                                                                          \
    if (typeid(T) == typeid(unsigned int)) {                                                   \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                    \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_uint));                         \
    }                                                                                          \
    if (typeid(T) == typeid(long long int)) {                                                  \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                    \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_longlong));                     \
    }                                                                                          \
    if (typeid(T) == typeid(unsigned long long int)) {                                         \
        init_test_case_kernel(&test_all_##TYPENAME##_cubin,                                    \
                              NVSHMEMI_TEST_STRINGIFY(alltoall_ulonglong));                    \
    }                                                                                          \
    CU_CHECK(cuLaunchKernel(test_all_##TYPENAME##_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm,     \
                            args_all_##TYPENAME, NULL));

#define TEST_NVSHMEM_RING_CUBIN(TYPENAME)                                                          \
    void *args_ring_##TYPENAME[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe}; \
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
    CU_CHECK(cuLaunchKernel(test_ring_##TYPENAME##_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm,        \
                            args_ring_##TYPENAME, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DEFINE_GROUP(TYPENAME, TYPE)                                                             \
    __global__ void alltoall_##TYPENAME(TYPE *src, TYPE *dest, size_t len, int mype, int npes) { \
        int tid = threadIdx.x;                                                                   \
        for (int i = 0; i < npes; i++) {                                                         \
            for (int j = tid; j < len; j += THREADS) {                                           \
                rma_inline_wrapper(src + i * len + j, dest + mype * len + j, i);                 \
            }                                                                                    \
            __syncthreads();                                                                     \
        }                                                                                        \
        if (!tid) nvshmem_quiet();                                                               \
    }                                                                                            \
    __global__ void ring_##TYPENAME(TYPE *src, TYPE *dest, size_t len, int nextpe) {             \
        int tid = threadIdx.x;                                                                   \
        for (int j = tid; j < len; j += THREADS) {                                               \
            rma_inline_wrapper(src + j, dest + j, nextpe);                                       \
        }                                                                                        \
        __syncthreads();                                                                         \
        if (!tid) nvshmem_quiet();                                                               \
    }

DEFINE_GROUP(int, int);
DEFINE_GROUP(uint, unsigned int);
DEFINE_GROUP(longlong, long long int);
DEFINE_GROUP(ulonglong, unsigned long long int);

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

template <typename T>
__global__ void alltoall(T *src, T *dest, size_t len, int mype, int npes) {
    int tid = threadIdx.x;

    for (int i = 0; i < npes; i++) {
        for (int j = tid; j < len; j += THREADS) {
            rma_inline_wrapper(src + i * len + j, dest + mype * len + j, i);
        }
        __syncthreads();
    }

    if (!tid) nvshmem_quiet();
}

template <typename T>
__global__ void ring(T *src, T *dest, int len, int nextpe) {
    int tid = threadIdx.x;

    for (int j = tid; j < len; j += THREADS) {
        rma_inline_wrapper(src + j, dest + j, nextpe);
    }
    __syncthreads();

    if (!tid) nvshmem_quiet();
}

template <typename T>
void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_ALL_CUBIN(T);
    } else {
        alltoall<T><<<1, THREADS, 0, cstrm>>>(src_, dest_, len, mype, npes);
    }
}

template <typename T>
void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_RING_CUBIN(T);
    } else {
        ring<T><<<1, THREADS, 0, cstrm>>>(src_, dest_, len, nextpe);
    }
}

int main(int c, char *v[]) {
    int status = 0;
    int max_msg_size = 8 * 1024;
    int iters = 50;

    status = setup(1, 1, max_msg_size, iters);
    if (status) goto out;

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    status = test<int>(launch_alltoall<int>, launch_ring<int>);
    if (status) goto out;

    status = test<unsigned int>(launch_alltoall<unsigned int>, launch_ring<unsigned int>);
    if (status) goto out;

    status = test<long long int>(launch_alltoall<long long int>, launch_ring<long long int>);
    if (status) goto out;

    status = test<unsigned long long int>(launch_alltoall<unsigned long long int>,
                                          launch_ring<unsigned long long int>);
    if (status) goto out;

    cleanup();

out:
    return status;
}
