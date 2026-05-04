/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "signal.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define THREADS 512

__device__ void rma_inline_wrapper(uint64_t *src, uint64_t *dest, uint64_t pe) {
    nvshmemx_signal_op(dest, *src, NVSHMEM_SIGNAL_SET, pe);
}

#define TEST_NVSHMEM_ALL_CUBIN()                                                          \
    size_t cubin_dynamic_smem_size = 0;                                                   \
    void *args_all[] = {(void *)&src_, (void *)&dest_, (void *)&len,                      \
                        (void *)&mype, (void *)&npes,  (void *)&cubin_dynamic_smem_size}; \
    CUfunction test_all_cubin;                                                            \
    if (typeid(T) == typeid(uint64_t)) {                                                  \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_uint64)); \
    }                                                                                     \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_all, NULL));

#define TEST_NVSHMEM_RING_CUBIN()                                                      \
    size_t cubin_dynamic_smem_size = 0;                                                \
    void *args_ring[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe, \
                         (void *)&cubin_dynamic_smem_size};                            \
    CUfunction test_ring_cubin;                                                        \
    if (typeid(T) == typeid(uint64_t)) {                                               \
        init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring_uint64)); \
    }                                                                                  \
    CU_CHECK(cuLaunchKernel(test_ring_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_ring, NULL));

template <typename T>
__global__ void alltoall(T *src, T *dest, size_t len, int mype, int npes,
                         size_t dynamic_smem_size) {
    int tid = threadIdx.x;

    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    for (int i = 0; i < npes; i++) {
        for (int j = tid; j < len; j += THREADS) {
            rma_inline_wrapper(src + i * len + j, dest + mype * len + j, i);
        }
        __syncthreads();
    }

    if (!tid) nvshmem_quiet();
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

template <typename T>
__global__ void ring(T *src, T *dest, int len, int nextpe, size_t dynamic_smem_size) {
    int tid = threadIdx.x;

    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
    for (int j = tid; j < len; j += THREADS) {
        rma_inline_wrapper(src + j, dest + j, nextpe);
    }
    __syncthreads();

    if (!tid) nvshmem_quiet();
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

#define ALLTOALL_TEMP(dynamic_smem_size)                                     \
    int tid = threadIdx.x;                                                   \
    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);                               \
    for (int i = 0; i < npes; i++) {                                         \
        for (int j = tid; j < len; j += THREADS) {                           \
            rma_inline_wrapper(src + i * len + j, dest + mype * len + j, i); \
        }                                                                    \
        __syncthreads();                                                     \
    }                                                                        \
    if (!tid) nvshmem_quiet();                                               \
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);

#define RING_TEMP(dynamic_smem_size)                   \
    int tid = threadIdx.x;                             \
    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);         \
    for (int j = tid; j < len; j += THREADS) {         \
        rma_inline_wrapper(src + j, dest + j, nextpe); \
    }                                                  \
    __syncthreads();                                   \
    if (!tid) nvshmem_quiet();                         \
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);

#if defined(NVSHMEM_HOSTLIB_ONLY)
extern "C" {

__global__ void alltoall_uint64(uint64_t *src, uint64_t *dest, size_t len, int mype, int npes,
                                size_t dynamic_smem_size);
__global__ void ring_uint64(uint64_t *src, uint64_t *dest, int len, int nextpe,
                            size_t dynamic_smem_size);

__global__ void alltoall_uint64(uint64_t *src, uint64_t *dest, size_t len, int mype, int npes,
                                size_t dynamic_smem_size) {
    ALLTOALL_TEMP(dynamic_smem_size);
}

__global__ void ring_uint64(uint64_t *src, uint64_t *dest, int len, int nextpe,
                            size_t dynamic_smem_size) {
    RING_TEMP(dynamic_smem_size);
}
}
#endif

template <typename T>
void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(alltoall<T>, _dynamic_smem_size);
    if (use_cubin) {
        TEST_NVSHMEM_ALL_CUBIN();
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
        TEST_NVSHMEM_RING_CUBIN();
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
    if (status) goto out;

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    status = test<uint64_t>(launch_alltoall<uint64_t>, launch_ring<uint64_t>);
    if (status) goto out;

    cleanup();

out:
    return status;
}
