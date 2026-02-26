/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
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
    void *args_all[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype,       \
                        (void *)&npes};                                                   \
    CUfunction test_all_cubin;                                                            \
    if (typeid(T) == typeid(uint64_t)) {                                                  \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_uint64)); \
    }                                                                                     \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_all, NULL));

#define TEST_NVSHMEM_RING_CUBIN()                                                       \
    void *args_ring[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe}; \
    CUfunction test_ring_cubin;                                                         \
    if (typeid(T) == typeid(uint64_t)) {                                                \
        init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring_uint64));  \
    }                                                                                   \
    CU_CHECK(cuLaunchKernel(test_ring_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_ring, NULL));

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

#define ALLTOALL_TEMP()                                                      \
    int tid = threadIdx.x;                                                   \
    for (int i = 0; i < npes; i++) {                                         \
        for (int j = tid; j < len; j += THREADS) {                           \
            rma_inline_wrapper(src + i * len + j, dest + mype * len + j, i); \
        }                                                                    \
        __syncthreads();                                                     \
    }                                                                        \
    if (!tid) nvshmem_quiet();

#define RING_TEMP()                                    \
    int tid = threadIdx.x;                             \
    for (int j = tid; j < len; j += THREADS) {         \
        rma_inline_wrapper(src + j, dest + j, nextpe); \
    }                                                  \
    __syncthreads();                                   \
    if (!tid) nvshmem_quiet();

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void alltoall_uint64(uint64_t *src, uint64_t *dest, size_t len, int mype, int npes);
__global__ void ring_uint64(uint64_t *src, uint64_t *dest, int len, int nextpe);

__global__ void alltoall_uint64(uint64_t *src, uint64_t *dest, size_t len, int mype, int npes) {
    ALLTOALL_TEMP();
}

__global__ void ring_uint64(uint64_t *src, uint64_t *dest, int len, int nextpe) { RING_TEMP(); }

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

template <typename T>
void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_ALL_CUBIN();
    } else {
        alltoall<T><<<1, THREADS, 0, cstrm>>>(src_, dest_, len, mype, npes);
    }
}

template <typename T>
void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_RING_CUBIN();
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

    status = test<uint64_t>(launch_alltoall<uint64_t>, launch_ring<uint64_t>);
    if (status) goto out;

    cleanup();

out:
    return status;
}
