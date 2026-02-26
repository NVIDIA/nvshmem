/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "g.cubin"

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define THREADS 512

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

__device__ void rma_inline_wrapper(int *src, int *dest, int len, int pe) {
    *dest = nvshmem_int_g(src, pe);
}

__device__ void rma_inline_wrapper(double *src, double *dest, int len, int pe) {
    *dest = nvshmem_double_g(src, pe);
}

#define TEST_NVSHMEM_ALL_CUBIN()                                                          \
    void *args_all[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype,       \
                        (void *)&npes};                                                   \
    CUfunction test_all_cubin;                                                            \
    if (typeid(T) == typeid(int)) {                                                       \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_int));    \
    }                                                                                     \
    if (typeid(T) == typeid(double)) {                                                    \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_double)); \
    }                                                                                     \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_all, NULL));

#define TEST_NVSHMEM_RING_CUBIN()                                                       \
    void *args_ring[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&prevpe}; \
    CUfunction test_ring_cubin;                                                         \
    if (typeid(T) == typeid(int)) {                                                     \
        init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring_int));     \
    }                                                                                   \
    if (typeid(T) == typeid(double)) {                                                  \
        init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring_double));  \
    }                                                                                   \
    CU_CHECK(cuLaunchKernel(test_ring_cubin, 1, 1, 1, THREADS, 1, 1, 0, cstrm, args_ring, NULL));

template <typename T>
__global__ void alltoall(T *src, T *dest, size_t len, int mype, int npes) {
    int tid = threadIdx.x;

    for (int i = 0; i < npes; i++) {
        for (int j = tid; j < len; j += THREADS) {
            rma_inline_wrapper(src + mype * len + j, dest + i * len + j, len, i);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void ring(T *src, T *dest, int len, int prevpe) {
    int tid = threadIdx.x;

    for (int j = tid; j < len; j += THREADS) {
        rma_inline_wrapper(src + j, dest + j, len, prevpe);
    }
}

#define ALLTOALL_TEMP()                                                           \
    int tid = threadIdx.x;                                                        \
    for (int i = 0; i < npes; i++) {                                              \
        for (int j = tid; j < len; j += THREADS) {                                \
            rma_inline_wrapper(src + mype * len + j, dest + i * len + j, len, i); \
        }                                                                         \
        __syncthreads();                                                          \
    }

#define RING_TEMP()                                         \
    int tid = threadIdx.x;                                  \
    for (int j = tid; j < len; j += THREADS) {              \
        rma_inline_wrapper(src + j, dest + j, len, prevpe); \
    }

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void alltoall_int(int *src, int *dest, size_t len, int mype, int npes);
__global__ void ring_int(int *src, int *dest, int len, int prevpe);
__global__ void alltoall_double(double *src, double *dest, size_t len, int mype, int npes);
__global__ void ring_double(double *src, double *dest, int len, int prevpe);

__global__ void alltoall_int(int *src, int *dest, size_t len, int mype, int npes) {
    ALLTOALL_TEMP();
}

__global__ void ring_int(int *src, int *dest, int len, int prevpe) { RING_TEMP(); }
__global__ void alltoall_double(double *src, double *dest, size_t len, int mype, int npes) {
    ALLTOALL_TEMP();
}
__global__ void ring_double(double *src, double *dest, int len, int prevpe) { RING_TEMP(); }

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
        ring<T><<<1, THREADS, 0, cstrm>>>(src_, dest_, len, prevpe);
    }
}
int main(int c, char *v[]) {
    int status = 0;
    int max_msg_size = 8 * 1024;
    int iters = 50;

    status = setup(1, 1, max_msg_size, iters, true);
    if (status) goto out;

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    DEBUG_PRINT("testing nvshmem_int_g\n");
    status = test<int>(launch_alltoall<int>, launch_ring<int>);
    if (status) goto out;
    DEBUG_PRINT("testing nvshmem_double_g\n");
    status = test<double>(launch_alltoall<double>, launch_ring<double>);
    if (status) goto out;

    cleanup();

out:
    return status;
}
