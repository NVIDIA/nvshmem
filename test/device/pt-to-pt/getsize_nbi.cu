/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "getsize_nbi.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define TEST_NVSHMEM_ALL_CUBIN()                                                       \
    void *args_all[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype,    \
                        (void *)&npes};                                                \
    CUfunction test_all_cubin;                                                         \
    if (typeid(T) == typeid(int)) {                                                    \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_int)); \
    }                                                                                  \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_all, NULL));

#define TEST_NVSHMEM_RING_CUBIN()                                                       \
    void *args_ring[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&prevpe}; \
    CUfunction test_ring_cubin;                                                         \
    if (typeid(T) == typeid(int)) {                                                     \
        init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring_int));     \
    }                                                                                   \
    CU_CHECK(cuLaunchKernel(test_ring_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_ring, NULL));

#define TEST_NVSHMEM_ALL_G_CUBIN(GROUP)                                                        \
    void *args_all_g[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype,          \
                          (void *)&npes};                                                      \
    CUfunction test_all_cubin;                                                                 \
    if (typeid(T) == typeid(int)) {                                                            \
        init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_int_##GROUP)); \
    }                                                                                          \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_all_g, NULL));

#define TEST_NVSHMEM_RING_G_CUBIN(GROUP)                                                      \
    void *args_ring_g[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&prevpe};     \
    CUfunction test_ring_g_cubin;                                                             \
    if (typeid(T) == typeid(int)) {                                                           \
        init_test_case_kernel(&test_ring_g_cubin, NVSHMEMI_TEST_STRINGIFY(ring_int_##GROUP)); \
    }                                                                                         \
    CU_CHECK(cuLaunchKernel(test_ring_g_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_ring_g, NULL));

#define DEFINE_RMA_GET_WRAPPER(Group)                                                  \
    __device__ void rma_get_wrapper_##Group(int *src, int *dest, size_t len, int pe) { \
        nvshmemx_get32_nbi_##Group(dest, src, len, pe);                                \
    }

DEFINE_RMA_GET_WRAPPER(warp)
DEFINE_RMA_GET_WRAPPER(block)

#define DEFINE_THREADGROUP_API(Group)                                                          \
    template <typename T>                                                                      \
    __global__ void alltoall_##Group(T *src, T *dest, size_t len, int mype, int npes) {        \
        for (int i = 0; i < npes; i++) {                                                       \
            rma_get_wrapper_##Group(src + (size_t)mype * len, dest + (size_t)i * len, len, i); \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }                                                                                          \
                                                                                               \
    template <typename T>                                                                      \
    __global__ void ring_##Group(T *src, T *dest, size_t len, int prevpe) {                    \
        rma_get_wrapper_##Group(src, dest, len, prevpe);                                       \
        nvshmem_quiet();                                                                       \
    }                                                                                          \
                                                                                               \
    template <typename T>                                                                      \
    void launch_alltoall_##Group(void *src, void *dest, size_t len, int mype, int npes,        \
                                 cudaStream_t cstrm) {                                         \
        T *src_ = (T *)src;                                                                    \
        T *dest_ = (T *)dest;                                                                  \
        if (use_cubin) {                                                                       \
            TEST_NVSHMEM_ALL_G_CUBIN(Group);                                                   \
        } else {                                                                               \
            alltoall_##Group<T><<<1, 1, 0, cstrm>>>(src_, dest_, len, mype, npes);             \
        }                                                                                      \
    }                                                                                          \
                                                                                               \
    template <typename T>                                                                      \
    void launch_ring_##Group(void *src, void *dest, size_t len, int nextpe, int prevpe,        \
                             cudaStream_t cstrm) {                                             \
        T *src_ = (T *)src;                                                                    \
        T *dest_ = (T *)dest;                                                                  \
        if (use_cubin) {                                                                       \
            TEST_NVSHMEM_RING_G_CUBIN(Group);                                                  \
        } else {                                                                               \
            ring_##Group<T><<<1, 1, 0, cstrm>>>(src_, dest_, len, prevpe);                     \
        }                                                                                      \
    }

DEFINE_THREADGROUP_API(warp)
DEFINE_THREADGROUP_API(block)

__device__ void rma_get_wrapper(int *src, int *dest, size_t len, int pe) {
    nvshmem_get32_nbi(dest, src, len, pe);
}

template <typename T>
__global__ void alltoall(T *src, T *dest, size_t len, int mype, int npes) {
    for (int i = 0; i < npes; i++) {
        rma_get_wrapper(src + (size_t)mype * len, dest + (size_t)i * len, len, i);
    }
    nvshmem_quiet();
}

template <typename T>
__global__ void ring(T *src, T *dest, size_t len, int prevpe) {
    rma_get_wrapper(src, dest, len, prevpe);
    nvshmem_quiet();
}

#define ALLTOALL_TEMP()                                                            \
    for (int i = 0; i < npes; i++) {                                               \
        rma_get_wrapper(src + (size_t)mype * len, dest + (size_t)i * len, len, i); \
    }                                                                              \
    nvshmem_quiet();

#define RING_TEMP()                          \
    rma_get_wrapper(src, dest, len, prevpe); \
    nvshmem_quiet();

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void alltoall_int(int *src, int *dest, size_t len, int mype, int npes) {
    ALLTOALL_TEMP();
}

__global__ void ring_int(int *src, int *dest, int len, int prevpe) { RING_TEMP(); }

#define DEFINE_Group(Group)                                                                     \
    __global__ void alltoall_int_##Group(int *src, int *dest, size_t len, int mype, int npes) { \
        for (int i = 0; i < npes; i++) {                                                        \
            rma_get_wrapper_##Group(src + (size_t)mype * len, dest + (size_t)i * len, len, i);  \
        }                                                                                       \
        nvshmem_quiet();                                                                        \
    }                                                                                           \
    __global__ void ring_int_##Group(int *src, int *dest, size_t len, int prevpe) {             \
        rma_get_wrapper_##Group(src, dest, len, prevpe);                                        \
        nvshmem_quiet();                                                                        \
    }
DEFINE_Group(warp) DEFINE_Group(block)

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
        alltoall<T><<<1, 1, 0, cstrm>>>(src_, dest_, len, mype, npes);
    }
}

template <typename T>
void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe, cudaStream_t cstrm) {
    T *src_ = (T *)src;
    T *dest_ = (T *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_RING_CUBIN();
    } else {
        ring<T><<<1, 1, 0, cstrm>>>(src_, dest_, len, prevpe);
    }
}

int main(int c, char *v[]) {
    int status = 0;

    status = setup(0, 1, MAX_MSG_SIZE, ITER, true);
    if (status) goto out;

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    status = test<int>(launch_alltoall<int>, launch_ring<int>);
    if (status) goto out;

    status = test<int>(launch_alltoall_warp<int>, launch_ring_warp<int>);
    if (status) goto out;

    status = test<int>(launch_alltoall_block<int>, launch_ring_block<int>);
    if (status) goto out;

    cleanup();

out:
    return status;
}
