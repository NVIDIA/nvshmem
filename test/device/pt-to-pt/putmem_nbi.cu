/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#define CUMODULE_NAME "putmem_nbi.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include "ring_alltoall.h"

#define TEST_NVSHMEM_ALL_CUBIN()                                                    \
    void *args_all[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype, \
                        (void *)&npes};                                             \
    CUfunction test_all_cubin;                                                      \
    init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall));      \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_all, NULL));

#define TEST_NVSHMEM_RING_CUBIN()                                                       \
    void *args_ring[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe}; \
    CUfunction test_ring_cubin;                                                         \
    init_test_case_kernel(&test_ring_cubin, NVSHMEMI_TEST_STRINGIFY(ring));             \
    CU_CHECK(cuLaunchKernel(test_ring_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_ring, NULL));

#define TEST_NVSHMEM_ALL_G_CUBIN(GROUP)                                                \
    void *args_all_g[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&mype,  \
                          (void *)&npes};                                              \
    CUfunction test_all_cubin;                                                         \
    init_test_case_kernel(&test_all_cubin, NVSHMEMI_TEST_STRINGIFY(alltoall_##GROUP)); \
    CU_CHECK(cuLaunchKernel(test_all_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_all_g, NULL));

#define TEST_NVSHMEM_RING_G_CUBIN(GROUP)                                                  \
    void *args_ring_g[] = {(void *)&src_, (void *)&dest_, (void *)&len, (void *)&nextpe}; \
    CUfunction test_ring_g_cubin;                                                         \
    init_test_case_kernel(&test_ring_g_cubin, NVSHMEMI_TEST_STRINGIFY(ring_##GROUP));     \
    CU_CHECK(cuLaunchKernel(test_ring_g_cubin, 1, 1, 1, 1, 1, 1, 0, cstrm, args_ring_g, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

#define DEFINE_THREADGROUP_API(Group)                                                           \
    __global__ void alltoall_##Group(int *src, int *dest, size_t len, int mype, int npes) {     \
        for (int i = 0; i < npes; i++) {                                                        \
            nvshmemx_putmem_nbi_##Group((void *)(dest + (size_t)mype * len),                    \
                                        (void *)(src + (size_t)i * len), len * sizeof(int), i); \
        }                                                                                       \
        nvshmem_quiet();                                                                        \
    }                                                                                           \
                                                                                                \
    __global__ void ring_##Group(int *src, int *dest, size_t len, int nextpe) {                 \
        nvshmemx_putmem_nbi_##Group((void *)dest, (void *)src, len * sizeof(int), nextpe);      \
        nvshmem_quiet();                                                                        \
    }                                                                                           \
                                                                                                \
    void launch_alltoall_##Group(void *src, void *dest, size_t len, int mype, int npes,         \
                                 cudaStream_t cstrm) {                                          \
        int *src_ = (int *)src;                                                                 \
        int *dest_ = (int *)dest;                                                               \
        if (use_cubin) {                                                                        \
            TEST_NVSHMEM_ALL_G_CUBIN(Group);                                                    \
        } else {                                                                                \
            alltoall_##Group<<<1, 1, 0, cstrm>>>(src_, dest_, len, mype, npes);                 \
        }                                                                                       \
    }                                                                                           \
                                                                                                \
    void launch_ring_##Group(void *src, void *dest, size_t len, int nextpe, int prevpe,         \
                             cudaStream_t cstrm) {                                              \
        int *src_ = (int *)src;                                                                 \
        int *dest_ = (int *)dest;                                                               \
        if (use_cubin) {                                                                        \
            TEST_NVSHMEM_RING_G_CUBIN(Group);                                                   \
        } else {                                                                                \
            ring_##Group<<<1, 1, 0, cstrm>>>(src_, dest_, len, nextpe);                         \
        }                                                                                       \
    }

DEFINE_THREADGROUP_API(warp)
DEFINE_THREADGROUP_API(block)

__global__ void alltoall(int *src, int *dest, size_t len, int mype, int npes) {
    for (int i = 0; i < npes; i++) {
        nvshmem_putmem_nbi((void *)(dest + (size_t)mype * len), (void *)(src + (size_t)i * len),
                           len * sizeof(int), i);
    }
    nvshmem_quiet();
}

__global__ void ring(int *src, int *dest, size_t len, int nextpe) {
    nvshmem_putmem_nbi((void *)dest, (void *)src, len * sizeof(int), nextpe);
    nvshmem_quiet();
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes, cudaStream_t cstrm) {
    int *src_ = (int *)src;
    int *dest_ = (int *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_ALL_CUBIN();
    } else {
        alltoall<<<1, 1, 0, cstrm>>>(src_, dest_, len, mype, npes);
    }
}

void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe, cudaStream_t cstrm) {
    int *src_ = (int *)src;
    int *dest_ = (int *)dest;
    if (use_cubin) {
        TEST_NVSHMEM_RING_CUBIN();
    } else {
        ring<<<1, 1, 0, cstrm>>>(src_, dest_, len, nextpe);
    }
}

int main(int c, char *v[]) {
    int status = 0;

    status = setup(0, 1);
    if (status) goto out;

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    status = test<int>(launch_alltoall, launch_ring);
    if (status) goto out;

    status = test<int>(launch_alltoall_warp, launch_ring_warp);
    if (status) goto out;

    status = test<int>(launch_alltoall_block, launch_ring_block);
    if (status) goto out;

    cleanup();

out:
    return status;
}
