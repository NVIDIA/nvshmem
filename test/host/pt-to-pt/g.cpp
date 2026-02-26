/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "nvshmem.h"
#include "nvshmemx.h"

#include "utils.h"

#define INIT_DEV_TEST_BUFF(nbytes)                                                               \
    for (int pe = 0; pe < npes; pe++) {                                                          \
        /*if(pe != mype) {*/                                                                     \
        for (size_t i = 0; i < nbytes; i++) {                                                    \
            *((char *)hostTestBuffer + pe * nbytes + i) = 0xAB;                                  \
        }                                                                                        \
        CUDA_CHECK(cudaMemcpy(devTestBuffer + pe * nbytes, hostTestBuffer + pe * nbytes, nbytes, \
                              cudaMemcpyHostToDevice));                                          \
        /*}*/                                                                                    \
    }

#define INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nused)                                                 \
    for (size_t i = 0; i < nelems * sizeof(long double); i++) {                                   \
        *((char *)hostRefBuffer + i) = 0;                                                         \
    }                                                                                             \
    CUDA_CHECK(                                                                                   \
        cudaMemcpy(buffer, hostRefBuffer, sizeof(long double) * nelems, cudaMemcpyHostToDevice)); \
    for (size_t i = 0; i < nused; i++) {                                                          \
        *((TYPE *)hostRefBuffer + i) = (TYPE)mype + (TYPE)npes;                                   \
    }                                                                                             \
    CUDA_CHECK(cudaMemcpy(buffer, hostRefBuffer, sizeof(TYPE) * nused, cudaMemcpyHostToDevice));

#define VALIDATE_G(TYPE, apiname)                                                 \
    for (pe = 0; pe < npes; pe++) {                                               \
        errs = 0;                                                                 \
        if (pe != mype) {                                                         \
            /*DEBUG_PRINT("[%d] %llf\n", mype, *((TYPE *)hostTestBuffer + pe));*/ \
            if (*((TYPE *)hostTestBuffer + pe) != (TYPE)pe + (TYPE)npes) {        \
                errs++;                                                           \
            }                                                                     \
            if (errs > 0) {                                                       \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d\n", mype, pe, errs); \
                status = -1;                                                      \
            }                                                                     \
        }                                                                         \
    }

#define TEST_SHMEM_TYPE_G(type, TYPE)                                                  \
    /*init host test buff and use that to init dev test buff*/                         \
    INIT_DEV_TEST_BUFF(sizeof(TYPE))                                                   \
    /*init host ref buff and use that to init shmem buff to be pulled from*/           \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, 1)                                              \
    nvshmem_barrier_all();                                                             \
    /*DEBUG_PRINT("[%d] before pull %x\n", mype, *((TYPE *)hostTestBuffer + !mype));*/ \
    /*issume shmem_g's*/                                                               \
    for (int pe = 0; pe < npes; pe++) {                                                \
        if (pe != mype) {                                                              \
            *((TYPE *)hostTestBuffer + pe) = nvshmem_##type##_g((TYPE *)buffer, pe);   \
        }                                                                              \
    }                                                                                  \
    /*nvshmem_quiet(); */                                                              \
    nvshmem_barrier_all();                                                             \
    /*copy dev test buff to host test buff and validate*/                              \
    /*DEBUG_PRINT("[%d] after pull %x\n", mype, *((TYPE *)hostTestBuffer + !mype));*/  \
    VALIDATE_G(TYPE, nvshmem_##type##_g)                                               \
    nvshmem_barrier_all();

#define TEST_SHMEMX_TYPE_G_ON_STREAM(type, TYPE)                                       \
    /*init host test buff and use that to init dev test buff*/                         \
    INIT_DEV_TEST_BUFF(sizeof(TYPE))                                                   \
    /*init host ref buff and use that to init shmem buff to be pulled from*/           \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, 1)                                              \
    nvshmem_barrier_all();                                                             \
    /*DEBUG_PRINT("[%d] before pull %x\n", mype, *((TYPE *)hostTestBuffer + !mype));*/ \
    /*issume shmem_g's*/                                                               \
    for (int pe = 0; pe < npes; pe++) {                                                \
        if (pe != mype) {                                                              \
            *((TYPE *)hostTestBuffer + pe) =                                           \
                nvshmemx_##type##_g_on_stream((TYPE *)buffer, pe, cstrm[pe]);          \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                              \
        }                                                                              \
    }                                                                                  \
    /*nvshmem_quiet(); */                                                              \
    nvshmem_barrier_all();                                                             \
    /*copy dev test buff to host test buff and validate*/                              \
    /*DEBUG_PRINT("[%d] after pull %x\n", mype, *((TYPE *)hostTestBuffer + !mype));*/  \
    VALIDATE_G(TYPE, nvshmem_##type##_g)                                               \
    nvshmem_barrier_all();

#define MIN_BYTES_PER_WORD 1

int main(int argc, char **argv) {
    int status = 0;
    int npes, pe;
    size_t nelemsAny =
        64;  // 8MB generates cuMemcpyHtoDAsync error in p2p.cpp:shmemt_p2p_cpu_initiated_write
    size_t nelems = nelemsAny;
    char *buffer = NULL, *devTestBuffer = NULL, *hostTestBuffer = NULL, *hostRefBuffer = NULL;
    int errs = 0;
    cudaStream_t *cstrm;
    read_args(argc, argv);
    if (sizeof(int) > MIN_BYTES_PER_WORD) {
        nelems = (int)std::ceil((float)nelemsAny * (float)MIN_BYTES_PER_WORD / (float)sizeof(int)) *
                 sizeof(int) / MIN_BYTES_PER_WORD;
    }

    init_wrapper(&argc, &argv);

    npes = nvshmem_n_pes();

    cstrm = (cudaStream_t *)malloc(npes * sizeof(cudaStream_t));
    if (!cstrm) {
        ERROR_PRINT("CUDA stream object creation failed \n");
        status = -1;
        goto out;
    }

    for (int i = 0; i < npes; i++) {
        CUDA_CHECK(cudaStreamCreate(&cstrm[i]));
    }
    if (use_mmap) {
        buffer =
            (char *)allocate_mmap_buffer(nelems * sizeof(long double), _mem_handle_type, use_egm);
    } else {
        buffer = (char *)nvshmem_malloc(nelems * sizeof(long double));
    }
    if (!buffer) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }

    hostRefBuffer = (char *)malloc(nelems * sizeof(long double));
    if (!hostRefBuffer) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    hostTestBuffer = (char *)malloc(nelems * sizeof(long double) * npes);
    if (!hostTestBuffer) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

#ifdef REGISTRATION_CACHE_ENABLED
    CUDA_CHECK(cudaMalloc(&devTestBuffer, nelems * sizeof(long double) * npes));
#else
    if (use_mmap) {
        devTestBuffer = (char *)allocate_mmap_buffer(nelems * sizeof(long double) * npes,
                                                     _mem_handle_type, use_egm);
    } else {
        devTestBuffer = (char *)nvshmem_malloc(nelems * sizeof(long double) * npes);
    }
    if (!devTestBuffer) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }
#endif

    TEST_SHMEM_TYPE_G(float, float)
    TEST_SHMEM_TYPE_G(double, double)
    /* char can be signed or unsigned depending on the platform and npes over 63
     * causes us to overflow the signed type. So skip this test if npes is over 64.
     */
    if (npes < 64) {
        TEST_SHMEM_TYPE_G(char, char)
    }
    TEST_SHMEM_TYPE_G(short, short)
    TEST_SHMEM_TYPE_G(int, int)
    TEST_SHMEM_TYPE_G(long, long)
    TEST_SHMEM_TYPE_G(longlong, long long)

    TEST_SHMEMX_TYPE_G_ON_STREAM(float, float)
    TEST_SHMEMX_TYPE_G_ON_STREAM(double, double)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_G_ON_STREAM(char, char)
    }
    TEST_SHMEMX_TYPE_G_ON_STREAM(short, short)
    TEST_SHMEMX_TYPE_G_ON_STREAM(int, int)
    TEST_SHMEMX_TYPE_G_ON_STREAM(long, long)
    TEST_SHMEMX_TYPE_G_ON_STREAM(longlong, long long)

    free(hostRefBuffer);
    free(hostTestBuffer);
#ifdef REGISTRATION_CACHE_ENABLED
    CUDA_CHECK(cudaFree(devTestBuffer));
#else
    if (use_mmap) {
        free_mmap_buffer(devTestBuffer);
    } else {
        nvshmem_free(devTestBuffer);
    }
#endif
    if (use_mmap) {
        free_mmap_buffer(buffer);
    } else {
        nvshmem_free(buffer);
    }
    for (int i = 0; i < npes; i++) {
        CUDA_CHECK(cudaStreamDestroy(cstrm[i]));
    }

out:
    finalize_wrapper();

    return status;
}
