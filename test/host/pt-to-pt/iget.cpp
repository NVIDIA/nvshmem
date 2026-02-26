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

#define INIT_DEV_TEST_BUFF(TYPE, nused, dst)                                                       \
    for (pe = 0; pe < npes; pe++) {                                                                \
        /*if(pe != mype) {*/                                                                       \
        for (size_t i = 0; i < nelems * (dst) * sizeof(long double); i++) {                        \
            *((unsigned char *)hostTestBuffer + (pe * nelems * (dst) * sizeof(long double)) + i) = \
                0;                                                                                 \
        }                                                                                          \
        /*}*/                                                                                      \
    }                                                                                              \
    CUDA_CHECK(cudaMemcpy(devTestBuffer, hostTestBuffer,                                           \
                          nelems *(dst) * sizeof(long double) * npes, cudaMemcpyHostToDevice));    \
    for (pe = 0; pe < npes; pe++) {                                                                \
        for (size_t i = 0; i < nused; i++) {                                                       \
            *((TYPE *)hostTestBuffer + (pe * nused * (dst)) + (i * (dst))) = (TYPE)0xab;           \
        }                                                                                          \
        CUDA_CHECK(cudaMemcpy((TYPE *)devTestBuffer + (pe * nused * (dst)),                        \
                              (TYPE *)hostTestBuffer + (pe * nused * (dst)),                       \
                              nused * (dst) * sizeof(TYPE), cudaMemcpyHostToDevice));              \
    }

#define INIT_BYTES_DEV_TEST_BUFF(bytesPerWord, nused, dst)                                         \
    for (pe = 0; pe < npes; pe++) {                                                                \
        /*if(pe != mype) {*/                                                                       \
        for (size_t i = 0; i < nelems * (dst) * sizeof(long double); i++) {                        \
            *((unsigned char *)hostTestBuffer + (pe * nelems * (dst) * sizeof(long double)) + i) = \
                0;                                                                                 \
        }                                                                                          \
        /*}*/                                                                                      \
    }                                                                                              \
    CUDA_CHECK(cudaMemcpy(devTestBuffer, hostTestBuffer,                                           \
                          nelems *(dst) * sizeof(long double) * npes, cudaMemcpyHostToDevice));    \
    for (pe = 0; pe < npes; pe++) {                                                                \
        for (size_t i = 0; i < nused; i++) {                                                       \
            for (int j = 0; j < bytesPerWord; j++) {                                               \
                *((unsigned char *)hostTestBuffer + (pe * nused * (dst)*bytesPerWord) +            \
                  (i * (dst)*bytesPerWord) + j) = (unsigned char)0xab;                             \
            }                                                                                      \
        }                                                                                          \
        CUDA_CHECK(cudaMemcpy((unsigned char *)devTestBuffer + (pe * nused * (dst)*bytesPerWord),  \
                              (unsigned char *)hostTestBuffer + (pe * nused * (dst)*bytesPerWord), \
                              nused * (dst)*bytesPerWord, cudaMemcpyHostToDevice));                \
    }

#define INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nused, sst)                                \
    for (size_t i = 0; i < nelems * (sst) * sizeof(long double); i++) {               \
        *((unsigned char *)hostRefBuffer + i) = 0;                                    \
    }                                                                                 \
    CUDA_CHECK(cudaMemcpy(buffer, hostRefBuffer, nelems *(sst) * sizeof(long double), \
                          cudaMemcpyHostToDevice));                                   \
    for (size_t i = 0; i < nused; i++) {                                              \
        *((TYPE *)hostRefBuffer + (i * (sst))) = (TYPE)mype + (TYPE)npes;             \
    }                                                                                 \
    CUDA_CHECK(                                                                       \
        cudaMemcpy(buffer, hostRefBuffer, nused *(sst) * sizeof(TYPE), cudaMemcpyHostToDevice));

#define INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(bytesPerWord, nused, sst)                  \
    for (size_t i = 0; i < nelems * (sst) * sizeof(long double); i++) {               \
        *((unsigned char *)hostRefBuffer + i) = 0;                                    \
    }                                                                                 \
    CUDA_CHECK(cudaMemcpy(buffer, hostRefBuffer, nelems *(sst) * sizeof(long double), \
                          cudaMemcpyHostToDevice));                                   \
    for (size_t i = 0; i < nused; i++) {                                              \
        for (int j = 0; j < bytesPerWord; j++) {                                      \
            *((unsigned char *)hostRefBuffer + (i * (sst)*bytesPerWord) + j) =        \
                (unsigned char)mype + (unsigned char)npes;                            \
        }                                                                             \
    }                                                                                 \
    CUDA_CHECK(                                                                       \
        cudaMemcpy(buffer, hostRefBuffer, nused *(sst)*bytesPerWord, cudaMemcpyHostToDevice));

#define VALIDATE_IGET(TYPE, nused, dst, apiname)                                                   \
    CUDA_CHECK(cudaMemcpy(hostTestBuffer, devTestBuffer, nused * sizeof(long double) * (dst)*npes, \
                          cudaMemcpyDeviceToHost));                                                \
    for (pe = 0; pe < npes; pe++) {                                                                \
        errs = 0;                                                                                  \
        if (pe != mype) {                                                                          \
            for (size_t i = 0; i < nused; i++) {                                                   \
                if (*((TYPE *)hostTestBuffer + pe * nused * (dst) + i * (dst)) !=                  \
                    (TYPE)pe + (TYPE)npes) {                                                       \
                    errs++;                                                                        \
                }                                                                                  \
            }                                                                                      \
            if (errs > 0) {                                                                        \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d\n", mype, pe, errs);                  \
                status = -1;                                                                       \
            }                                                                                      \
        }                                                                                          \
    }

#define VALIDATE_BYTES_IGET(bytesPerWord, nused, dst, apiname)                                     \
    CUDA_CHECK(cudaMemcpy(hostTestBuffer, devTestBuffer, nused * sizeof(long double) * (dst)*npes, \
                          cudaMemcpyDeviceToHost));                                                \
    for (pe = 0; pe < npes; pe++) {                                                                \
        errs = 0;                                                                                  \
        if (pe != mype) {                                                                          \
            for (size_t i = 0; i < nused; i++) {                                                   \
                for (int j = 0; j < bytesPerWord; j++) {                                           \
                    if (*((unsigned char *)hostTestBuffer + (pe * nused * (dst)*bytesPerWord) +    \
                          (i * (dst)*bytesPerWord) + j) !=                                         \
                        (unsigned char)pe + (unsigned char)npes) {                                 \
                        errs++;                                                                    \
                    }                                                                              \
                    break;                                                                         \
                }                                                                                  \
            }                                                                                      \
            if (errs > 0) {                                                                        \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d\n", mype, pe, errs);                  \
                status = -1;                                                                       \
            }                                                                                      \
        }                                                                                          \
    }

#define TEST_SHMEM_IGET_WORDSIZE(bitsPerWord, bytesPerWord, nused, dst, sst)                \
    /*init host test buff and use that to init dev test buff*/                              \
    INIT_BYTES_DEV_TEST_BUFF(bytesPerWord, nused, dst)                                      \
    /*init host ref buff and use that to init shmem buff to be pulled from*/                \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(bytesPerWord, nused, sst)                            \
    nvshmem_barrier_all();                                                                  \
    /*issue shmem iget's*/                                                                  \
    for (int pe = 0; pe < npes; pe++) {                                                     \
        if (pe != mype) {                                                                   \
            nvshmem_iget##bitsPerWord(                                                      \
                (void *)((unsigned char *)devTestBuffer + nused * bytesPerWord * (dst)*pe), \
                (void *)buffer, dst, sst, nused, pe);                                       \
        }                                                                                   \
    }                                                                                       \
    /*nvshmem_quiet();*/                                                                    \
    nvshmem_barrier_all();                                                                  \
    /*copy dev test buff to host test buff and validate*/                                   \
    VALIDATE_BYTES_IGET(bytesPerWord, nused, dst, nvshmem_iget##bitsPerWord)                \
    nvshmem_barrier_all();

#define TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(bitsPerWord, bytesPerWord, nused, dst, sst)     \
    /*init host test buff and use that to init dev test buff*/                              \
    INIT_BYTES_DEV_TEST_BUFF(bytesPerWord, nused, dst)                                      \
    /*init host ref buff and use that to init shmem buff to be pulled from*/                \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(bytesPerWord, nused, sst)                            \
    nvshmem_barrier_all();                                                                  \
    /*issue shmem iget's*/                                                                  \
    for (int pe = 0; pe < npes; pe++) {                                                     \
        if (pe != mype) {                                                                   \
            nvshmemx_iget##bitsPerWord##_on_stream(                                         \
                (void *)((unsigned char *)devTestBuffer + nused * bytesPerWord * (dst)*pe), \
                (void *)buffer, dst, sst, nused, pe, cstrm[pe]);                            \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                   \
        }                                                                                   \
    }                                                                                       \
    /*nvshmem_quiet();*/                                                                    \
    nvshmem_barrier_all();                                                                  \
    /*copy dev test buff to host test buff and validate*/                                   \
    VALIDATE_BYTES_IGET(bytesPerWord, nused, dst, nvshmem_iget##bitsPerWord)                \
    nvshmem_barrier_all();

#define TEST_SHMEM_TYPE_IGET(type, TYPE, nused, dst, sst)                                          \
    /*init host test buff and use that to init dev test buff*/                                     \
    INIT_DEV_TEST_BUFF(TYPE, nused, dst)                                                           \
    /*init host ref buff and use that to init shmem buff to be pulled from*/                       \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nused, sst)                                                 \
    nvshmem_barrier_all();                                                                         \
    /*issue shmem iget's*/                                                                         \
    for (pe = 0; pe < npes; pe++) {                                                                \
        if (pe != mype) {                                                                          \
            nvshmem_##type##_iget(((TYPE *)devTestBuffer + nused * (dst)*pe), (TYPE *)buffer, dst, \
                                  sst, nused, pe);                                                 \
        }                                                                                          \
    }                                                                                              \
    /*nvshmem_quiet();*/                                                                           \
    nvshmem_barrier_all();                                                                         \
    /*copy dev test buff to host test buff and validate*/                                          \
    VALIDATE_IGET(TYPE, nused, dst, nvshmem_##type##_iget)                                         \
    nvshmem_barrier_all();

#define TEST_SHMEMX_TYPE_IGET_ON_STREAM(type, TYPE, nused, dst, sst)                          \
    /*init host test buff and use that to init dev test buff*/                                \
    INIT_DEV_TEST_BUFF(TYPE, nused, dst)                                                      \
    /*init host ref buff and use that to init shmem buff to be pulled from*/                  \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nused, sst)                                            \
    nvshmem_barrier_all();                                                                    \
    /*issue shmem iget's*/                                                                    \
    for (pe = 0; pe < npes; pe++) {                                                           \
        if (pe != mype) {                                                                     \
            nvshmemx_##type##_iget_on_stream(((TYPE *)devTestBuffer + nused * (dst)*pe),      \
                                             (TYPE *)buffer, dst, sst, nused, pe, cstrm[pe]); \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                     \
        }                                                                                     \
    }                                                                                         \
    /*nvshmem_quiet();*/                                                                      \
    nvshmem_barrier_all();                                                                    \
    /*copy dev test buff to host test buff and validate*/                                     \
    VALIDATE_IGET(TYPE, nused, dst, nvshmem_##type##_iget)                                    \
    nvshmem_barrier_all();

#define MIN_BYTES_PER_WORD 1
#define MAX_STRIDE 2

int main(int argc, char **argv) {
    int status = 0;
    int npes, pe;
    size_t nelemsAny =
        64;  // 8MB generates cuMemcpyHtoDAsync error in p2p.cpp:shmemt_p2p_cpu_initiated_write
    size_t nelems = nelemsAny;
    char *buffer = NULL, *devTestBuffer = NULL, *hostTestBuffer = NULL, *hostRefBuffer = NULL;
    int errs = 0;
    const size_t count = 2, dstride = MAX_STRIDE, sstride = MAX_STRIDE;
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
        buffer = (char *)allocate_mmap_buffer(nelems * sizeof(long double) * (MAX_STRIDE),
                                              _mem_handle_type, use_egm);
        DEBUG_PRINT("Allocating mmap buffer\n");
    } else {
        buffer = (char *)nvshmem_malloc(nelems * sizeof(long double) * (MAX_STRIDE));
    }
    if (!buffer) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }

    hostRefBuffer = (char *)malloc(nelems * sizeof(long double) * (MAX_STRIDE));
    if (!hostRefBuffer) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    hostTestBuffer = (char *)malloc(nelems * sizeof(long double) * (MAX_STRIDE)*npes);
    if (!hostTestBuffer) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    CUDA_CHECK(cudaMalloc(&devTestBuffer, nelems * sizeof(long double) * (MAX_STRIDE)*npes));

    TEST_SHMEM_TYPE_IGET(float, float, count, dstride, sstride)
    TEST_SHMEM_TYPE_IGET(double, double, count, dstride, sstride)
    /* char can be signed or unsigned depending on the platform and npes over 63
     * causes us to overflow the signed type. So skip this test if npes is over 64.
     */
    if (npes < 64) {
        TEST_SHMEM_TYPE_IGET(char, char, count, dstride, sstride)
    }
    TEST_SHMEM_TYPE_IGET(short, short, count, dstride, sstride)
    TEST_SHMEM_TYPE_IGET(int, int, count, dstride, sstride)
    TEST_SHMEM_TYPE_IGET(long, long, count, dstride, sstride)
    TEST_SHMEM_TYPE_IGET(longlong, long long, count, dstride, sstride)

    TEST_SHMEMX_TYPE_IGET_ON_STREAM(float, float, count, dstride, sstride)
    TEST_SHMEMX_TYPE_IGET_ON_STREAM(double, double, count, dstride, sstride)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_IGET_ON_STREAM(char, char, count, dstride, sstride)
    }
    TEST_SHMEMX_TYPE_IGET_ON_STREAM(short, short, count, dstride, sstride)
    TEST_SHMEMX_TYPE_IGET_ON_STREAM(int, int, count, dstride, sstride)
    TEST_SHMEMX_TYPE_IGET_ON_STREAM(long, long, count, dstride, sstride)
    TEST_SHMEMX_TYPE_IGET_ON_STREAM(longlong, long long, count, dstride, sstride)

    TEST_SHMEM_IGET_WORDSIZE(8, 1, count, dstride, sstride)
    TEST_SHMEM_IGET_WORDSIZE(16, 2, count, dstride, sstride)
    TEST_SHMEM_IGET_WORDSIZE(32, 4, count, dstride, sstride)
    TEST_SHMEM_IGET_WORDSIZE(64, 8, count, dstride, sstride)
    TEST_SHMEM_IGET_WORDSIZE(128, 16, count, dstride, sstride)

    TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(8, 1, count, dstride, sstride)
    TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(16, 2, count, dstride, sstride)
    TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(32, 4, count, dstride, sstride)
    TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(64, 8, count, dstride, sstride)
    TEST_SHMEMX_IGET_WORDSIZE_ON_STREAM(128, 16, count, dstride, sstride)

    free(hostRefBuffer);
    free(hostTestBuffer);
    CUDA_CHECK(cudaFree(devTestBuffer));
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
