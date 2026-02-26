/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <getopt.h>
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

#define INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nbytes)                                                \
    for (size_t i = 0; i < nelems * sizeof(long double); i++) {                                   \
        *((unsigned char *)hostRefBuffer + i) = 0;                                                \
    }                                                                                             \
    CUDA_CHECK(                                                                                   \
        cudaMemcpy(buffer, hostRefBuffer, sizeof(long double) * nelems, cudaMemcpyHostToDevice)); \
    for (size_t i = 0; i < nbytes; i++) {                                                         \
        *((unsigned char *)hostRefBuffer + i) = (unsigned char)mype + (unsigned char)npes;        \
    }                                                                                             \
    CUDA_CHECK(cudaMemcpy(buffer, hostRefBuffer, nbytes, cudaMemcpyHostToDevice));

#define VALIDATE_GET(TYPE, nelems, apiname)                                                      \
    for (pe = 0; pe < npes; pe++) {                                                              \
        errs = 0;                                                                                \
        if (pe != mype) {                                                                        \
            CUDA_CHECK(cudaMemcpy(hostTestBuffer + pe * nelems * sizeof(TYPE),                   \
                                  devTestBuffer + pe * nelems * sizeof(TYPE),                    \
                                  nelems * sizeof(TYPE), cudaMemcpyDeviceToHost));               \
            for (size_t i = 0; i < nelems; i++) {                                                \
                /*DEBUG_PRINT("[%d] %llf\n", mype, *((TYPE *)hostTestBuffer + pe*nelems + i));*/ \
                if (*((TYPE *)hostTestBuffer + pe * nelems + i) != (TYPE)pe + (TYPE)npes) {      \
                    errs++;                                                                      \
                }                                                                                \
            }                                                                                    \
            if (errs > 0) {                                                                      \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d\n", mype, pe, errs);                \
                status = -1;                                                                     \
            }                                                                                    \
        }                                                                                        \
    }

#define VALIDATE_BYTES_GET(bytesPerWord, nelems, apiname)                                     \
    for (pe = 0; pe < npes; pe++) {                                                           \
        errs = 0;                                                                             \
        if (pe != mype) {                                                                     \
            CUDA_CHECK(cudaMemcpy(hostTestBuffer + pe * nelems * bytesPerWord,                \
                                  devTestBuffer + pe * nelems * bytesPerWord,                 \
                                  nelems * bytesPerWord, cudaMemcpyDeviceToHost));            \
            for (size_t i = 0; i < nelems; i++) {                                             \
                for (int j = 0; j < bytesPerWord; j++) {                                      \
                    /*DEBUG_PRINT("[%d] %llf\n", mype, *((unsigned char *)hostTestBuffer +    \
                     * pe*nelems*bytesPerWord + i*bytesPerWord + j));*/                       \
                    if (*((unsigned char *)hostTestBuffer + pe * nelems * bytesPerWord +      \
                          i * bytesPerWord + j) != (unsigned char)pe + (unsigned char)npes) { \
                        errs++;                                                               \
                    }                                                                         \
                }                                                                             \
            }                                                                                 \
            if (errs > 0) {                                                                   \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d\n", mype, pe, errs);             \
                status = -1;                                                                  \
            }                                                                                 \
        }                                                                                     \
    }

#define TEST_SHMEM_GETMEM(nbytes)                                                                 \
    /*init host test buff and use that to init dev test buff*/                                    \
    INIT_DEV_TEST_BUFF(nbytes)                                                                    \
    /*init host ref buff and use that to init shmem buff*/                                        \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nbytes)                                                    \
    nvshmem_barrier_all();                                                                        \
    /*issue shmem get's*/                                                                         \
    for (int pe = 0; pe < npes; pe++) {                                                           \
        if (pe != mype) {                                                                         \
            nvshmem_getmem((void *)((char *)devTestBuffer + pe * nbytes), (void *)buffer, nbytes, \
                           pe);                                                                   \
        }                                                                                         \
    }                                                                                             \
    /*nvshmem_quiet();*/                                                                          \
    nvshmem_barrier_all();                                                                        \
    /*copy dev test buff to host test buff and validate*/                                         \
    VALIDATE_BYTES_GET(1, nbytes, nvshmem_getmem)                                                 \
    nvshmem_barrier_all();

#define TEST_SHMEMX_GETMEM_ON_STREAM(nbytes)                                         \
    /*init host test buff and use that to init dev test buff*/                       \
    INIT_DEV_TEST_BUFF(nbytes)                                                       \
    /*init host ref buff and use that to init shmem buff*/                           \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nbytes)                                       \
    nvshmem_barrier_all();                                                           \
    /*issue shmem get's*/                                                            \
    for (int pe = 0; pe < npes; pe++) {                                              \
        if (pe != mype) {                                                            \
            nvshmemx_getmem_on_stream((void *)((char *)devTestBuffer + pe * nbytes), \
                                      (void *)buffer, nbytes, pe, cstrm[pe]);        \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                            \
        }                                                                            \
    }                                                                                \
    /*nvshmem_quiet();*/                                                             \
    nvshmem_barrier_all();                                                           \
    /*copy dev test buff to host test buff and validate*/                            \
    VALIDATE_BYTES_GET(1, nbytes, nvshmem_getmem)                                    \
    nvshmem_barrier_all();

#define TEST_SHMEM_TYPE_GET(type, TYPE, nelems)                                                 \
    /*init host test buff and use that to init dev test buff*/                                  \
    INIT_DEV_TEST_BUFF(nelems * sizeof(TYPE))                                                   \
    /*init host ref buff and use that to init shmem buff*/                                      \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nelems)                                                  \
    nvshmem_barrier_all();                                                                      \
    /*issue shmem get's*/                                                                       \
    for (int pe = 0; pe < npes; pe++) {                                                         \
        if (pe != mype) {                                                                       \
            nvshmem_##type##_get(((TYPE *)devTestBuffer + nelems * pe), (TYPE *)buffer, nelems, \
                                 pe);                                                           \
        }                                                                                       \
    }                                                                                           \
    /*nvshmem_quiet();*/                                                                        \
    nvshmem_barrier_all();                                                                      \
    /*copy dev test buff to host test buff and validate*/                                       \
    VALIDATE_GET(TYPE, nelems, nvshmem_##type##_get)                                            \
    nvshmem_barrier_all();

#define TEST_SHMEMX_TYPE_GET_ON_STREAM(type, TYPE, nelems)                                         \
    /*init host test buff and use that to init dev test buff*/                                     \
    INIT_DEV_TEST_BUFF(nelems * sizeof(TYPE))                                                      \
    /*init host ref buff and use that to init shmem buff*/                                         \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nelems)                                                     \
    nvshmem_barrier_all();                                                                         \
    /*issue shmem get's*/                                                                          \
    for (int pe = 0; pe < npes; pe++) {                                                            \
        if (pe != mype) {                                                                          \
            nvshmemx_##type##_get_on_stream(((TYPE *)devTestBuffer + nelems * pe), (TYPE *)buffer, \
                                            nelems, pe, cstrm[pe]);                                \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                          \
        }                                                                                          \
    }                                                                                              \
    /*nvshmem_quiet();*/                                                                           \
    nvshmem_barrier_all();                                                                         \
    /*copy dev test buff to host test buff and validate*/                                          \
    VALIDATE_GET(TYPE, nelems, nvshmem_##type##_get)                                               \
    nvshmem_barrier_all();

#define TEST_SHMEM_GET_WORDSIZE(bitsPerWord, bytesPerWord, nelems)                                 \
    /*init host test buff and use that to init dev test buff*/                                     \
    INIT_DEV_TEST_BUFF(nelems *bytesPerWord)                                                       \
    /*init host ref buff and use that to init shmem buff*/                                         \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nelems *bytesPerWord)                                       \
    nvshmem_barrier_all();                                                                         \
    /*issue shmem get's*/                                                                          \
    for (int pe = 0; pe < npes; pe++) {                                                            \
        if (pe != mype) {                                                                          \
            nvshmem_get##bitsPerWord((void *)((char *)devTestBuffer + nelems * bytesPerWord * pe), \
                                     (void *)buffer, nelems, pe);                                  \
        }                                                                                          \
    }                                                                                              \
    /*nvshmem_quiet();*/                                                                           \
    nvshmem_barrier_all();                                                                         \
    /*copy dev test buff to host test buff and validate*/                                          \
    VALIDATE_BYTES_GET(bytesPerWord, nelems, nvshmem_get##bitsPerWord)                             \
    nvshmem_barrier_all();

#define TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(bitsPerWord, bytesPerWord, nelems)                 \
    /*init host test buff and use that to init dev test buff*/                                \
    INIT_DEV_TEST_BUFF(nelems *bytesPerWord)                                                  \
    /*init host ref buff and use that to init shmem buff*/                                    \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nelems *bytesPerWord)                                  \
    nvshmem_barrier_all();                                                                    \
    /*issue shmem get's*/                                                                     \
    for (int pe = 0; pe < npes; pe++) {                                                       \
        if (pe != mype) {                                                                     \
            nvshmemx_get##bitsPerWord##_on_stream(                                            \
                (void *)((char *)devTestBuffer + nelems * bytesPerWord * pe), (void *)buffer, \
                nelems, pe, cstrm[pe]);                                                       \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                     \
        }                                                                                     \
    }                                                                                         \
    /*nvshmem_quiet();*/                                                                      \
    nvshmem_barrier_all();                                                                    \
    /*copy dev test buff to host test buff and validate*/                                     \
    VALIDATE_BYTES_GET(bytesPerWord, nelems, nvshmem_get##bitsPerWord)                        \
    nvshmem_barrier_all();

#define TEST_SHMEM_TYPE_GET_NBI(type, TYPE, nelems)                                         \
    /*init host test buff and use that to init dev test buff*/                              \
    INIT_DEV_TEST_BUFF(nelems * sizeof(TYPE))                                               \
    /*init host ref buff and use that to init shmem buff*/                                  \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nelems)                                              \
    nvshmem_barrier_all();                                                                  \
    /*issue shmem get's*/                                                                   \
    for (int pe = 0; pe < npes; pe++) {                                                     \
        if (pe != mype) {                                                                   \
            nvshmem_##type##_get_nbi(((TYPE *)devTestBuffer + nelems * pe), (TYPE *)buffer, \
                                     nelems, pe);                                           \
        }                                                                                   \
    }                                                                                       \
    nvshmem_quiet();                                                                        \
    nvshmem_barrier_all();                                                                  \
    /*copy dev test buff to host test buff and validate*/                                   \
    VALIDATE_GET(TYPE, nelems, nvshmem_##type##_get_nbi)                                    \
    nvshmem_barrier_all();

#define TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(type, TYPE, nelems)                          \
    /*init host test buff and use that to init dev test buff*/                          \
    INIT_DEV_TEST_BUFF(nelems * sizeof(TYPE))                                           \
    /*init host ref buff and use that to init shmem buff*/                              \
    INIT_SHMEM_REF_BUFF_FOR_PULL(TYPE, nelems)                                          \
    nvshmem_barrier_all();                                                              \
    /*issue shmem get's*/                                                               \
    for (int pe = 0; pe < npes; pe++) {                                                 \
        if (pe != mype) {                                                               \
            nvshmemx_##type##_get_nbi_on_stream(((TYPE *)devTestBuffer + nelems * pe),  \
                                                (TYPE *)buffer, nelems, pe, cstrm[pe]); \
            nvshmemx_quiet_on_stream(cstrm[pe]);                                        \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                               \
        }                                                                               \
    }                                                                                   \
    nvshmem_quiet();                                                                    \
    nvshmem_barrier_all();                                                              \
    /*copy dev test buff to host test buff and validate*/                               \
    VALIDATE_GET(TYPE, nelems, nvshmem_##type##_get_nbi)                                \
    nvshmem_barrier_all();

#define TEST_SHMEM_GET_WORDSIZE_NBI(bitsPerWord, bytesPerWord, nelems)                        \
    /*init host test buff and use that to init dev test buff*/                                \
    INIT_DEV_TEST_BUFF(nelems *bytesPerWord)                                                  \
    /*init host ref buff and use that to init shmem buff*/                                    \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nelems *bytesPerWord)                                  \
    nvshmem_barrier_all();                                                                    \
    /*issue shmem get's*/                                                                     \
    for (pe = 0; pe < npes; pe++) {                                                           \
        if (pe != mype) {                                                                     \
            nvshmem_get##bitsPerWord##_nbi(                                                   \
                (void *)((char *)devTestBuffer + nelems * pe * bytesPerWord), (void *)buffer, \
                nelems, pe);                                                                  \
        }                                                                                     \
    }                                                                                         \
    nvshmem_quiet();                                                                          \
    nvshmem_barrier_all();                                                                    \
    /*copy dev test buff to host test buff and validate*/                                     \
    VALIDATE_BYTES_GET(bytesPerWord, nelems, nvshmem_get##bitsPerWord##_nbi)                  \
    nvshmem_barrier_all();

#define TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(bitsPerWord, bytesPerWord, nelems)             \
    /*init host test buff and use that to init dev test buff*/                                \
    INIT_DEV_TEST_BUFF(nelems *bytesPerWord)                                                  \
    /*init host ref buff and use that to init shmem buff*/                                    \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nelems *bytesPerWord)                                  \
    nvshmem_barrier_all();                                                                    \
    /*issue shmem get's*/                                                                     \
    for (pe = 0; pe < npes; pe++) {                                                           \
        if (pe != mype) {                                                                     \
            nvshmemx_get##bitsPerWord##_nbi_on_stream(                                        \
                (void *)((char *)devTestBuffer + nelems * pe * bytesPerWord), (void *)buffer, \
                nelems, pe, cstrm[pe]);                                                       \
            nvshmemx_quiet_on_stream(cstrm[pe]);                                              \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                     \
        }                                                                                     \
    }                                                                                         \
    nvshmem_quiet();                                                                          \
    nvshmem_barrier_all();                                                                    \
    /*copy dev test buff to host test buff and validate*/                                     \
    VALIDATE_BYTES_GET(bytesPerWord, nelems, nvshmem_get##bitsPerWord##_nbi)                  \
    nvshmem_barrier_all();

#define TEST_SHMEM_GETMEM_NBI(nbytes)                                                         \
    /*init host test buff and use that to init dev test buff*/                                \
    INIT_DEV_TEST_BUFF(nbytes)                                                                \
    /*init host ref buff and use that to init shmem buff*/                                    \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nbytes)                                                \
    nvshmem_barrier_all();                                                                    \
    /*issue shmem get's*/                                                                     \
    for (int pe = 0; pe < npes; pe++) {                                                       \
        if (pe != mype) {                                                                     \
            nvshmem_getmem_nbi((void *)((char *)devTestBuffer + pe * nbytes), (void *)buffer, \
                               nbytes, pe);                                                   \
        }                                                                                     \
    }                                                                                         \
    nvshmem_quiet();                                                                          \
    nvshmem_barrier_all();                                                                    \
    /*copy dev test buff to host test buff and validate*/                                     \
    VALIDATE_BYTES_GET(1, nbytes, nvshmem_getmem)                                             \
    nvshmem_barrier_all();

#define TEST_SHMEMX_GETMEM_NBI_ON_STREAM(nbytes)                                         \
    /*init host test buff and use that to init dev test buff*/                           \
    INIT_DEV_TEST_BUFF(nbytes)                                                           \
    /*init host ref buff and use that to init shmem buff*/                               \
    INIT_BYTES_SHMEM_REF_BUFF_FOR_PULL(nbytes)                                           \
    nvshmem_barrier_all();                                                               \
    /*issue shmem get's*/                                                                \
    for (int pe = 0; pe < npes; pe++) {                                                  \
        if (pe != mype) {                                                                \
            nvshmemx_getmem_nbi_on_stream((void *)((char *)devTestBuffer + pe * nbytes), \
                                          (void *)buffer, nbytes, pe, cstrm[pe]);        \
            nvshmemx_quiet_on_stream(cstrm[pe]);                                         \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                \
        }                                                                                \
    }                                                                                    \
    /*nvshmem_quiet();*/                                                                 \
    nvshmem_barrier_all();                                                               \
    /*copy dev test buff to host test buff and validate*/                                \
    VALIDATE_BYTES_GET(1, nbytes, nvshmem_getmem)                                        \
    nvshmem_barrier_all();

#define MIN_BYTES_PER_WORD 1
#define MAX_ITER 1
#define MAX_NPES 8
#define MAX_WORDSIZE 16
#define MAX_SHEAP_BYTES (1024 * 1024 * 1024)
#define LIB_SHEAP_FACTOR 2
#define MAX_ELEMS_FULL (MAX_SHEAP_BYTES / (MAX_WORDSIZE * MAX_NPES * 2))
#undef MAX_ELEMS
#define MAX_ELEMS (MAX_ELEMS_FULL - (MAX_ELEMS_FULL / LIB_SHEAP_FACTOR))

int main(int argc, char *argv[]) {
    int status = 0;
    int npes, pe;
    size_t nelemsAny = MAX_ELEMS;
    // 8MB generates cuMemcpyHtoDAsync error in p2p.cpp:shmemt_p2p_cpu_initiated_write
    size_t nelems = nelemsAny;
    char *buffer = NULL, *devTestBuffer = NULL, *hostTestBuffer = NULL, *hostRefBuffer = NULL;
    int errs = 0;
    const size_t count = 2;
    cudaStream_t *cstrm;
    bool asymmetric_local_buffer = false;
    bool register_local_buffer = false;
    if (sizeof(int) > MIN_BYTES_PER_WORD) {
        nelems = (int)std::ceil((float)nelemsAny * (float)MIN_BYTES_PER_WORD / (float)sizeof(int)) *
                 sizeof(int) / MIN_BYTES_PER_WORD;
    }

    init_wrapper(&argc, &argv);

    while (1) {
        int c;
        c = getopt(argc, argv, "arh");
        if (c == -1) break;

        switch (c) {
            case 'a':
                asymmetric_local_buffer = true;
                break;
            case 'r':
                register_local_buffer = true;
                break;
            default:
            case 'h':
                printf("pass -a to allocate an asymmetric local buffer.\n");
                printf(
                    "pass -r to register an asymmetric local buffer. only valid when combined with "
                    "-a.\n");
                goto out;
        }
    }

    mype = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
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

    if (asymmetric_local_buffer) {
        CUDA_CHECK(cudaMalloc(&devTestBuffer, nelems * sizeof(long double) * npes));
        if (register_local_buffer) {
            status = nvshmemx_buffer_register(devTestBuffer, nelems * sizeof(long double) * npes);
            if (status) {
                ERROR_PRINT("nvshmemx_buffer_register failed.\n");
                goto out;
            }
        }
    } else {
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
    }

    TEST_SHMEM_TYPE_GET(float, float, count)
    TEST_SHMEM_TYPE_GET(double, double, count)
    /* char can be signed or unsigned depending on the platform and npes over 63
     * causes us to overflow the signed type. So skip this test if npes is over 64.
     */
    if (npes < 64) {
        TEST_SHMEM_TYPE_GET(char, char, count)
    }
    TEST_SHMEM_TYPE_GET(short, short, count)
    TEST_SHMEM_TYPE_GET(int, int, count)
    TEST_SHMEM_TYPE_GET(long, long, count)
    TEST_SHMEM_TYPE_GET(longlong, long long, count)

    TEST_SHMEM_GET_WORDSIZE(8, 1, count)
    TEST_SHMEM_GET_WORDSIZE(16, 2, count)
    TEST_SHMEM_GET_WORDSIZE(32, 4, count)
    TEST_SHMEM_GET_WORDSIZE(64, 8, count)
    TEST_SHMEM_GET_WORDSIZE(128, 16, count)

    TEST_SHMEM_GETMEM(sizeof(int))

    TEST_SHMEM_TYPE_GET_NBI(float, float, count)
    TEST_SHMEM_TYPE_GET_NBI(double, double, count)
    if (npes < 64) {
        TEST_SHMEM_TYPE_GET_NBI(char, char, count)
    }
    TEST_SHMEM_TYPE_GET_NBI(short, short, count)
    TEST_SHMEM_TYPE_GET_NBI(int, int, count)
    TEST_SHMEM_TYPE_GET_NBI(long, long, count)
    TEST_SHMEM_TYPE_GET_NBI(longlong, long long, count)

    TEST_SHMEM_GET_WORDSIZE_NBI(8, 1, count)
    TEST_SHMEM_GET_WORDSIZE_NBI(16, 2, count)
    TEST_SHMEM_GET_WORDSIZE_NBI(32, 4, count)
    TEST_SHMEM_GET_WORDSIZE_NBI(64, 8, count)
    TEST_SHMEM_GET_WORDSIZE_NBI(128, 16, count)

    TEST_SHMEM_GETMEM_NBI(sizeof(int))

    TEST_SHMEMX_TYPE_GET_ON_STREAM(float, float, count)
    TEST_SHMEMX_TYPE_GET_ON_STREAM(double, double, count)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_GET_ON_STREAM(char, char, count)
    }
    TEST_SHMEMX_TYPE_GET_ON_STREAM(short, short, count)
    TEST_SHMEMX_TYPE_GET_ON_STREAM(int, int, count)
    TEST_SHMEMX_TYPE_GET_ON_STREAM(long, long, count)
    TEST_SHMEMX_TYPE_GET_ON_STREAM(longlong, long long, count)

    TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(8, 1, count)
    TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(16, 2, count)
    TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(32, 4, count)
    TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(64, 8, count)
    TEST_SHMEMX_GET_WORDSIZE_ON_STREAM(128, 16, count)

    TEST_SHMEMX_GETMEM_ON_STREAM(sizeof(int))

    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(float, float, count)
    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(double, double, count)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(char, char, count)
    }
    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(short, short, count)
    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(int, int, count)
    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(long, long, count)
    TEST_SHMEMX_TYPE_GET_NBI_ON_STREAM(longlong, long long, count)

    TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(8, 1, count)
    TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(16, 2, count)
    TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(32, 4, count)
    TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(64, 8, count)
    TEST_SHMEMX_GET_WORDSIZE_NBI_ON_STREAM(128, 16, count)

    TEST_SHMEMX_GETMEM_NBI_ON_STREAM(sizeof(int))

    free(hostRefBuffer);
    free(hostTestBuffer);
    if (asymmetric_local_buffer) {
        nvshmemx_buffer_unregister(devTestBuffer);
        CUDA_CHECK(cudaFree(devTestBuffer));
    } else {
        if (use_mmap) {
            free_mmap_buffer(devTestBuffer);
        } else {
            nvshmem_free(devTestBuffer);
        }
    }

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
