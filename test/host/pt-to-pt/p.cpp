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
#include <unistd.h>

#define INIT_SHMEM_TEST_BUFF(nbytes)                                                           \
    for (pe = 0; pe < npes; pe++) {                                                            \
        /*if(pe != mype) {*/                                                                   \
        for (size_t i = 0; i < nbytes; i++) {                                                  \
            *((char *)hostTestBuffer + pe * nbytes + i) = 0xAB;                                \
        }                                                                                      \
        CUDA_CHECK(cudaMemcpyAsync(buffer + pe * nbytes, hostTestBuffer + pe * nbytes, nbytes, \
                                   cudaMemcpyHostToDevice, cstrm[pe]));                        \
        CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                          \
        /*}*/                                                                                  \
    }

#define INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, nused)                                           \
    for (size_t i = 0; i < nelems * sizeof(long double); i++) {                           \
        *((char *)hostRefBuffer + i) = 0;                                                 \
    }                                                                                     \
    CUDA_CHECK(cudaMemcpyAsync(devRefBuffer, hostRefBuffer, sizeof(long double) * nelems, \
                               cudaMemcpyHostToDevice, cstrm[mype]));                     \
    CUDA_CHECK(cudaStreamSynchronize(cstrm[mype]));                                       \
    for (size_t i = 0; i < nused; i++) {                                                  \
        *((TYPE *)hostRefBuffer + i) = (TYPE)mype + (TYPE)npes;                           \
    }                                                                                     \
    CUDA_CHECK(cudaMemcpyAsync(devRefBuffer, hostRefBuffer, sizeof(TYPE) * nused,         \
                               cudaMemcpyHostToDevice, cstrm[mype]));                     \
    CUDA_CHECK(cudaStreamSynchronize(cstrm[mype]));

#define VALIDATE_PUT(TYPE, nelems, apiname)                                                        \
    for (pe = 0; pe < npes; pe++) {                                                                \
        errs = 0;                                                                                  \
        /*if(pe != mype) {*/                                                                       \
        CUDA_CHECK(cudaMemcpyAsync(hostTestBuffer + pe * nelems * sizeof(TYPE),                    \
                                   buffer + pe * nelems * sizeof(TYPE), nelems * sizeof(TYPE),     \
                                   cudaMemcpyDeviceToHost, cstrm[pe]));                            \
        CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                              \
        for (size_t i = 0; i < nelems; i++) {                                                      \
            DEBUG_PRINT("[%d] %s\n", mype, TOSTRING(*((TYPE *)hostTestBuffer + pe * nelems + i))); \
            /*printf("[%d] sizeof(TYPE) %d %p %lx %lx \n", mype, sizeof(TYPE), (TYPE *)buffer +    \
             * pe*nelems + i, *((TYPE *)hostTestBuffer + pe*nelems + i), (TYPE)pe + (TYPE)npes);*/ \
            if (*((TYPE *)hostTestBuffer + pe * nelems + i) != pe + npes) {                        \
                errs++;                                                                            \
            }                                                                                      \
        }                                                                                          \
        if (errs > 0) {                                                                            \
            ERROR_PRINT("[%d][%d] " #apiname " errors %d nelems %d\n", mype, pe, errs, nelems);    \
            status = -1;                                                                           \
        }                                                                                          \
        /*}*/                                                                                      \
    }

#define TEST_SHMEM_TYPE_P(type, TYPE, niter)                                                 \
    /*init host test buff and use that to init shmem buff to be pushed to*/                  \
    INIT_SHMEM_TEST_BUFF(sizeof(TYPE))                                                       \
    /*init host ref buff and use that to init dev ref buff*/                                 \
    INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, 1)                                                      \
    nvshmem_barrier_all();                                                                   \
    /*issue shmem put's*/                                                                    \
    {                                                                                        \
        TYPE val;                                                                            \
        CUDA_CHECK(cudaMemcpyAsync(&val, devRefBuffer, sizeof(TYPE), cudaMemcpyDeviceToHost, \
                                   cstrm[mype]));                                            \
        for (pe = 0; pe < npes; pe++) {                                                      \
            for (int iter = 0; iter < niter; iter++) {                                       \
                /*if(pe != mype) {*/                                                         \
                nvshmem_##type##_p((TYPE *)buffer + mype, val, pe);                          \
                /*}*/                                                                        \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
    nvshmem_barrier_all();                                                                   \
    /*copy shmem test buff to host test buff and validate*/                                  \
    VALIDATE_PUT(TYPE, 1, nvshmem_##type##_p)

#define TEST_SHMEMX_TYPE_P_ON_STREAM(type, TYPE, niter)                                      \
    /*init host test buff and use that to init shmem buff to be pushed to*/                  \
    INIT_SHMEM_TEST_BUFF(sizeof(TYPE))                                                       \
    /*init host ref buff and use that to init dev ref buff*/                                 \
    INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, 1)                                                      \
    nvshmem_barrier_all();                                                                   \
    /*issue shmem put's*/                                                                    \
    {                                                                                        \
        TYPE val;                                                                            \
        CUDA_CHECK(cudaMemcpyAsync(&val, devRefBuffer, sizeof(TYPE), cudaMemcpyDeviceToHost, \
                                   cstrm[mype]));                                            \
        for (pe = 0; pe < npes; pe++) {                                                      \
            for (int iter = 0; iter < niter; iter++) {                                       \
                /*if(pe != mype) {*/                                                         \
                nvshmemx_##type##_p_on_stream((TYPE *)buffer + mype, val, pe, cstrm[pe]);    \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                \
                /*}*/                                                                        \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
    nvshmem_barrier_all();                                                                   \
    /*copy shmem test buff to host test buff and validate*/                                  \
    VALIDATE_PUT(TYPE, 1, nvshmemx_##type##_p_on_stream)

#define MIN_BYTES_PER_WORD 1
#define MAX_ITER 1
#define MAX_NPES 8
#define MAX_WORDSIZE 16
#define MAX_SHEAP_BYTES (1024 * 1024 * 1024)
#define LIB_SHEAP_FACTOR 2
#define MAX_ELEMS_FULL (MAX_SHEAP_BYTES / (MAX_WORDSIZE * MAX_NPES * 2))
#undef MAX_ELEMS
#define MAX_ELEMS (MAX_ELEMS_FULL - (MAX_ELEMS_FULL / LIB_SHEAP_FACTOR))

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes, pe;
    int niter = MAX_ITER;
    size_t nelemsAny = MAX_ELEMS;  // 8MB generates cuMemcpyHtoDAsync error in
                                   // p2p.cpp:shmemt_p2p_cpu_initiated_write
    size_t nelems = nelemsAny;
    char *buffer = NULL, *devRefBuffer = NULL, *hostTestBuffer = NULL, *hostRefBuffer = NULL;
    int errs = 0;
    cudaStream_t *cstrm;
    read_args(argc, argv);
    if (sizeof(int) > MIN_BYTES_PER_WORD) {
        nelems = (int)std::ceil((float)nelemsAny * (float)MIN_BYTES_PER_WORD / (float)sizeof(int)) *
                 sizeof(int) / MIN_BYTES_PER_WORD;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    cstrm = (cudaStream_t *)malloc(npes * sizeof(cudaStream_t));
    if (!cstrm) {
        ERROR_PRINT("CUDA stream object creation failed \n");
        status = -1;
        goto out;
    }

    for (int i = 0; i < npes; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm[i], cudaStreamNonBlocking));
    }
    if (use_mmap) {
        buffer = (char *)allocate_mmap_buffer(nelems * sizeof(long double) * npes, _mem_handle_type,
                                              use_egm);
    } else {
        buffer = (char *)nvshmem_malloc(nelems * sizeof(long double) * npes);
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
    CUDA_CHECK(cudaMalloc(&devRefBuffer, nelems * sizeof(long double)));
#else
    if (use_mmap) {
        devRefBuffer =
            (char *)allocate_mmap_buffer(nelems * sizeof(long double), _mem_handle_type, use_egm);
    } else {
        devRefBuffer = (char *)nvshmem_malloc(nelems * sizeof(long double));
    }
    if (!devRefBuffer) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }
#endif

    nvshmem_barrier_all();

    TEST_SHMEM_TYPE_P(float, float, niter)
    TEST_SHMEM_TYPE_P(double, double, niter)
    /* char can be signed or unsigned depending on the platform and npes over 63
     * causes us to overflow the signed type. So skip this test if npes is over 64.
     */
    if (npes < 64) {
        TEST_SHMEM_TYPE_P(char, char, niter)
    }
    TEST_SHMEM_TYPE_P(short, short, niter)
    TEST_SHMEM_TYPE_P(int, int, niter)
    TEST_SHMEM_TYPE_P(long, long, niter)
    TEST_SHMEM_TYPE_P(longlong, long long, niter)

    free(hostRefBuffer);
    free(hostTestBuffer);
    for (int i = 0; i < npes; i++) {
        CUDA_CHECK(cudaStreamDestroy(cstrm[i]));
    }

#ifdef REGISTRATION_CACHE_ENABLED
    CUDA_CHECK(cudaFree(devRefBuffer));
#else
    if (use_mmap) {
        free_mmap_buffer(devRefBuffer);
    } else {
        nvshmem_free(devRefBuffer);
    }
#endif
    if (use_mmap) {
        free_mmap_buffer(buffer);
    } else {
        nvshmem_free(buffer);
    }
out:
    finalize_wrapper();
    return status;
}
