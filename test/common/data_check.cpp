/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <string.h>
#include "cuda.h"
#include "data_check.h"
#include "utils.h"

static int mype_h;
static int npes_h;
static size_t size_h;
static int disp_h;
static int seed_h;
static int iters_h;

#define FUNC(off, id, peer, seed) (off + (id + 1) * (peer + 1) * (seed + 1))

template <typename T>
int init_data_ring(T *buf, size_t size, int disp, int iters, int mype, int npes, int *nextpe,
                   int *prevpe, int seed, cudaStream_t cstrm) {
    int status = 0;
    T *buf_h = NULL;
    size_t i;
    size_t elems = (size * (size_t)disp) / sizeof(T);

    mype_h = mype;
    npes_h = npes;
    size_h = size;
    disp_h = disp;
    seed_h = seed;
    iters_h = iters;

    if (size % sizeof(T) != 0) {
        ERROR_PRINT("size of buffer has to be a multiple of %lu bytes \n", sizeof(T));
        status = -1;
        goto out;
    }

    CUDA_CHECK(cudaMallocHost((void **)&buf_h, size * disp * iters));
    memset(buf_h, 0, size * disp * iters);

    for (int n = 0; n < iters; n++) {
        for (i = 0; i < elems; i++) {
            *(buf_h + (size_t)n * elems + i) = 1 * FUNC(i, mype_h, 0, seed_h + n);
            // max_uint16_t 65535 max_int16_t 32767 max_uint32_t 4294967295 max_int32_t 2147483647
            // max_uint64_t 9223372036854775807 max_int64_t 4611686018427387903
        }
    }

    status = cudaMemcpyAsync(buf, buf_h, size * (size_t)disp * (size_t)iters,
                             cudaMemcpyHostToDevice, cstrm);
    if (status != CUDA_SUCCESS) {
        ERROR_PRINT("cudaMemcpyAsync failed with error: %d \n", status);
        goto out;
    }
    CUDA_CHECK(cudaStreamSynchronize(cstrm));

    *nextpe = (mype_h + 1) % npes_h;
    *prevpe = (mype_h - 1 + npes_h) % npes_h;

out:
    CUDA_CHECK(cudaFreeHost(buf_h));

    return status;
}
template int init_data_ring<char>(char *buf, size_t size, int disp, int iters, int mype, int npes,
                                  int *nextpe, int *prevpe, int seed, cudaStream_t cstrm);
template int init_data_ring<unsigned char>(unsigned char *buf, size_t size, int disp, int iters,
                                           int mype, int npes, int *nextpe, int *prevpe, int seed,
                                           cudaStream_t cstrm);
template int init_data_ring<short>(short *buf, size_t size, int disp, int iters, int mype, int npes,
                                   int *nextpe, int *prevpe, int seed, cudaStream_t cstrm);
template int init_data_ring<unsigned short>(unsigned short *buf, size_t size, int disp, int iters,
                                            int mype, int npes, int *nextpe, int *prevpe, int seed,
                                            cudaStream_t cstrm);
template int init_data_ring<int>(int *buf, size_t size, int disp, int iters, int mype, int npes,
                                 int *nextpe, int *prevpe, int seed, cudaStream_t cstrm);
template int init_data_ring<unsigned int>(unsigned int *buf, size_t size, int disp, int iters,
                                          int mype, int npes, int *nextpe, int *prevpe, int seed,
                                          cudaStream_t cstrm);
template int init_data_ring<long long int>(long long int *buf, size_t size, int disp, int iters,
                                           int mype, int npes, int *nextpe, int *prevpe, int seed,
                                           cudaStream_t cstrm);
template int init_data_ring<unsigned long long int>(unsigned long long int *buf, size_t size,
                                                    int disp, int iters, int mype, int npes,
                                                    int *nextpe, int *prevpe, int seed,
                                                    cudaStream_t cstrm);
template int init_data_ring<float>(float *buf, size_t size, int disp, int iters, int mype, int npes,
                                   int *nextpe, int *prevpe, int seed, cudaStream_t cstrm);
template int init_data_ring<double>(double *buf, size_t size, int disp, int iters, int mype,
                                    int npes, int *nextpe, int *prevpe, int seed,
                                    cudaStream_t cstrm);
template int init_data_ring<uint64_t>(uint64_t *buf, size_t size, int disp, int iters, int mype,
                                      int npes, int *nextpe, int *prevpe, int seed,
                                      cudaStream_t cstrm);

template <typename T>
int init_data_alltoall(T *buf, size_t size, int disp, int iters, int mype, int npes, int seed,
                       cudaStream_t cstrm) {
    int status = 0;
    size_t i;
    T *buf_h = NULL;
    size_t elems = (size * (size_t)disp) / sizeof(T);

    mype_h = mype;
    npes_h = npes;
    size_h = size;
    disp_h = disp;
    seed_h = seed;
    iters_h = iters;

    if (size % sizeof(T) != 0) {
        ERROR_PRINT("size of buffer has to be a multiple of %lu bytes \n", sizeof(T));
        status = -1;
        goto out;
    }

    CUDA_CHECK(cudaMallocHost((void **)&buf_h, size * disp * npes * iters));
    memset(buf_h, 0, size * disp * npes * iters);

    for (int n = 0; n < iters; n++) {
        for (int p = 0; p < npes; p++) {
            for (i = 0; i < elems; i++) {
                *(buf_h + (size_t)n * elems * (size_t)npes + (size_t)p * elems + i) =
                    -1 * FUNC(i, mype_h, p, seed_h + n);
            }
        }
    }

    status = cudaMemcpyAsync(buf, buf_h, size * disp * npes * iters, cudaMemcpyHostToDevice, cstrm);
    if (status != CUDA_SUCCESS) {
        ERROR_PRINT("cudaMemcpyAsync failed with error: %d \n", status);
        goto out;
    }
    CUDA_CHECK(cudaStreamSynchronize(cstrm));

out:
    CUDA_CHECK(cudaFreeHost(buf_h));

    return status;
}
template int init_data_alltoall<char>(char *buf, size_t size, int disp, int iters, int mype,
                                      int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<unsigned char>(unsigned char *buf, size_t size, int disp, int iters,
                                               int mype, int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<short>(short *buf, size_t size, int disp, int iters, int mype,
                                       int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<unsigned short>(unsigned short *buf, size_t size, int disp,
                                                int iters, int mype, int npes, int seed,
                                                cudaStream_t cstrm);
template int init_data_alltoall<int>(int *buf, size_t size, int disp, int iters, int mype, int npes,
                                     int seed, cudaStream_t cstrm);
template int init_data_alltoall<unsigned int>(unsigned int *buf, size_t size, int disp, int iters,
                                              int mype, int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<long long int>(long long int *buf, size_t size, int disp, int iters,
                                               int mype, int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<unsigned long long int>(unsigned long long int *buf, size_t size,
                                                        int disp, int iters, int mype, int npes,
                                                        int seed, cudaStream_t cstrm);
template int init_data_alltoall<float>(float *buf, size_t size, int disp, int iters, int mype,
                                       int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<double>(double *buf, size_t size, int disp, int iters, int mype,
                                        int npes, int seed, cudaStream_t cstrm);
template int init_data_alltoall<uint64_t>(uint64_t *buf, size_t size, int disp, int iters, int mype,
                                          int npes, int seed, cudaStream_t cstrm);

template <typename T>
int check_data_ring(T *buf, cudaStream_t cstrm) {
    int i, status = 0;
    T *buf_h = NULL;
    int prevpe = (mype_h - 1 + npes_h) % npes_h;
    int elems = (size_h * disp_h) / sizeof(T);

    CUDA_CHECK(cudaMallocHost((void **)&buf_h, size_h * disp_h * iters_h));
    memset(buf_h, 0, size_h * disp_h * iters_h);

    status = cudaMemcpyAsync(buf_h, buf, size_h * disp_h * iters_h, cudaMemcpyDeviceToHost, cstrm);
    if (status != CUDA_SUCCESS) {
        ERROR_PRINT("cudaMemcpyAsync failed with error: %d \n", status);
        goto out;
    }
    CUDA_CHECK(cudaStreamSynchronize(cstrm));

    for (int n = 0; n < iters_h; n++) {
        for (i = 0; i < elems; i++) {
            T expected = (i % disp_h != 0) ? 0 : (1 * FUNC(i, prevpe, 0, seed_h + n));
            // max_uint16_t 65535 max_int16_t 32767 max_uint32_t 4294967295 max_int32_t 2147483647
            // max_uint64_t 9223372036854775807 max_int64_t 4611686018427387903
            if (*(buf_h + n * elems + i) != expected) {
                ERROR_PRINT(
                    "[%d] ring data validation failed at iter %d offset %d addr: %p expected: %s "
                    "actual: %s \n",
                    mype_h, n, i, (buf + n * elems + i), TOSTRING(expected),
                    TOSTRING(*(buf_h + n * elems + i)));
                status = -1;
                goto out;
            }
        }
    }

out:
    CUDA_CHECK(cudaFreeHost(buf_h));
    return status;
}
template int check_data_ring<char>(char *buf, cudaStream_t cstrm);
template int check_data_ring<unsigned char>(unsigned char *buf, cudaStream_t cstrm);
template int check_data_ring<short>(short *buf, cudaStream_t cstrm);
template int check_data_ring<unsigned short>(unsigned short *buf, cudaStream_t cstrm);
template int check_data_ring<int>(int *buf, cudaStream_t cstrm);
template int check_data_ring<unsigned int>(unsigned int *buf, cudaStream_t cstrm);
template int check_data_ring<long long int>(long long int *buf, cudaStream_t cstrm);
template int check_data_ring<unsigned long long int>(unsigned long long int *buf,
                                                     cudaStream_t cstrm);
template int check_data_ring<float>(float *buf, cudaStream_t cstrm);
template int check_data_ring<double>(double *buf, cudaStream_t cstrm);
template int check_data_ring<uint64_t>(uint64_t *buf, cudaStream_t cstrm);

template <typename T>
int check_data_alltoall(T *buf, cudaStream_t cstrm) {
    int j, status = 0;
    size_t i;
    T *buf_h = NULL;
    T *buf_to_check = NULL;
    size_t elems = (size_h * (size_t)disp_h) / sizeof(T);

    CUDA_CHECK(cudaMallocHost((void **)&buf_h, size_h * disp_h * npes_h * iters_h));
    memset(buf_h, 0, size_h * disp_h * npes_h * iters_h);

    status = cudaMemcpyAsync(buf_h, buf, size_h * disp_h * npes_h * iters_h, cudaMemcpyDeviceToHost,
                             cstrm);
    if (status != CUDA_SUCCESS) {
        ERROR_PRINT("cudaMemcpyAsync failed with error: %d \n", status);
        goto out;
    }
    CUDA_CHECK(cudaStreamSynchronize(cstrm));

    for (int n = 0; n < iters_h; n++) {
        for (j = 0; j < npes_h; j++) {
            for (i = 0; i < elems; i++) {
                T expected = (i % disp_h != 0) ? 0 : (-1 * FUNC(i, j, mype_h, seed_h + n));
                buf_to_check = buf_h + (size_t)n * (size_t)npes_h * elems + (size_t)j * elems + i;
                if (*(buf_to_check) != expected) {
                    ERROR_PRINT(
                        "[%d] alltoall data validation failed at offset %lu iteration %d peer %d "
                        "expected: %s actual: %s \n",
                        mype_h, i, n, j, TOSTRING(expected), TOSTRING(*(buf_to_check)));
                    status = -1;
                    goto out;
                }
            }
        }
    }

out:
    CUDA_CHECK(cudaFreeHost(buf_h));
    return status;
}
template int check_data_alltoall<char>(char *buf, cudaStream_t cstrm);
template int check_data_alltoall<unsigned char>(unsigned char *buf, cudaStream_t cstrm);
template int check_data_alltoall<short>(short *buf, cudaStream_t cstrm);
template int check_data_alltoall<unsigned short>(unsigned short *buf, cudaStream_t cstrm);
template int check_data_alltoall<int>(int *buf, cudaStream_t cstrm);
template int check_data_alltoall<unsigned int>(unsigned int *buf, cudaStream_t cstrm);
template int check_data_alltoall<long long int>(long long int *buf, cudaStream_t cstrm);
template int check_data_alltoall<unsigned long long int>(unsigned long long int *buf,
                                                         cudaStream_t cstrm);
template int check_data_alltoall<float>(float *buf, cudaStream_t cstrm);
template int check_data_alltoall<double>(double *buf, cudaStream_t cstrm);
template int check_data_alltoall<uint64_t>(uint64_t *buf, cudaStream_t cstrm);
