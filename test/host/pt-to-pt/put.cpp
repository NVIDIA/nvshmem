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

#define INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(nbytes)                                            \
    for (size_t i = 0; i < nelems * sizeof(long double); i++) {                             \
        *((unsigned char *)hostRefBuffer + i) = 0;                                          \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpyAsync(devRefBuffer, hostRefBuffer, sizeof(long double) * nelems,   \
                               cudaMemcpyHostToDevice, cstrm[mype]));                       \
    CUDA_CHECK(cudaStreamSynchronize(cstrm[mype]));                                         \
    for (size_t i = 0; i < nbytes; i++) {                                                   \
        *((unsigned char *)hostRefBuffer + i) = (unsigned char)mype + (unsigned char)npes;  \
    }                                                                                       \
    CUDA_CHECK(cudaMemcpyAsync(devRefBuffer, hostRefBuffer, nbytes, cudaMemcpyHostToDevice, \
                               cstrm[mype]));                                               \
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
            /*DEBUG_PRINT("[%d] %s\n", mype, TOSTRING(*((TYPE *)hostTestBuffer + pe * nelems +     \
             * i))); */                                                                            \
            /*printf("[%d] sizeof(TYPE) %d %p %lx %lx \n", mype, sizeof(TYPE), (TYPE *)buffer +    \
             * pe*nelems + i, *((TYPE *)hostTestBuffer + pe*nelems + i), (TYPE)pe + (TYPE)npes);*/ \
            if (*((TYPE *)hostTestBuffer + pe * nelems + i) != pe + npes) {                        \
                errs++;                                                                            \
            }                                                                                      \
        }                                                                                          \
        if (errs > 0) {                                                                            \
            ERROR_PRINT("[%d][%d] " #apiname " errors %d nelems %zu\n", mype, pe, errs, nelems);   \
            status = -1;                                                                           \
        }                                                                                          \
        /*}*/                                                                                      \
    }

#define VALIDATE_BYTES_PUT(bytesPerWord, nelems, apiname)                                          \
    for (pe = 0; pe < npes; pe++) {                                                                \
        errs = 0;                                                                                  \
        if (pe != mype) {                                                                          \
            CUDA_CHECK(cudaMemcpyAsync(hostTestBuffer + pe * nelems * bytesPerWord,                \
                                       buffer + pe * nelems * bytesPerWord, nelems * bytesPerWord, \
                                       cudaMemcpyDeviceToHost, cstrm[pe]));                        \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                          \
            for (size_t i = 0; i < nelems; i++) {                                                  \
                for (int j = 0; j < bytesPerWord; j++) {                                           \
                    /*DEBUG_PRINT("[%d] %x\n", mype, *((unsigned char *)hostTestBuffer +           \
                     * pe*nelems*bytesPerWord + i*bytesPerWord + j));*/                            \
                    if (*((unsigned char *)hostTestBuffer + pe * nelems * bytesPerWord +           \
                          i * bytesPerWord + j) != (unsigned char)pe + (unsigned char)npes) {      \
                        errs++;                                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            if (errs > 0) {                                                                        \
                ERROR_PRINT("[%d][%d] " #apiname " errors %d nelems %zu\n", mype, pe, errs,        \
                            nelems);                                                               \
                status = -1;                                                                       \
            }                                                                                      \
        }                                                                                          \
    }

#define TEST_SHMEM_PUTMEM(nbytes, niter)                                                      \
    for (size_t bytes = 1; bytes <= nbytes; bytes *= 2) {                                     \
        /*init host test buff and use that to init shmem buff to be pushed to*/               \
        INIT_SHMEM_TEST_BUFF(bytes)                                                           \
        /*init host ref buff and use that to init dev ref buff*/                              \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(bytes)                                               \
        nvshmem_barrier_all();                                                                \
        /*issue shmem put's*/                                                                 \
        for (pe = 0; pe < npes; pe++) {                                                       \
            for (int iter = 0; iter < niter; iter++) {                                        \
                /*if(pe != mype) {*/                                                          \
                nvshmem_putmem((void *)((char *)buffer + bytes * mype), (void *)devRefBuffer, \
                               bytes, pe);                                                    \
                /*}*/                                                                         \
            }                                                                                 \
        }                                                                                     \
        nvshmem_barrier_all();                                                                \
        /*copy shmem test buff to host test buff and validate*/                               \
        VALIDATE_BYTES_PUT(1, bytes, nvshmem_putmem)                                          \
    }

#define TEST_SHMEMX_PUTMEM_ON_STREAM(nbytes, niter)                                    \
    for (size_t bytes = 1; bytes <= nbytes; bytes *= 2) {                              \
        /*init host test buff and use that to init shmem buff to be pushed to*/        \
        INIT_SHMEM_TEST_BUFF(bytes)                                                    \
        /*init host ref buff and use that to init dev ref buff*/                       \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(bytes)                                        \
        nvshmem_barrier_all();                                                         \
        /*issue shmem put's*/                                                          \
        for (pe = 0; pe < npes; pe++) {                                                \
            for (int iter = 0; iter < niter; iter++) {                                 \
                /*if(pe != mype) {*/                                                   \
                nvshmemx_putmem_on_stream((void *)((char *)buffer + bytes * mype),     \
                                          (void *)devRefBuffer, bytes, pe, cstrm[pe]); \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                          \
                /*}*/                                                                  \
            }                                                                          \
        }                                                                              \
        nvshmem_barrier_all();                                                         \
        /*copy shmem test buff to host test buff and validate*/                        \
        VALIDATE_BYTES_PUT(1, bytes, nvshmemx_putmem_on_stream)                        \
    }

#define TEST_SHMEM_TYPE_PUT(type, TYPE, nelems, niter)                                             \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                          \
        /*init host test buff and use that to init shmem buff to be pushed to*/                    \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                                 \
        /*init host ref buff and use that to init dev ref buff*/                                   \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                                    \
        nvshmem_barrier_all();                                                                     \
        /*issue shmem put's*/                                                                      \
        for (pe = 0; pe < npes; pe++) {                                                            \
            for (int iter = 0; iter < niter; iter++) {                                             \
                /*printf("[%d] peer %d elems %ld iter %d\n", mype, pe, elems, iter);*/             \
                /*if(pe != mype) {*/                                                               \
                nvshmem_##type##_put(((TYPE *)buffer + elems * mype), (TYPE *)devRefBuffer, elems, \
                                     pe);                                                          \
                /*}*/                                                                              \
            }                                                                                      \
        }                                                                                          \
        nvshmem_barrier_all();                                                                     \
        /*copy shmem test buff to host test buff and validate*/                                    \
        VALIDATE_PUT(TYPE, elems, nvshmem_##type##_put)                                            \
    }

#define TEST_SHMEMX_TYPE_PUT_ON_STREAM(type, TYPE, nelems, niter)                            \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                    \
        /*init host test buff and use that to init shmem buff to be pushed to*/              \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                           \
        /*init host ref buff and use that to init dev ref buff*/                             \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                              \
        nvshmem_barrier_all();                                                               \
        /*issue shmem put's*/                                                                \
        for (pe = 0; pe < npes; pe++) {                                                      \
            for (int iter = 0; iter < niter; iter++) {                                       \
                /*if(pe != mype) {*/                                                         \
                nvshmemx_##type##_put_on_stream(((TYPE *)buffer + elems * mype),             \
                                                (TYPE *)devRefBuffer, elems, pe, cstrm[pe]); \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                \
                /*}*/                                                                        \
            }                                                                                \
        }                                                                                    \
        nvshmem_barrier_all();                                                               \
        /*copy shmem test buff to host test buff and validate*/                              \
        VALIDATE_PUT(TYPE, elems, nvshmemx_##type##_put_on_stream)                           \
    }

#define TEST_SHMEM_PUT_WORDSIZE(bitsPerWord, bytesPerWord, nelems, niter)                        \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                        \
        /*init host test buff and use that to init shmem buff to be pushed to*/                  \
        INIT_SHMEM_TEST_BUFF(elems *bytesPerWord)                                                \
        /*init host ref buff and use that to init dev ref buff*/                                 \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(elems *bytesPerWord)                                    \
        nvshmem_barrier_all();                                                                   \
        /*issue shmem put's*/                                                                    \
        for (pe = 0; pe < npes; pe++) {                                                          \
            for (int iter = 0; iter < niter; iter++) {                                           \
                /*if(pe != mype) {*/                                                             \
                nvshmem_put##bitsPerWord((void *)((char *)buffer + elems * bytesPerWord * mype), \
                                         (void *)devRefBuffer, elems, pe);                       \
                /*}*/                                                                            \
            }                                                                                    \
        }                                                                                        \
        nvshmem_barrier_all();                                                                   \
        /*copy shmem test buff to host test buff and validate*/                                  \
        VALIDATE_BYTES_PUT(bytesPerWord, elems, nvshmem_put##bitsPerWord)                        \
    }

#define TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(bitsPerWord, bytesPerWord, nelems, niter)              \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                         \
        /*init host test buff and use that to init shmem buff to be pushed to*/                   \
        INIT_SHMEM_TEST_BUFF(elems *bytesPerWord)                                                 \
        /*init host ref buff and use that to init dev ref buff*/                                  \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(elems *bytesPerWord)                                     \
        nvshmem_barrier_all();                                                                    \
        /*issue shmem put's*/                                                                     \
        for (pe = 0; pe < npes; pe++) {                                                           \
            for (int iter = 0; iter < niter; iter++) {                                            \
                /*if(pe != mype) {*/                                                              \
                nvshmemx_put##bitsPerWord##_on_stream(                                            \
                    (void *)((char *)buffer + elems * bytesPerWord * mype), (void *)devRefBuffer, \
                    elems, pe, cstrm[pe]);                                                        \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                     \
                /*}*/                                                                             \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        /*copy shmem test buff to host test buff and validate*/                                   \
        VALIDATE_BYTES_PUT(bytesPerWord, elems, nvshmemx_put##bitsPerWord##_on_stream)            \
    }

#define TEST_SHMEM_TYPE_PUT_NBI(type, TYPE, nelems, niter)                                      \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                       \
        /*init host test buff and use that to init shmem buff to be pushed to*/                 \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                              \
        /*init host ref buff and use that to init dev ref buff*/                                \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                                 \
        nvshmem_barrier_all();                                                                  \
        /*issue shmem put's*/                                                                   \
        for (pe = 0; pe < npes; pe++) {                                                         \
            for (int iter = 0; iter < niter; iter++) {                                          \
                /*if(pe != mype) {*/                                                            \
                nvshmem_##type##_put_nbi(((TYPE *)buffer + elems * mype), (TYPE *)devRefBuffer, \
                                         elems, pe);                                            \
                nvshmem_quiet();                                                                \
                /*}*/                                                                           \
            }                                                                                   \
        }                                                                                       \
        nvshmem_barrier_all();                                                                  \
        /*copy shmem test buff to host test buff and validate*/                                 \
        VALIDATE_PUT(TYPE, elems, nvshmem_##type##_put_nbi)                                     \
    }

#define TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(type, TYPE, nelems, niter)                            \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                        \
        /*init host test buff and use that to init shmem buff to be pushed to*/                  \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                               \
        /*init host ref buff and use that to init dev ref buff*/                                 \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                                  \
        nvshmem_barrier_all();                                                                   \
        /*issue shmem put's*/                                                                    \
        for (pe = 0; pe < npes; pe++) {                                                          \
            for (int iter = 0; iter < niter; iter++) {                                           \
                /*if(pe != mype) {*/                                                             \
                nvshmemx_##type##_put_nbi_on_stream(((TYPE *)buffer + elems * mype),             \
                                                    (TYPE *)devRefBuffer, elems, pe, cstrm[pe]); \
                nvshmemx_quiet_on_stream(cstrm[pe]);                                             \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                    \
                /*}*/                                                                            \
            }                                                                                    \
        }                                                                                        \
        nvshmem_barrier_all();                                                                   \
        /*copy shmem test buff to host test buff and validate*/                                  \
        VALIDATE_PUT(TYPE, elems, nvshmemx_##type##_put_nbi_on_stream)                           \
    }

#define TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(bitsPerWord, bytesPerWord, nelems, niter)                \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                         \
        /*init host test buff and use that to init shmem buff to be pushed to*/                   \
        INIT_SHMEM_TEST_BUFF(elems *bytesPerWord)                                                 \
        /*init host ref buff and use that to init dev ref buff*/                                  \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(elems *bytesPerWord)                                     \
        nvshmem_barrier_all();                                                                    \
        /*issue shmem put's*/                                                                     \
        for (pe = 0; pe < npes; pe++) {                                                           \
            for (int iter = 0; iter < niter; iter++) {                                            \
                /*if(pe != mype) {*/                                                              \
                nvshmem_put##bitsPerWord##_nbi(                                                   \
                    (void *)((char *)buffer + elems * bytesPerWord * mype), (void *)devRefBuffer, \
                    elems, pe);                                                                   \
                nvshmem_quiet();                                                                  \
                /*}*/                                                                             \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        /*copy shmem test buff to host test buff and validate*/                                   \
        VALIDATE_BYTES_PUT(bytesPerWord, elems, nvshmem_put##bitsPerWord##_nbi)                   \
    }

#define TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(bitsPerWord, bytesPerWord, nelems, niter)     \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                         \
        /*init host test buff and use that to init shmem buff to be pushed to*/                   \
        INIT_SHMEM_TEST_BUFF(elems *bytesPerWord)                                                 \
        /*init host ref buff and use that to init dev ref buff*/                                  \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(elems *bytesPerWord)                                     \
        nvshmem_barrier_all();                                                                    \
        /*issue shmem put's*/                                                                     \
        for (pe = 0; pe < npes; pe++) {                                                           \
            for (int iter = 0; iter < niter; iter++) {                                            \
                /*if(pe != mype) {*/                                                              \
                nvshmemx_put##bitsPerWord##_nbi_on_stream(                                        \
                    (void *)((char *)buffer + elems * bytesPerWord * mype), (void *)devRefBuffer, \
                    elems, pe, cstrm[pe]);                                                        \
                nvshmemx_quiet_on_stream(cstrm[pe]);                                              \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                     \
                /*}*/                                                                             \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        /*copy shmem test buff to host test buff and validate*/                                   \
        VALIDATE_BYTES_PUT(bytesPerWord, elems, nvshmemx_put##bitsPerWord##_nbi_on_stream)        \
    }

#define TEST_SHMEM_PUTMEM_NBI(nbytes, niter)                                                      \
    for (size_t bytes = 1; bytes <= nbytes; bytes *= 2) {                                         \
        /*init host test buff and use that to init shmem buff to be pushed to*/                   \
        INIT_SHMEM_TEST_BUFF(bytes)                                                               \
        /*init host ref buff and use that to init dev ref buff*/                                  \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(bytes)                                                   \
        nvshmem_barrier_all();                                                                    \
        /*issue shmem put's*/                                                                     \
        for (pe = 0; pe < npes; pe++) {                                                           \
            for (int iter = 0; iter < niter; iter++) {                                            \
                /*if(pe != mype) {*/                                                              \
                nvshmem_putmem_nbi((void *)((char *)buffer + bytes * mype), (void *)devRefBuffer, \
                                   bytes, pe);                                                    \
                nvshmem_quiet();                                                                  \
                /*}*/                                                                             \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier_all();                                                                    \
        /*copy shmem test buff to host test buff and validate*/                                   \
        VALIDATE_BYTES_PUT(1, bytes, nvshmem_putmem_nbi)                                          \
    }

#define TEST_SHMEMX_PUTMEM_NBI_ON_STREAM(nbytes, niter)                                    \
    for (size_t bytes = 1; bytes <= nbytes; bytes *= 2) {                                  \
        /*init host test buff and use that to init shmem buff to be pushed to*/            \
        INIT_SHMEM_TEST_BUFF(bytes)                                                        \
        /*init host ref buff and use that to init dev ref buff*/                           \
        INIT_BYTES_DEV_REF_BUFF_FOR_PUSH(bytes)                                            \
        nvshmem_barrier_all();                                                             \
        /*issue shmem put's*/                                                              \
        for (pe = 0; pe < npes; pe++) {                                                    \
            for (int iter = 0; iter < niter; iter++) {                                     \
                /*if(pe != mype) {*/                                                       \
                nvshmemx_putmem_nbi_on_stream((void *)((char *)buffer + bytes * mype),     \
                                              (void *)devRefBuffer, bytes, pe, cstrm[pe]); \
                nvshmemx_quiet_on_stream(cstrm[pe]);                                       \
                CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                              \
                /*}*/                                                                      \
            }                                                                              \
        }                                                                                  \
        nvshmem_barrier_all();                                                             \
        /*copy shmem test buff to host test buff and validate*/                            \
        VALIDATE_BYTES_PUT(1, bytes, nvshmem_putmem_nbi_on_stream)                         \
    }

#define TEST_SHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(type, TYPE, nelems, niter)                         \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                        \
        /*init host test buff and use that to init shmem buff to be pushed to*/                  \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                               \
        /*init host ref buff and use that to init dev ref buff*/                                 \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                                  \
        nvshmem_barrier_all();                                                                   \
        /*issue shmem put's*/                                                                    \
        for (pe = 0; pe < npes; pe++) {                                                          \
            nvshmemx_##type##_put_signal_on_stream(((TYPE *)buffer + elems * mype),              \
                                                   (TYPE *)devRefBuffer, elems, &sig_addr[mype], \
                                                   1, NVSHMEM_SIGNAL_SET, pe, cstrm[pe]);        \
        }                                                                                        \
        for (pe = 0; pe < npes; pe++) {                                                          \
            nvshmemx_signal_wait_until_on_stream(&sig_addr[pe], NVSHMEM_CMP_EQ, 1, cstrm[pe]);   \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                        \
        }                                                                                        \
        /*copy shmem test buff to host test buff and validate*/                                  \
        VALIDATE_PUT(TYPE, elems, nvshmemx_##type##_put_signal_on_stream)                        \
        if (use_egm) {                                                                           \
            memset(sig_addr, 0, sizeof(uint64_t) * npes);                                        \
        } else {                                                                                 \
            CUDA_CHECK(cudaMemset(sig_addr, 0, sizeof(uint64_t) * npes));                        \
        }                                                                                        \
        nvshmem_barrier_all();                                                                   \
    }                                                                                            \
    /* Now test with NVSHMEM_SIGNAL_ADD */                                                       \
    for (size_t elems = 1; elems <= nelems; elems *= 2) {                                        \
        /*init host test buff and use that to init shmem buff to be pushed to*/                  \
        INIT_SHMEM_TEST_BUFF(elems * sizeof(TYPE))                                               \
        /*init host ref buff and use that to init dev ref buff*/                                 \
        INIT_DEV_REF_BUFF_FOR_PUSH(TYPE, elems)                                                  \
        nvshmem_barrier_all();                                                                   \
        /*issue shmem put's*/                                                                    \
        for (pe = 0; pe < npes; pe++) {                                                          \
            nvshmemx_##type##_put_signal_on_stream(((TYPE *)buffer + elems * mype),              \
                                                   (TYPE *)devRefBuffer, elems, &sig_addr[mype], \
                                                   1, NVSHMEM_SIGNAL_ADD, pe, cstrm[pe]);        \
        }                                                                                        \
        for (pe = 0; pe < npes; pe++) {                                                          \
            nvshmemx_signal_wait_until_on_stream(&sig_addr[pe], NVSHMEM_CMP_EQ, 1, cstrm[pe]);   \
            CUDA_CHECK(cudaStreamSynchronize(cstrm[pe]));                                        \
        }                                                                                        \
        /*copy shmem test buff to host test buff and validate*/                                  \
        VALIDATE_PUT(TYPE, elems, nvshmemx_##type##_put_signal_on_stream)                        \
        CUDA_CHECK(cudaMemset(sig_addr, 0, sizeof(uint64_t) * npes));                            \
        nvshmem_barrier_all();                                                                   \
    }

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
    int mype, npes, pe;
    int niter = MAX_ITER;
    size_t nelemsAny = MAX_ELEMS;  // 8MB generates cuMemcpyHtoDAsync error in
                                   // p2p.cpp:shmemt_p2p_cpu_initiated_write
    size_t nelems = nelemsAny;
    char *buffer = NULL, *devRefBuffer = NULL, *hostTestBuffer = NULL, *hostRefBuffer = NULL;
    int errs = 0;
    cudaStream_t *cstrm;
    uint64_t *sig_addr;
    bool asymmetric_local_buffer = false;
    bool register_local_buffer = false;
    if (sizeof(int) > MIN_BYTES_PER_WORD) {
        nelems = (int)std::ceil((float)nelemsAny * (float)MIN_BYTES_PER_WORD / (float)sizeof(int)) *
                 sizeof(int) / MIN_BYTES_PER_WORD;
    }

    nelems /= 2; /* To control the time that this test takes */

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

    if (asymmetric_local_buffer) {
        CUDA_CHECK(cudaMalloc(&devRefBuffer, nelems * sizeof(long double) * npes));
        if (register_local_buffer) {
            status = nvshmemx_buffer_register(devRefBuffer, nelems * sizeof(long double) * npes);
            if (status) {
                ERROR_PRINT("nvshmemx_buffer_register failed.\n");
                goto out;
            }
        }
    } else {
        if (use_mmap) {
            devRefBuffer = (char *)allocate_mmap_buffer(nelems * sizeof(long double) * npes,
                                                        _mem_handle_type, use_egm);
        } else {
            devRefBuffer = (char *)nvshmem_malloc(nelems * sizeof(long double) * npes);
        }
        if (!devRefBuffer) {
            ERROR_PRINT("shmem_malloc failed \n");
            status = -1;
            goto out;
        }
    }
    if (use_mmap) {
        sig_addr = (uint64_t *)allocate_mmap_buffer(npes * sizeof(uint64_t), _mem_handle_type,
                                                    use_egm, true);
    } else {
        sig_addr = (uint64_t *)nvshmem_calloc(npes, sizeof(uint64_t));
    }
    nvshmem_barrier_all();

    TEST_SHMEM_TYPE_PUT(float, float, nelems, niter)
    TEST_SHMEM_TYPE_PUT(double, double, nelems, niter)
    // TEST_SHMEM_TYPE_PUT(char, char, nelems, niter)
    // TEST_SHMEM_TYPE_PUT(short, short, nelems, niter)
    TEST_SHMEM_TYPE_PUT(int, int, nelems, niter)
    TEST_SHMEM_TYPE_PUT(long, long, nelems, niter)
    TEST_SHMEM_TYPE_PUT(longlong, long long, nelems, niter)

    // TEST_SHMEM_PUT_WORDSIZE(8, 1, nelems, niter)
    // TEST_SHMEM_PUT_WORDSIZE(16, 2, nelems, niter)
    TEST_SHMEM_PUT_WORDSIZE(32, 4, nelems, niter)
    TEST_SHMEM_PUT_WORDSIZE(64, 8, nelems, niter)
    TEST_SHMEM_PUT_WORDSIZE(128, 16, nelems, niter)

    TEST_SHMEM_PUTMEM(sizeof(long double) * nelems, niter)

    TEST_SHMEM_TYPE_PUT_NBI(float, float, nelems, niter)
    TEST_SHMEM_TYPE_PUT_NBI(double, double, nelems, niter)
    /* char can be signed or unsigned depending on the platform and npes over 63
     * causes us to overflow the signed type. So skip this test if npes is over 64.
     */
    if (npes < 64) {
        TEST_SHMEM_TYPE_PUT_NBI(char, char, nelems, niter)
    }
    TEST_SHMEM_TYPE_PUT_NBI(short, short, nelems, niter)
    TEST_SHMEM_TYPE_PUT_NBI(int, int, nelems, niter)
    TEST_SHMEM_TYPE_PUT_NBI(long, long, nelems, niter)
    TEST_SHMEM_TYPE_PUT_NBI(longlong, long long, nelems, niter)

    TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(8, 1, nelems, niter)
    TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(16, 2, nelems, niter)
    TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(32, 4, nelems, niter)
    TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(64, 8, nelems, niter)
    TEST_SHMEM_TYPE_PUT_WORDSIZE_NBI(128, 16, nelems, niter)

    TEST_SHMEM_PUTMEM_NBI(sizeof(long double) * nelems, niter)

    TEST_SHMEMX_TYPE_PUT_ON_STREAM(float, float, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_ON_STREAM(double, double, nelems, niter)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_PUT_ON_STREAM(char, char, nelems, niter)
    }
    TEST_SHMEMX_TYPE_PUT_ON_STREAM(short, short, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_ON_STREAM(int, int, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_ON_STREAM(long, long, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_ON_STREAM(longlong, long long, nelems, niter)

    TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(8, 1, nelems, niter)
    TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(16, 2, nelems, niter)
    TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(32, 4, nelems, niter)
    TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(64, 8, nelems, niter)
    TEST_SHMEMX_PUT_WORDSIZE_ON_STREAM(128, 16, nelems, niter)

    TEST_SHMEMX_PUTMEM_ON_STREAM(sizeof(long double) * nelems, niter)

    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(float, float, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(double, double, nelems, niter)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(char, char, nelems, niter)
    }
    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(short, short, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(int, int, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(long, long, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_NBI_ON_STREAM(longlong, long long, nelems, niter)

    TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(8, 1, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(16, 2, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(32, 4, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(64, 8, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_WORDSIZE_NBI_ON_STREAM(128, 16, nelems, niter)

    TEST_SHMEMX_PUTMEM_NBI_ON_STREAM(sizeof(long double) * nelems, niter)

    TEST_SHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(float, float, nelems, niter)
    TEST_SHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(longlong, long long, nelems, niter)
    if (npes < 64) {
        TEST_SHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(char, char, nelems, niter)
    }

    free(hostRefBuffer);
    free(hostTestBuffer);
    for (int i = 0; i < npes; i++) {
        CUDA_CHECK(cudaStreamDestroy(cstrm[i]));
    }

    if (asymmetric_local_buffer) {
        nvshmemx_buffer_unregister(devRefBuffer);
        CUDA_CHECK(cudaFree(devRefBuffer));
    } else {
        if (use_mmap) {
            free_mmap_buffer(devRefBuffer);
        } else {
            nvshmem_free(devRefBuffer);
        }
    }

    if (use_mmap) {
        free_mmap_buffer(buffer);
    } else {
        nvshmem_free(buffer);
    }

    if (use_mmap) {
        free_mmap_buffer(sig_addr);
    } else {
        nvshmem_free(sig_addr);
    }
out:
    finalize_wrapper();
    return status;
}
