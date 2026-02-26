/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cstdio>
#include <cstdlib>
#include "nvshmem.h"
#include "nvshmemx.h"

#include "utils.h"

#define TEST_SHMEM_TYPE_WAIT_COMMON(type, TYPE, wait_end_val)                                      \
    /*init shmem buff val to begin wait, init valid buff to 0*/                                    \
    for (size_t offset = 0; offset < npes * sizeof(long long); offset++) {                         \
        *((char *)waitBeginValOnHost + offset) = 0;                                                \
    }                                                                                              \
    CUDA_CHECK(                                                                                    \
        cudaMemcpy(valid, waitBeginValOnHost, sizeof(TYPE) * npes, cudaMemcpyHostToDevice));       \
    for (np = 0; np < npes; np++) {                                                                \
        *((TYPE *)waitBeginValOnHost + np) = (TYPE)np + (TYPE)npes;                                \
    }                                                                                              \
    CUDA_CHECK(cudaMemcpy(ivar, waitBeginValOnHost, sizeof(TYPE) * npes, cudaMemcpyHostToDevice)); \
    /*Push the validation value*/                                                                  \
    /*for (int np = 0; np < npes; np++) {*/                                                        \
    /*if(np != mype) {*/                                                                           \
    /*nvshmem_longlong_p((long long *)valid + mype, (long long)mype, np);*/                        \
    /*}*/                                                                                          \
    /*}*/                                                                                          \
    /*XXX: Bypass shmem_p, force shmem buff val to end wait*/                                      \
    /*for(int np=0; np<npes; np++){*/                                                              \
    /**((TYPE *)waitEndValOnHost + np) = (TYPE)np + (TYPE)npes + (TYPE)wait_end_val;*/             \
    /*}*/                                                                                          \
    /*CUDA_CHECK(cudaMemcpy(ivar, waitEndValOnHost, sizeof(TYPE)*npes, cudaMemcpyHostToDevice));*/ \
    /*integrated test, issue shmem_p to end wait*/                                                 \
    nvshmem_barrier_all();                                                                         \
    for (np = 0; np < npes; np++) {                                                                \
        if (np != mype) {                                                                          \
            nvshmem_##type##_p((TYPE *)ivar + mype, (TYPE)mype + (TYPE)npes + (TYPE)wait_end_val,  \
                               np);                                                                \
        }                                                                                          \
    }                                                                                              \
    /*nvshmem_quiet();*/                                                                           \
    nvshmem_barrier_all();

#define TEST_SHMEM_SYNC_VALIDATE                                        \
    errs = 0;                                                           \
    /*CUDA_CHECK(cudaMemcpy(validOnHost, valid, sizeof(long long)*npes, \
     * cudaMemcpyDeviceToHost));*/                                      \
    /*for (int np = 0; np < npes; np++) {*/                             \
    /*if(np != mype) {*/                                                \
    /*if(*((long long *)validOnHost + np) != (long long)np) {*/         \
    /*errs++;*/                                                         \
    /*}*/                                                               \
    /*}*/                                                               \
    /*}*/                                                               \
    if (errs > 0) {                                                     \
        ERROR_PRINT("[%d]Found %d errors\n", mype, errs);               \
    }

#define TEST_SHMEM_TYPE_WAIT_UNTIL_COMMON(type, TYPE, wait_end_val_w)                              \
    /*init shmem buff val to begin wait*/                                                          \
    for (size_t offset = 0; offset < npes * sizeof(long long); offset++) {                         \
        *((char *)waitBeginValOnHost + offset) = 0;                                                \
    }                                                                                              \
    CUDA_CHECK(                                                                                    \
        cudaMemcpy(valid, waitBeginValOnHost, sizeof(TYPE) * npes, cudaMemcpyHostToDevice));       \
    for (np = 0; np < npes; np++) {                                                                \
        *((TYPE *)waitBeginValOnHost + np) = (TYPE)np + (TYPE)npes;                                \
    }                                                                                              \
    CUDA_CHECK(cudaMemcpy(ivar, waitBeginValOnHost, sizeof(TYPE) * npes, cudaMemcpyHostToDevice)); \
    /*Push the validation value*/                                                                  \
    /*for (int np = 0; np < npes; np++) {*/                                                        \
    /*if(np != mype) {*/                                                                           \
    /*nvshmem_longlong_p((long long *)valid + mype, (long long)mype, np);*/                        \
    /*}*/                                                                                          \
    /*}*/                                                                                          \
    /*XXX: Bypass shmem_p, force shmem buff val to end wait*/                                      \
    for (np = 0; np < npes; np++) {                                                                \
        *((TYPE *)waitEndValOnHost + np) = (TYPE)np + (TYPE)npes + (TYPE)wait_end_val_w;           \
    }                                                                                              \
    CUDA_CHECK(cudaMemcpy(ivar, waitEndValOnHost, sizeof(TYPE) * npes, cudaMemcpyHostToDevice));   \
    /*integrated test, issue shmem_p to end wait*/                                                 \
    /*for (int np = 0; np < npes; np++) {*/                                                        \
    /*if(np != mype) {*/                                                                           \
    /*nvshmem_##type##_p((TYPE *)ivar + mype, (TYPE)mype + (TYPE)npes + (TYPE)wait_end_val_w,      \
     * np);*/                                                                                      \
    /*}*/                                                                                          \
    /*}*/                                                                                          \
    /*nvshmem_quiet(); */

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, wait_end_val_r, wait_end_val_w)  \
    TEST_SHMEM_TYPE_WAIT_UNTIL_COMMON(type, TYPE, wait_end_val_w)                               \
    /*wait on every other peer*/                                                                \
    for (np = 0; np < npes; np++) {                                                             \
        if (np != mype) {                                                                       \
            nvshmemx_##type##_wait_until_on_stream(                                             \
                (TYPE *)ivar + np, cmp, (TYPE)np + (TYPE)npes + (TYPE)wait_end_val_r, cstream); \
            CUDA_CHECK(cudaStreamSynchronize(cstream));                                         \
        }                                                                                       \
    }                                                                                           \
    TEST_SHMEM_SYNC_VALIDATE

/*With CUDA stream, typed*/
#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_EQ(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 0, 0)  \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 1, 1)  \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, -1, -1)

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LT(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 0, -1) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 1, 0)  \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, -1, -2)

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LE(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_EQ(type, TYPE, cmp)     \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LT(type, TYPE, cmp)

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GT(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 0, 1)  \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, 1, 2)  \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE, cmp, -1, 0)

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GE(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_EQ(type, TYPE, cmp)     \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GT(type, TYPE, cmp)

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_NE(type, TYPE, cmp) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LT(type, TYPE, cmp)     \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GT(type, TYPE, cmp)

int main(int argc, char **argv) {
    int status = 0;
    char *ivar = NULL, *valid = NULL;
    char *validOnHost = NULL, *waitBeginValOnHost = NULL, *waitEndValOnHost = NULL;
    int mype, npes;
    int np;
    int errs;
    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (use_mmap) {
        ivar = (char *)allocate_mmap_buffer(sizeof(long long) * npes, _mem_handle_type, use_egm);
        valid = (char *)allocate_mmap_buffer(sizeof(long long) * npes, _mem_handle_type, use_egm);
    } else {
        ivar = (char *)nvshmem_malloc(sizeof(long long) * npes);
        valid = (char *)nvshmem_malloc(sizeof(long long) * npes);
    }
    if (!ivar) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }
    if (!valid) {
        ERROR_PRINT("shmem_malloc failed \n");
        status = -1;
        goto out;
    }

    validOnHost = (char *)malloc(sizeof(long long) * npes);
    if (!validOnHost) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    waitBeginValOnHost = (char *)malloc(sizeof(long long) * npes);
    if (!waitBeginValOnHost) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    waitEndValOnHost = (char *)malloc(sizeof(long long) * npes);
    if (!waitEndValOnHost) {
        ERROR_PRINT("malloc failed \n");
        status = -1;
        goto out;
    }

    cudaStream_t cstream;
    CUDA_CHECK(cudaStreamCreate(&cstream));

#define TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_SWEEP(type, TYPE)          \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_EQ(type, TYPE, NVSHMEM_CMP_EQ) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GT(type, TYPE, NVSHMEM_CMP_GT) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LT(type, TYPE, NVSHMEM_CMP_LT) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_GE(type, TYPE, NVSHMEM_CMP_GE) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_LE(type, TYPE, NVSHMEM_CMP_LE) \
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_NE(type, TYPE, NVSHMEM_CMP_NE)

    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_SWEEP(int, int)
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_SWEEP(long, long)
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_SWEEP(longlong, long long)
    TEST_SHMEMX_TYPE_WAIT_UNTIL_ON_STREAM_SWEEP(short, short)

    CUDA_CHECK(cudaStreamDestroy(cstream));

    if (use_mmap) {
        free_mmap_buffer(ivar);
        free_mmap_buffer(valid);
    } else {
        nvshmem_free(ivar);
        nvshmem_free(valid);
    }
    free(validOnHost);
    free(waitBeginValOnHost);
    free(waitEndValOnHost);

out:
    finalize_wrapper();
    return status;
}
