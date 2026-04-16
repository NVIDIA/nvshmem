/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "wait.cubin"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define THREADS 512

#define TEST_NVSHMEM_WAIT_CUBIN()                                                          \
    void *args_wait[] = {(void *)&sig_addr, (void *)&mype,   (void *)&npes, (void *)&cmp,  \
                         (void *)&setval,   (void *)&cmpval, (void *)&ret};                \
    CUfunction test_wait_cubin;                                                            \
    init_test_case_kernel(&test_wait_cubin, NVSHMEMI_TEST_STRINGIFY(perform_signal_wait)); \
    CU_CHECK(cuLaunchKernel(test_wait_cubin, 1, 1, 1, THREADS, 1, 1, 0, 0, args_wait, NULL));

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void perform_signal_wait(uint64_t *sig_addr, int mype, int npes, int cmp,
                                    uint64_t setval, uint64_t cmpval, int *ret) {
    int tid = threadIdx.x;
    int cpe = (mype + 1) % npes;
    int err;
    uint64_t *my_sig_addr = sig_addr + tid;
    uint64_t rc;

    nvshmemx_signal_op(my_sig_addr, setval, NVSHMEM_SIGNAL_SET, cpe);
    rc = nvshmem_signal_wait_until(my_sig_addr, cmp, cmpval);

    switch (cmp) {
        case NVSHMEM_CMP_EQ:
            err = rc == cmpval ? 0 : 1;
            break;
        case NVSHMEM_CMP_NE:
            err = rc != cmpval ? 0 : 1;
            break;
        case NVSHMEM_CMP_GT:
            err = rc > cmpval ? 0 : 1;
            break;
        case NVSHMEM_CMP_GE:
            err = rc >= cmpval ? 0 : 1;
            break;
        case NVSHMEM_CMP_LT:
            err = rc < cmpval ? 0 : 1;
            break;
        case NVSHMEM_CMP_LE:
            err = rc <= cmpval ? 0 : 1;
            break;
        default:
            err = 1;
            break;
    }
    if (err) {
        *ret = 1;
    }
}

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

static int wait_loop(uint64_t *sig_addr, int mype, int npes, int cmp, uint64_t cmpval, int *ret,
                     uint64_t setval) {
    int host_kern_status = 0;
    if (use_cubin) {
        TEST_NVSHMEM_WAIT_CUBIN();
    } else {
        perform_signal_wait<<<1, THREADS, 0>>>(sig_addr, mype, npes, cmp, setval, cmpval, ret);
    }

    CUDA_CHECK(cudaMemcpy(&host_kern_status, ret, sizeof(int), cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();
    if (host_kern_status != 0) {
        return -1;
    }

    return 0;
}

int main(int argc, char *argv[]) {
    uint64_t *signal_addr = NULL;
    int npes, mype, status = 0;
    int *kern_status = NULL;
    read_args(argc, argv);

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
    }

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (use_mmap) {
        signal_addr = (uint64_t *)allocate_mmap_buffer(sizeof(uint64_t) * THREADS, _mem_handle_type,
                                                       use_egm, true);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        signal_addr = (uint64_t *)nvshmem_calloc(THREADS, sizeof(uint64_t));
    }
    if (!signal_addr) {
        ERROR_PRINT("shmem_calloc failed.\n");
        status = -1;
        goto out;
    }

    kern_status = (int *)nvshmem_calloc(1, sizeof(int));
    if (!kern_status) {
        ERROR_PRINT("shmem_calloc failed.\n");
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_EQ, 365, kern_status, 365)) {
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_LT, 365, kern_status, 301)) {
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_LE, 300, kern_status, 300)) {
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_NE, 300, kern_status, 400)) {
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_GE, 500, kern_status, 500)) {
        status = -1;
        goto out;
    }

    if (wait_loop(signal_addr, mype, npes, NVSHMEM_CMP_GT, 500, kern_status, 501)) {
        status = -1;
        goto out;
    }

out:
    finalize_wrapper();
    return status;
}
