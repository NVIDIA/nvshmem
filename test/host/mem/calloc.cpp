/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "utils.h"

#define NELEM (1 << 20)

int main(int argc, char **argv) {
    long *dev_buf, *host_buf;
    size_t i, err = 0;

    init_wrapper(&argc, &argv);

    /* Check count == 0 */
    dev_buf = (long *)nvshmem_calloc(0, sizeof(long));
    if (dev_buf != NULL) {
        printf("Error, zero element calloc did not return NULL\n");
        ++err;
    }
    nvshmem_free(dev_buf);

    /* Check size == 0 */
    dev_buf = (long *)nvshmem_calloc(NELEM, 0);
    if (dev_buf != NULL) {
        printf("Error, zero size calloc did not return NULL\n");
        ++err;
    }
    nvshmem_free(dev_buf);

    /* Check that memory is cleared: calloc, set, free, calloc */
    dev_buf = (long *)nvshmem_calloc(NELEM, sizeof(long));
    cudaMemset(dev_buf, 0xAA, NELEM * sizeof(long));
    nvshmem_free(dev_buf);

    host_buf = (long *)calloc(NELEM, sizeof(long));
    dev_buf = (long *)nvshmem_calloc(NELEM, sizeof(long));
    cudaMemcpy(host_buf, dev_buf, NELEM * sizeof(long), cudaMemcpyDeviceToHost);

    for (i = 0; i < NELEM; i++)
        if (host_buf[i]) ++err;

    free(host_buf);
    nvshmem_free(dev_buf);
    finalize_wrapper();

    return err != 0;
}
