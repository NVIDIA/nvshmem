/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

int main(int c, char *v[]) {
    int mype, status = 0;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    int npes = nvshmem_n_pes();
#endif

    DEBUG_PRINT("[%d of %d] hello world! \n", mype, npes);

    nvshmem_barrier_all();

    if (mype == 0) {
        nvshmem_global_exit(0);
        /* Note, this should be unreachable. return a unique error code if we reach here. */
        status = 2;
    } else {
        sleep(60);
        fprintf(stderr, "Was able to get to the end of the test.\n");
        finalize_wrapper();
        return 1;
    }

    return status;
}
