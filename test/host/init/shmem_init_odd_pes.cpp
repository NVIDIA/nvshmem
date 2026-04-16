/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Regression test for bootstrap_shmem_allgather() with non-4-byte-aligned
 * payloads on an odd number of PEs.
 *
 * Bug: bootstrap_shmem_allgather() called shmem_collect32 with (length / 4)
 * as the element count, which rounds down to 0 for sub-4-byte payloads (e.g.
 * bool).  With an odd PE count the symmetric-memory algorithm in
 * shmem_collect32 would then write out-of-bounds or corrupt data during
 * nvshmemx_init_attr().
 *
 * Fix: round length up to the nearest multiple of 4 before dividing.
 *
 * This test must be run with an ODD number of PEs.  It exits with status 77
 * (skip) when an even number of PEs is detected, so CI can flag the result
 * appropriately.  With the buggy code it crashes or hangs during
 * nvshmemx_init_attr(); with the fix it completes normally.
 */

#include <stdio.h>
#include <stdlib.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "utils.h"

/* Exit code conventionally used by automake/ctest to signal "skipped". */
#define TEST_SKIP_EXIT_CODE 77

int main(int c, char *v[]) {
    int nv_npes_node, nv_mype_node;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int dev_count;

    shmem_init();

    int shmem_npes = shmem_n_pes();
    int shmem_mype = shmem_my_pe();

    /* This test specifically exercises the odd-PE code path.  Skip cleanly
     * when an even number of PEs is detected so the result is not silently
     * ignored. */
    if (shmem_npes % 2 == 0) {
        if (shmem_mype == 0) {
            fprintf(stderr,
                    "shmem_init_odd_pes: SKIP – requires an odd number of PEs "
                    "(got %d).  Re-run with e.g. 3 or 5 PEs.\n",
                    shmem_npes);
        }
        shmem_finalize();
        exit(TEST_SKIP_EXIT_CODE);
    }

    /*
     * nvshmemx_init_attr() internally calls bootstrap_shmem_allgather() with
     * sub-4-byte payloads (bool uid data).  With an odd PE count this
     * triggered out-of-bounds writes in the legacy shmem_collect32 path.
     * The call below will crash/hang on buggy builds and succeed with the fix.
     */
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

    nv_mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nv_npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (nv_npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(nv_mype_node / npes_per_gpu));

    nvshmem_barrier_all();
    shmem_barrier_all();

    nvshmem_finalize();
    shmem_finalize();

    return 0;
}
