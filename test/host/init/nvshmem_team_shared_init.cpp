/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

/* This test verifies that NVSHMEM_TEAM_SHARED is initialized correctly by checking the equivalence
 * of the P2P accessibility using nvshmem_ptr and nvshmem_team_translate_pe. */

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // dummy pointer to query PTR accessibility
    int *ptr = (int *)nvshmem_malloc(sizeof(int));

    // Check for equivalence P2P accessibility using nvshmem_ptr and nvshmem_team_translate_pe
    bool p2p_accessible_mismatch = false;
    for (int pe = 0; pe < npes; pe++) {
        bool p2p_accessible_ptr = nvshmem_ptr(ptr, pe) != nullptr;
        bool p2p_accessible_translate_pe =
            nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, pe, NVSHMEM_TEAM_SHARED) != -1;

        if (p2p_accessible_ptr != p2p_accessible_translate_pe) {
            fprintf(stderr,
                    "P2P accessibility mismatch detected between PE %d and PE %d (ptr: %d, "
                    "translate_pe: %d)\n",
                    mype, pe, p2p_accessible_ptr, p2p_accessible_translate_pe);
            p2p_accessible_mismatch = true;
        }
    }

    nvshmem_free(ptr);

    if (p2p_accessible_mismatch) {
        fprintf(stderr, "P2P accessibility mismatches detected\n");
        finalize_wrapper();
        return 1;
    }

    finalize_wrapper();

    return 0;
}