/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

static bool validate_team(nvshmem_team_t team, const char *name, int mype, int npes) {
    const int team_npes = nvshmem_team_n_pes(team);
    const int team_mype = nvshmem_team_my_pe(team);
    int translated_npes = 0;
    bool mismatch = false;

    for (int pe = 0; pe < npes; pe++) {
        const int team_pe = nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, pe, team);
        if (team_pe == -1) {
            continue;
        }

        translated_npes++;
        if (team_pe < 0 || team_pe >= team_npes ||
            nvshmem_team_translate_pe(team, team_pe, NVSHMEM_TEAM_WORLD) != pe) {
            fprintf(stderr, "%s has an invalid translation for world PE %d\n", name, pe);
            mismatch = true;
        }
    }

    const int translated_mype = nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, mype, team);
    if (translated_npes != team_npes || translated_mype != team_mype) {
        fprintf(stderr,
                "%s membership mismatch on PE %d "
                "(translated size: %d, team size: %d, translated PE: %d, team PE: %d)\n",
                name, mype, translated_npes, team_npes, translated_mype, team_mype);
        mismatch = true;
    }

    return mismatch;
}

/* Verify that the predefined pointer and logical-endpoint teams have consistent membership. */

int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // dummy pointer to query PTR accessibility
    int *ptr = (int *)nvshmem_malloc(sizeof(int));

    bool membership_mismatch = false;
    for (int pe = 0; pe < npes; pe++) {
        const bool shared_accessible = nvshmem_ptr(ptr, pe) != nullptr;
        const bool shared_member =
            nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, pe, NVSHMEM_TEAM_SHARED) != -1;

        if (shared_accessible != shared_member) {
            fprintf(stderr,
                    "NVSHMEM_TEAM_SHARED mismatch between PE %d and PE %d "
                    "(accessible: %d, member: %d)\n",
                    mype, pe, shared_accessible, shared_member);
            membership_mismatch = true;
        }
    }

    membership_mismatch |=
        validate_team(NVSHMEM_TEAM_MC_SHARED, "NVSHMEM_TEAM_MC_SHARED", mype, npes);
    membership_mismatch |=
        validate_team(NVSHMEM_TEAM_CFT_SHARED, "NVSHMEM_TEAM_CFT_SHARED", mype, npes);
    membership_mismatch |=
        validate_team(NVSHMEM_TEAM_CFT_MC_SHARED, "NVSHMEM_TEAM_CFT_MC_SHARED", mype, npes);

    if (nvshmemx_mc_ptr(NVSHMEM_TEAM_CFT_MC_SHARED, ptr) != nullptr) {
        fprintf(stderr, "NVSHMEM_TEAM_CFT_MC_SHARED unexpectedly exposes a multicast pointer\n");
        membership_mismatch = true;
    }

    nvshmem_free(ptr);

    if (membership_mismatch) {
        fprintf(stderr, "Predefined team membership mismatches detected\n");
        finalize_wrapper();
        return 1;
    }

    finalize_wrapper();

    return 0;
}