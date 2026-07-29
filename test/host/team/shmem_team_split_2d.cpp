/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

static int check_2d(nvshmem_team_t parent_team, int xdim) {
    int me = nvshmem_team_my_pe(parent_team);

    nvshmem_team_t xteam = NVSHMEM_TEAM_INVALID;
    nvshmem_team_t yteam = NVSHMEM_TEAM_INVALID;

    int ret = nvshmem_team_split_2d(parent_team, xdim, NULL, 0, &xteam, NULL, 0, &yteam);
    int errors = 0;

    if (ret == 0) {
        int me_x = nvshmem_team_my_pe(xteam);
        int me_y = nvshmem_team_my_pe(yteam);
        int npes_x = nvshmem_team_n_pes(xteam);
        int npes_y = nvshmem_team_n_pes(yteam);

        if (xteam == NVSHMEM_TEAM_INVALID || yteam == NVSHMEM_TEAM_INVALID) {
            printf("%d: Error, received an invalid team\n", nvshmem_my_pe());
            ++errors;
        }

        /* Try converting the PE ids from xteam and yteam to parent and global
         * PE indices and compare with the expected indices */
        for (int i = 0; i < npes_x; i++) {
            int expected_parent = me_y * xdim + i; /* row (fixed) + column */
            int pe_parent = nvshmem_team_translate_pe(xteam, i, parent_team);
            int pe_world = nvshmem_team_translate_pe(xteam, i, NVSHMEM_TEAM_WORLD);
            int expected_world =
                nvshmem_team_translate_pe(parent_team, expected_parent, NVSHMEM_TEAM_WORLD);

            if (expected_parent != pe_parent) {
                printf("%d: xteam[%d] expected parent PE id %d, got %d\n", me, i, expected_parent,
                       pe_parent);
                errors++;
            }

            if (expected_world != pe_world) {
                printf("%d: xteam[%d] expected world PE id %d, got %d\n", me, i, expected_world,
                       pe_world);
                errors++;
            }
        }

        for (int i = 0; i < npes_y; i++) {
            int expected_parent = i * xdim + me_x; /* row + column (fixed) */
            int pe_parent = nvshmem_team_translate_pe(yteam, i, parent_team);
            int pe_world = nvshmem_team_translate_pe(yteam, i, NVSHMEM_TEAM_WORLD);
            int expected_world =
                nvshmem_team_translate_pe(parent_team, expected_parent, NVSHMEM_TEAM_WORLD);

            if (expected_parent != pe_parent) {
                printf("%d: yteam[%d] expected parent PE id %d, got %d\n", me, i, expected_parent,
                       pe_parent);
                errors++;
            }

            if (expected_world != pe_world) {
                printf("%d: yteam[%d] expected world PE id %d, got %d\n", me, i, expected_world,
                       pe_world);
                errors++;
            }
        }
    } else {
        printf("%d: 2d split failed, xdim: %d\n", nvshmem_my_pe(), xdim);
    }

    if (xteam != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(xteam);
    if (yteam != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(yteam);

    return errors != 0;
}

static int check_2d_destroy_reuse(nvshmem_team_t parent_team, int xdim, int iterations) {
    int errors = 0;

    for (int i = 0; i < iterations; ++i) {
        nvshmem_team_t xteam = NVSHMEM_TEAM_INVALID;
        nvshmem_team_t yteam = NVSHMEM_TEAM_INVALID;

        int ret = nvshmem_team_split_2d(parent_team, xdim, NULL, 0, &xteam, NULL, 0, &yteam);
        if (ret != 0 || xteam == NVSHMEM_TEAM_INVALID || yteam == NVSHMEM_TEAM_INVALID) {
            printf("%d: 2d split reuse failed on iteration %d, ret: %d, xteam: %d, yteam: %d\n",
                   nvshmem_my_pe(), i, ret, xteam, yteam);
            ++errors;
            break;
        }

        nvshmem_team_destroy(xteam);
        nvshmem_team_destroy(yteam);
    }

    return errors != 0;
}

int main(int argc, char **argv) {
    int errors = 0, me, npes, ret;
    nvshmem_team_t even_team = NVSHMEM_TEAM_INVALID;

    setenv("NVSHMEM_MAX_CTAS", "2", 1);
    setenv("NVSHMEM_MAX_TEAMS", "128", 1);

    init_wrapper(&argc, &argv);

    me = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (me == 0) printf("Performing 2d split test on NVSHMEM_TEAM_WORLD\n");

    errors += check_2d(NVSHMEM_TEAM_WORLD, 1);
    errors += check_2d(NVSHMEM_TEAM_WORLD, 2);
    errors += check_2d(NVSHMEM_TEAM_WORLD, 3);

    ret = nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 2, (npes - 1) / 2 + 1, NULL, 0,
                                     &even_team);

    if (ret == 0) {
        if (me == 0) printf("Performing 2d split test on even team\n");

        errors += check_2d(even_team, 1);
        errors += check_2d(even_team, 2);
        errors += check_2d(even_team, 3);
        nvshmem_team_destroy(even_team);
    } else {
        if (me == 0) printf("Unable to create even team\n");
    }

    errors += check_2d_destroy_reuse(NVSHMEM_TEAM_WORLD, 2, 40);

    finalize_wrapper();
    return errors != 0;
}
