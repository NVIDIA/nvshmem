/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "utils.h"

int main(int argc, char **argv) {
    int i, me, npes;
    int ret = 0, errors = 0;

    init_wrapper(&argc, &argv);

    me = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (me == 0) printf("Reuse teams test\n");

    nvshmem_team_t old_team, new_team;
    ret = nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, NULL, 0, &old_team);
    if (ret) ++errors;

    /* A total of npes-1 iterations are performed, where the active set in iteration i
     * includes PEs i..npes-1.  The size of the team decreases by 1 each iteration.  */
    for (i = 1; i < npes; i++) {
        if (me == i) {
            printf("%3d: creating new team (start, stride, size): %3d, %3d, %3d\n", me,
                   nvshmem_team_translate_pe(old_team, 1, NVSHMEM_TEAM_WORLD), 1,
                   nvshmem_team_n_pes(old_team) - 1);
        }

        ret = nvshmem_team_split_strided(old_team, 1, 1, nvshmem_team_n_pes(old_team) - 1, NULL, 0,
                                         &new_team);
        if (old_team != NVSHMEM_TEAM_INVALID && ret) ++errors;

        nvshmem_team_destroy(old_team);
        old_team = new_team;
    }
    nvshmem_team_destroy(old_team);
    finalize_wrapper();

    return errors != 0;
}
