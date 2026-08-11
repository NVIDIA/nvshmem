/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_teams.h"
#include "utils.h"
#include <string>
#include <algorithm>
#include <unordered_map>

using namespace std;

#define MAX_TEAMS 100

nvshmem_team_t *teams;
string *team_names;
unordered_map<nvshmem_team_t, string> map_team_to_string;
unordered_map<string, nvshmem_team_t> map_string_to_team;
nvshmemx_team_uniqueid_t *uniqueid, *uniqueid_device;
int num_teams = 0;
int team_idx;

void set_team_mappings(nvshmem_team_t team, string name) {
    map_team_to_string[team] = name;
    map_string_to_team[name] = team;
    // printf("team:name = %d:%s\n", (int)team, name.c_str());
}

static void share_team_uid(nvshmem_team_config_t *config, int *team_members, int npes,
                           bool in_team) {
    int my_pe = nvshmem_my_pe();

    if (team_members[0] == my_pe) {
        if (nvshmemx_team_get_uniqueid(uniqueid) != 0) {
            printf("Failed to get uniqueid for team_members[%d]\n", my_pe);
            exit(1);
        }
        cudaMemcpy(uniqueid_device, uniqueid, sizeof(nvshmemx_team_uniqueid_t),
                   cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 1; i < npes; i++) {
            nvshmem_putmem_nbi(uniqueid_device, uniqueid_device, sizeof(nvshmemx_team_uniqueid_t),
                               team_members[i]);
        }
    }

    nvshmem_barrier_all();

    if (team_members[0] != my_pe && in_team) {
        CUDA_CHECK(cudaMemcpy(uniqueid, uniqueid_device, sizeof(nvshmemx_team_uniqueid_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (in_team) {
        config->uniqueid = *uniqueid;
    }
}

static int create_uneven_stride_team(nvshmem_team_t *teams, int *ctr, bool increasing,
                                     bool reverse) {
    nvshmem_team_config_t *config;

    config = (nvshmem_team_config_t *)malloc(sizeof(nvshmem_team_config_t));
    *config = NVSHMEM_TEAM_CONFIG_INITIALIZER;

    int status = 0;
    int my_pe = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int *team_members = (int *)malloc(npes * sizeof(int));
    int iterator_max;
    int iterator_min;
    int my_idx_in_team;
    int team_size = 0;
    int stride;

    if (increasing) {
        stride = 0;
    } else {
        stride = npes / 2 + 1;
    }

    bool in_team = false;

    if (team_members == NULL) {
        printf("Failed to allocate team_members\n");
        exit(1);
    }

    if (reverse) {
        iterator_max = 0;
        iterator_min = npes - 1;
        for (int i = iterator_min; i >= iterator_max; i -= stride) {
            if (i == my_pe) {
                in_team = true;
                my_idx_in_team = team_size;
            }
            team_members[team_size] = i;
            if (increasing) {
                stride++;
            } else {
                stride--;
            }
            team_size++;
            if (stride <= 0) {
                break;
            }
        }
    } else {
        iterator_max = npes;
        iterator_min = 0;
        for (int i = iterator_min; i < iterator_max; i += stride) {
            if (i == my_pe) {
                in_team = true;
                my_idx_in_team = team_size;
            }
            team_members[team_size] = i;
            if (increasing) {
                stride++;
            } else {
                stride--;
            }
            team_size++;
            if (stride <= 0) {
                break;
            }
        }
    }

    share_team_uid(config, team_members, team_size, in_team);

    if (in_team) {
        if (nvshmemx_team_init(&teams[*ctr], config, NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID, team_size,
                               my_idx_in_team)) {
            fprintf(stderr, "Failed to create uneven strided team[%d]\n", *ctr);
            status = -1;
            goto cleanup;
        }
        if (increasing) {
            set_team_mappings(teams[*ctr], "TEAM_SPLIT_INCREASING_STRIDE");
        } else {
            set_team_mappings(teams[*ctr], "TEAM_SPLIT_DECREASING_STRIDE");
        }
    } else {
        teams[*ctr] = NVSHMEM_TEAM_INVALID;
    }

cleanup:
    (*ctr)++;
    free(team_members);
    free(config);
    return status;
}

void init_test_teams() {
    DEBUG_PRINT("In init_test_teams\n");
    int n_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);

    teams = (nvshmem_team_t *)malloc(sizeof(nvshmem_team_t) * MAX_TEAMS);

    uniqueid = (nvshmemx_team_uniqueid_t *)malloc(sizeof(nvshmemx_team_uniqueid_t));
    uniqueid_device = (nvshmemx_team_uniqueid_t *)nvshmem_malloc(sizeof(nvshmemx_team_uniqueid_t));
    int ctr = 0;

    /* Set pre-defined teams */
    teams[ctr] = NVSHMEM_TEAM_WORLD;
    set_team_mappings(teams[ctr], "NVSHMEM_TEAM_WORLD");
    ctr++;

    teams[ctr] = NVSHMEM_TEAM_SHARED;
    set_team_mappings(teams[ctr], "NVSHMEM_TEAM_SHARED");
    ctr++;

    teams[ctr] = NVSHMEM_TEAM_MC_SHARED;
    set_team_mappings(teams[ctr], "NVSHMEM_TEAM_MC_SHARED");
    ctr++;

    teams[ctr] = NVSHMEMX_TEAM_NODE;
    set_team_mappings(teams[ctr], "NVSHMEMX_TEAM_NODE");
    ctr++;

    /** Create teams using team_split_strided API **/

    DEBUG_PRINT("Create arbitrary teams start");
    /* increasing stride */
    create_uneven_stride_team(teams, &ctr, true, false);

    /* decreasing stride */
    create_uneven_stride_team(teams, &ctr, false, false);

    /* unordered stride */
    create_uneven_stride_team(teams, &ctr, true, true);

    DEBUG_PRINT("split_strided start\n");
    /* All even PEs */
    nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 2, max(1, n_pes / 2), NULL, 0, &teams[ctr]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_EVEN");
    ctr++;

    /* All odd PEs */
    if (n_pes > 1) {
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 1, 2, n_pes / 2, NULL, 0, &teams[ctr]);
        set_team_mappings(teams[ctr], "TEAM_SPLIT_ODD");
        ctr++;
    }
    /* Multiple of 3 PEs */
    nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 3, max(1, n_pes / 3), NULL, 0, &teams[ctr]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_MULTIPLE_OF_3");
    ctr++;

    /** Create teams using team_split_2d **/

    /* xrange = 1 */
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, 1, NULL, 0, &teams[ctr], NULL, 0, &teams[ctr + 1]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_2D_1_X");
    set_team_mappings(teams[ctr + 1], "TEAM_SPLIT_2D_1_Y");
    ctr += 2;

    /* xrange = n_pes */
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, n_pes, NULL, 0, &teams[ctr], NULL, 0,
                          &teams[ctr + 1]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_2D_NPES_X");
    set_team_mappings(teams[ctr + 1], "TEAM_SPLIT_2D_NPES_Y");
    ctr += 2;

    /* xrange = n_pes/2 */
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, max(1, n_pes / 2), NULL, 0, &teams[ctr], NULL, 0,
                          &teams[ctr + 1]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_2D_NPESBY2_X");
    set_team_mappings(teams[ctr + 1], "TEAM_SPLIT_2D_NPESBY2_Y");
    ctr += 2;

    /* xrange = n_pes/3 */
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, max(1, n_pes / 3), NULL, 0, &teams[ctr], NULL, 0,
                          &teams[ctr + 1]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_2D_NPESBY3_X");
    set_team_mappings(teams[ctr + 1], "TEAM_SPLIT_2D_NPESBY3_Y");
    ctr += 2;

    /** Create teams by further splitting user-defined teams **/

    nvshmem_team_t parent_team = map_string_to_team["TEAM_SPLIT_EVEN"];
    int parent_team_npes = nvshmem_team_n_pes(parent_team);
    nvshmem_team_split_2d(parent_team, max(1, parent_team_npes / 2), NULL, 0, &teams[ctr], NULL, 0,
                          &teams[ctr + 1]);
    set_team_mappings(teams[ctr], "TEAM_SPLIT_EVEN_SPLIT_2D_NPESBY2_X");
    set_team_mappings(teams[ctr + 1], "TEAM_SPLIT_EVEN_SPLIT_2D_NPESBY2_Y");
    ctr += 2;

    num_teams = ctr;
    team_idx = 0;
    free(uniqueid);
    nvshmem_free(uniqueid_device);
    DEBUG_PRINT("teams created = %d\n", num_teams);
}

bool get_next_team(nvshmem_team_t *team) {
    if (team_idx < num_teams) {
        *team = teams[team_idx++];
        return 1;
    } else {
        *team = NVSHMEM_TEAM_INVALID;
        return 0;
    }
}

void finalize_test_teams() {
    for (int i = 0; i < num_teams; i++) {
        if (map_team_to_string[teams[i]] != "NVSHMEM_TEAM_WORLD" &&
            map_team_to_string[teams[i]] != "NVSHMEM_TEAM_SHARED" &&
            map_team_to_string[teams[i]] != "NVSHMEM_TEAM_MC_SHARED" &&
            map_team_to_string[teams[i]] != "NVSHMEMX_TEAM_NODE") {
            nvshmem_team_destroy(teams[i]);
        }
    }
}
