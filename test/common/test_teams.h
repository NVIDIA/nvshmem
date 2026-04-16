/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TEST_TEAMS_H
#define TEST_TEAMS_H

#include "nvshmem.h"
#include "nvshmemx.h"
#include <algorithm>
#include <unordered_map>
#include <string>

extern std::unordered_map<nvshmem_team_t, std::string> map_team_to_string;
extern std::unordered_map<std::string, nvshmem_team_t> map_string_to_team;

bool get_next_team(nvshmem_team_t *team);
void init_test_teams();
void finalize_test_teams();

#endif /* TEST_TEAMS_H */
