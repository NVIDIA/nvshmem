/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nonuniform_teams.cuh"

// Force hierarchical selection below the automatic size threshold.
int main(int argc, char **argv) {
    setenv("NVSHMEM_BCAST_ALGO", "2", 1);
    init_wrapper(&argc, &argv);
    check_nonuniform_teams();
    finalize_wrapper();
    return 0;
}
