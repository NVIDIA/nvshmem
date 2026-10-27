/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nonuniform_teams.cuh"

// The existing launcher defines participation for each supported bootstrap.
int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);
    check_nonuniform_teams();
    finalize_wrapper();
    return 0;
}
