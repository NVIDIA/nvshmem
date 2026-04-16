/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include "nvshmem.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern "C" {
void simplelib1_init();
int simplelib1_dowork();
void simplelib1_finalize();
}
