/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include "nvshmem.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern "C" {
void simplelib2_init();
int simplelib2_dowork();
void simplelib2_finalize();
}
