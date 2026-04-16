/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DATA_CHECK_H_
#define _DATA_CHECK_H_

#include "cuda_runtime.h"
template <typename T>
int init_data_ring(T *buf, size_t size, int disp, int iters, int mype, int npes, int *nextpe,
                   int *prevpe, int seed, cudaStream_t cstrm);
template <typename T>
int init_data_alltoall(T *buf, size_t size, int disp, int iters, int mype, int npes, int seed,
                       cudaStream_t cstrm);
template <typename T>
int check_data_ring(T *buf, cudaStream_t);
template <typename T>
int check_data_alltoall(T *buf, cudaStream_t);

#endif
