/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NCCL_SOCKET_PARAM_H_
#define NCCL_SOCKET_PARAM_H_

#include <stdlib.h>  // for getenv

static inline const char *ncclGetEnv(const char *name) {
  return getenv(name);
}

#endif