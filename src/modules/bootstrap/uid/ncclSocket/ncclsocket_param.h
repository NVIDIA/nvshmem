/*************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_PARAM_H_
#define NCCL_SOCKET_PARAM_H_

#include <stdlib.h>  // for getenv

static inline const char *ncclGetEnv(const char *name) {
  return getenv(name);
}

#endif