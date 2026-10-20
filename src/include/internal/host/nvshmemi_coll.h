/*
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/nvshmem_macros.h"
#include "non_abi/device/common/nvshmemi_runtime_types.h"

#ifndef NVSHMEMI_COLL_H
#define NVSHMEMI_COLL_H

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemi_barrier(nvshmem_team_t team);

#endif /* NVSHMEMI_COLL_H */
