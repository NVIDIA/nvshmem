/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_CFT_H
#define NVSHMEMI_CFT_H

#include <stdbool.h>

#if defined(__cplusplus)
extern "C" {
#endif

bool nvshmemid_is_cft_p_on_stream_supported(int pe);

#if defined(__cplusplus)
}
#endif

#endif
