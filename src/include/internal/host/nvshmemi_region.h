/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_H_
#define _NVSHMEMI_REGION_H_

#include "internal/host_transport/transport.h"

struct nvshmemi_state_dec;

bool nvshmemi_region_host_prepare_rma_attrs(nvshmem_transport_op_attrs_t *attrs);
int nvshmemi_region_flush(struct nvshmemi_state_dec *state, nvshmem_transport_region_role_t role,
                          uint64_t issuer_id, uint64_t region_id);
int nvshmemi_region_host_flush_active();

#endif
