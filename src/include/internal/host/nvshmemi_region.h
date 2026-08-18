/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_REGION_H_
#define _NVSHMEMI_REGION_H_

#include "internal/host_transport/transport.h"
#include "non_abi/nvshmemi_region_constants.h"

struct nvshmemi_state_dec;

extern uint32_t nvshmemi_region_host_active_hints;

bool nvshmemi_region_host_prepare_rma_attrs(nvshmem_transport_op_attrs_t *attrs);
int nvshmemi_region_flush(struct nvshmemi_state_dec *state, nvshmem_transport_region_role_t role,
                          uint64_t issuer_id, uint64_t region_id);
int nvshmemi_region_host_flush_active();
void nvshmemi_region_host_reset();

#endif
