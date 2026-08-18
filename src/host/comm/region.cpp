/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

#include "host/nvshmemx_api.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_region.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "internal/host/nvshmemi_types.h"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmemi_region_constants.h"
#include "non_abi/nvshmemx_error.h"

namespace {
bool nvshmemi_region_hints_are_valid(uint32_t hints) {
    constexpr uint32_t supported_hints = NVSHMEMX_REGION_HINT_BATCH_RMA;
    return (hints & ~supported_hints) == 0;
}

class nvshmemi_region_host_state {
   public:
    bool active() const { return active_; }
    bool has_hints(uint32_t hints) const {
        return active_ && (hints == NVSHMEMX_REGION_HINT_NONE || (hints_ & hints) == hints);
    }
    uint64_t region_id() const { return region_id_; }

    void start(uint64_t region_id, uint32_t hints) {
        region_id_ = region_id;
        hints_ = hints;
        active_ = true;
    }

    void reset() {
        active_ = false;
        region_id_ = 0;
        hints_ = NVSHMEMX_REGION_HINT_NONE;
    }

   private:
    uint64_t region_id_ = 0;
    uint32_t hints_ = NVSHMEMX_REGION_HINT_NONE;
    bool active_ = false;
};

constexpr uint64_t nvshmemi_region_host_issuer_id = 1;
/* Host regions are PE scoped. NVSHMEM's host API thread-funneled requirement makes additional
 * synchronization here unnecessary. */
nvshmemi_region_host_state region_state;
uint64_t next_region_id = 1;

uint64_t nvshmemi_region_next_id() {
    uint64_t id = next_region_id++;
    return id ? id : next_region_id++;
}
}  // namespace

/* Avoid an out-of-line region-state query on the common no-region host RMA path. */
uint32_t nvshmemi_region_host_active_hints = NVSHMEMX_REGION_HINT_NONE;

bool nvshmemi_region_host_prepare_rma_attrs(nvshmem_transport_op_attrs_t *attrs) {
    if (!region_state.has_hints(NVSHMEMI_REGION_HINT_BATCH_RMA)) {
        return false;
    }

    attrs->flags = NVSHMEM_TRANSPORT_OP_FLAG_NONE;
    attrs->hints = NVSHMEMI_REGION_HINT_BATCH_RMA;
    attrs->issuer_id = nvshmemi_region_host_issuer_id;
    attrs->region_id = region_state.region_id();
    return true;
}

int nvshmemi_region_flush(nvshmemi_state_t *state, nvshmem_transport_region_role_t role,
                          uint64_t issuer_id, uint64_t region_id) {
    int tbitmap = state->transport_bitmap;
    for (int j = 0; j < state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr = state->transports[j];
            if (tcurr->host_ops.region_flush) {
                int status = tcurr->host_ops.region_flush(tcurr, role, issuer_id, region_id);
                if (status) {
                    return status;
                }
            }
        }
        tbitmap >>= 1;
    }
    return NVSHMEMX_SUCCESS;
}

int nvshmemi_region_host_flush_active() {
    if (!region_state.has_hints(NVSHMEMI_REGION_HINT_BATCH_RMA)) {
        return NVSHMEMX_SUCCESS;
    }

    TRACE(NVSHMEM_P2P, "Host batched RMA region flush: issuer=%llu region=%llu",
          static_cast<unsigned long long>(nvshmemi_region_host_issuer_id),
          static_cast<unsigned long long>(region_state.region_id()));

    int status = nvshmemi_region_flush(nvshmemi_state, NVSHMEM_TRANSPORT_REGION_ROLE_HOST,
                                       nvshmemi_region_host_issuer_id, region_state.region_id());
    return status;
}

void nvshmemi_region_host_reset() {
    region_state.reset();
    nvshmemi_region_host_active_hints = NVSHMEMX_REGION_HINT_NONE;
}

int nvshmemx_region_start(nvshmemx_region_handle_t *handle, const nvshmemx_region_attrs_t *attrs) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();

    if (handle == nullptr) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    uint32_t hints =
        attrs == nullptr ? static_cast<uint32_t>(NVSHMEMX_REGION_HINT_NONE) : attrs->hints;
    if (!nvshmemi_region_hints_are_valid(hints)) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (region_state.active()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    uint64_t region_id = nvshmemi_region_next_id();
    region_state.start(region_id, hints);
    nvshmemi_region_host_active_hints = hints;
    *handle = region_id;

    TRACE(NVSHMEM_P2P, "Host region start: issuer=%llu region=%llu hints=0x%x",
          static_cast<unsigned long long>(nvshmemi_region_host_issuer_id),
          static_cast<unsigned long long>(region_id), hints);

    return NVSHMEMX_SUCCESS;
}

int nvshmemx_region_stop(nvshmemx_region_handle_t handle) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status;

    if (!region_state.active() || handle != region_state.region_id()) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    status = nvshmemi_region_host_flush_active();
    if (status) {
        return status;
    }

    TRACE(NVSHMEM_P2P, "Host region stop: issuer=%llu region=%llu",
          static_cast<unsigned long long>(nvshmemi_region_host_issuer_id),
          static_cast<unsigned long long>(region_state.region_id()));

    nvshmemi_region_host_reset();

    return NVSHMEMX_SUCCESS;
}

int nvshmemx_region_is_active(uint32_t hints, int *active) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();

    if (active == nullptr) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }
    if (!nvshmemi_region_hints_are_valid(hints)) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    *active = region_state.has_hints(hints) ? 1 : 0;
    return NVSHMEMX_SUCCESS;
}
