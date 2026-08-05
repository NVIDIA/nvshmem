/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMT_TRANSPORT_BATCH_RMA_HPP_
#define _NVSHMEMT_TRANSPORT_BATCH_RMA_HPP_

#include <array>
#include <memory>
#include <new>
#include <vector>

#include "internal/host_transport/region.hpp"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmemi_region_constants.h"

struct nvshmemt_batch_rma_entry {
    int pe;
    rma_verb_t verb;
    rma_memdesc_t remote;
    rma_memdesc_t local;
    rma_bytesdesc_t bytes;
    int qp_index;
    void *transport_domain;
};

struct nvshmemt_batch_rma_region {
    std::vector<nvshmemt_batch_rma_entry> entries;
    size_t next_entry = 0;
    size_t op_count = 0;
};

static constexpr size_t nvshmemt_batch_rma_initial_region_capacity = 8;
static constexpr size_t nvshmemt_region_table_probe_limit = NVSHMEMI_REGION_SLOT_PROBE_LIMIT;

static inline nvshmem_transport_region_role_t nvshmemt_region_role_from_qp_index(int qp_index) {
    return qp_index == NVSHMEMX_QP_HOST ? NVSHMEM_TRANSPORT_REGION_ROLE_HOST
                                        : NVSHMEM_TRANSPORT_REGION_ROLE_PROXY;
}

static inline bool nvshmemt_region_role_is_valid(nvshmem_transport_region_role_t role) {
    return role == NVSHMEM_TRANSPORT_REGION_ROLE_HOST ||
           role == NVSHMEM_TRANSPORT_REGION_ROLE_PROXY;
}

static inline std::array<size_t, NVSHMEM_TRANSPORT_REGION_ROLE_COUNT>
nvshmemt_region_table_capacities(size_t proxy_capacity) {
    std::array<size_t, NVSHMEM_TRANSPORT_REGION_ROLE_COUNT> capacities{};
    capacities[NVSHMEM_TRANSPORT_REGION_ROLE_HOST] = 1;
    capacities[NVSHMEM_TRANSPORT_REGION_ROLE_PROXY] = proxy_capacity;
    return capacities;
}

using nvshmemt_batch_rma_lifecycle =
    nvshmemi_region_lifecycle<nvshmemt_batch_rma_region, NVSHMEM_TRANSPORT_REGION_ROLE_COUNT>;

class nvshmemt_batch_rma_state {
    nvshmemt_batch_rma_lifecycle regions_;
    size_t max_batch_ops_;

   public:
    nvshmemt_batch_rma_state(size_t capacity, size_t max_batch_ops)
        : regions_(nvshmemt_region_table_capacities(capacity), nvshmemt_region_table_probe_limit),
          max_batch_ops_(max_batch_ops) {}

    template <typename Submit>
    int try_accumulate(int pe, rma_verb_t verb, rma_memdesc_t *remote, rma_memdesc_t *local,
                       rma_bytesdesc_t bytes, int qp_index,
                       const nvshmem_transport_op_attrs_t *attrs, void *transport_domain,
                       Submit submit, bool *handled) noexcept {
        *handled = false;
        if (attrs == nullptr || (attrs->hints & NVSHMEMI_REGION_HINT_BATCH_RMA) == 0 ||
            attrs->issuer_id == 0 || attrs->region_id == 0) {
            return 0;
        }

        nvshmemt_batch_rma_region *region = nullptr;
        try {
            nvshmem_transport_region_role_t role = nvshmemt_region_role_from_qp_index(qp_index);
            nvshmemi_region_key key = {attrs->issuer_id, attrs->region_id};
            region = regions_.find_or_create(role, key);
            if (region == nullptr) {
                return 0;
            }

            if (region->entries.capacity() == 0) {
                region->entries.reserve(nvshmemt_batch_rma_initial_region_capacity);
            }
            region->entries.push_back(
                {pe, verb, *remote, *local, bytes, qp_index, transport_domain});
            *handled = true;
            if ((attrs->flags & NVSHMEM_TRANSPORT_OP_FLAG_MORE_FOLLOWS) == 0) {
                region->op_count++;
            }
            if (max_batch_ops_ != 0 && region->op_count >= max_batch_ops_) {
                int status = submit(*region);
                if (status) {
                    return status;
                }
                region->entries.clear();
                region->next_entry = 0;
                region->op_count = 0;
            }
            return 0;
        } catch (const std::bad_alloc &) {
            if (region != nullptr) {
                int status = submit(*region);
                if (status) {
                    return status;
                }
                region->entries.clear();
                region->next_entry = 0;
                region->op_count = 0;
            }
            *handled = false;
            return 0;
        } catch (...) {
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }

    template <typename Submit>
    int flush(nvshmem_transport_region_role_t role, uint64_t issuer_id, uint64_t region_id,
              Submit submit) {
        if (!nvshmemt_region_role_is_valid(role)) {
            return NVSHMEMX_ERROR_INVALID_VALUE;
        }
        nvshmemi_region_key key = {issuer_id, region_id};
        return regions_.flush(role, key, submit);
    }
};

static inline std::unique_ptr<nvshmemt_batch_rma_state> nvshmemt_make_batch_rma_state(
    size_t capacity, size_t max_batch_ops) noexcept {
    try {
        return std::make_unique<nvshmemt_batch_rma_state>(capacity, max_batch_ops);
    } catch (...) {
        return nullptr;
    }
}

#endif /* _NVSHMEMT_TRANSPORT_BATCH_RMA_HPP_ */
