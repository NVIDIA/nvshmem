/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_HOST_TRANSPORT_REGION_HPP_
#define _NVSHMEMI_HOST_TRANSPORT_REGION_HPP_

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <vector>

struct nvshmemi_region_key {
    uint64_t issuer_id;
    uint64_t region_id;

    bool operator==(const nvshmemi_region_key &other) const {
        return issuer_id == other.issuer_id && region_id == other.region_id;
    }
};

static constexpr bool nvshmemi_region_table_capacity_is_valid(size_t capacity) {
    return capacity > 0 && (capacity & (capacity - 1)) == 0;
}

/* Fixed-capacity hash map with bounded linear probing. Lookups scan the full probe window, so an
 * erased entry can simply be marked inactive without terminating the search. */
template <typename RegionState>
class nvshmemi_region_table {
    // Keep probe metadata separate from transport-owned state, which is only needed on a match.
    std::vector<nvshmemi_region_key> keys_;
    std::vector<RegionState> regions_;
    std::vector<uint8_t> active_;
    size_t probe_limit_;

    size_t start(const nvshmemi_region_key &key) const {
        size_t issuer_hash = std::hash<uint64_t>{}(key.issuer_id);
        size_t region_hash = std::hash<uint64_t>{}(key.region_id);
        return (issuer_hash ^ (region_hash << 1)) & (keys_.size() - 1);
    }

   public:
    nvshmemi_region_table(size_t capacity, size_t probe_limit)
        : keys_(capacity),
          regions_(capacity),
          active_(capacity),
          probe_limit_(probe_limit < capacity ? probe_limit : capacity) {
        assert(nvshmemi_region_table_capacity_is_valid(capacity));
    }

    RegionState *find(const nvshmemi_region_key &key) {
        size_t slot_index = start(key);
        for (size_t i = 0; i < probe_limit_; i++) {
            size_t candidate = (slot_index + i) & (keys_.size() - 1);
            if (active_[candidate] && keys_[candidate] == key) {
                return &regions_[candidate];
            }
        }
        return nullptr;
    }

    RegionState *find_or_create(const nvshmemi_region_key &key) {
        size_t slot_index = start(key);
        size_t free_slot = keys_.size();
        for (size_t i = 0; i < probe_limit_; i++) {
            size_t candidate = (slot_index + i) & (keys_.size() - 1);
            if (active_[candidate] && keys_[candidate] == key) {
                return &regions_[candidate];
            }
            if (!active_[candidate] && free_slot == keys_.size()) {
                free_slot = candidate;
            }
        }
        if (free_slot == keys_.size()) {
            return nullptr;
        }

        keys_[free_slot] = key;
        regions_[free_slot] = RegionState{};
        active_[free_slot] = true;
        return &regions_[free_slot];
    }

    void erase(const nvshmemi_region_key &key) {
        // Lookups scan the full probe window, so inactive slots do not need tombstones.
        size_t slot_index = start(key);
        for (size_t i = 0; i < probe_limit_; i++) {
            size_t candidate = (slot_index + i) & (keys_.size() - 1);
            if (active_[candidate] && keys_[candidate] == key) {
                regions_[candidate] = RegionState{};
                active_[candidate] = false;
                return;
            }
        }
    }
};

template <typename RegionState, size_t TABLE_COUNT>
class nvshmemi_region_lifecycle {
    std::array<std::unique_ptr<nvshmemi_region_table<RegionState>>, TABLE_COUNT> tables_;

   public:
    nvshmemi_region_lifecycle(const std::array<size_t, TABLE_COUNT> &capacities,
                              size_t probe_limit) {
        for (size_t i = 0; i < TABLE_COUNT; i++) {
            tables_[i] =
                std::make_unique<nvshmemi_region_table<RegionState>>(capacities[i], probe_limit);
        }
    }

    RegionState *find_or_create(size_t table_index, const nvshmemi_region_key &key) {
        assert(table_index < TABLE_COUNT);
        return tables_[table_index]->find_or_create(key);
    }

    template <typename Submit>
    int flush(size_t table_index, const nvshmemi_region_key &key, Submit submit) {
        assert(table_index < TABLE_COUNT);
        RegionState *region = tables_[table_index]->find(key);
        if (region == nullptr) {
            return 0;
        }

        int status = submit(*region);
        if (status) {
            return status;
        }
        tables_[table_index]->erase(key);
        return 0;
    }
};

#endif
