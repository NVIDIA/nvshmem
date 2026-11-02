/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEMI_PE_TOPOLOGY_HPP
#define NVSHMEMI_PE_TOPOLOGY_HPP

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

/* Node IDs follow first occurrence in world-PE order. Local ranks follow world-PE order
 * within each node, independently of the number of PEs on other nodes. */
struct nvshmemi_pe_topology_t {
    std::vector<int> node_of_pe;
    std::vector<int> local_rank;
    std::vector<int> node_sizes;
    int max_pes_per_node = 0;

    void initialize(const uint64_t *host_hashes, int npes) {
        node_of_pe.resize(npes);
        local_rank.resize(npes);
        node_sizes.clear();
        max_pes_per_node = 0;
        std::unordered_map<uint64_t, int> nodes;
        for (int pe = 0; pe < npes; ++pe) {
            auto entry = nodes.emplace(host_hashes[pe], static_cast<int>(node_sizes.size()));
            if (entry.second) {
                node_sizes.push_back(0);
            }
            int node = entry.first->second;
            node_of_pe[pe] = node;
            local_rank[pe] = node_sizes[node]++;
            max_pes_per_node = std::max(max_pes_per_node, node_sizes[node]);
        }
    }

    bool is_uniform() const {
        return !node_sizes.empty() &&
               std::all_of(node_sizes.begin(), node_sizes.end(),
                           [this](int size) { return size == max_pes_per_node; });
    }

    bool is_block_distribution() const {
        if (!is_uniform()) {
            return false;
        }
        for (size_t pe = 0; pe < node_of_pe.size(); ++pe) {
            if (node_of_pe[pe] != static_cast<int>(pe / max_pes_per_node)) {
                return false;
            }
        }
        return true;
    }

    // Require uniform PEs per node, node-major ordering, and matching local ranks.
    bool has_block_hierarchy(const int *pes, int count) const {
        if (count <= 0) {
            return false;
        }
        // Infer the child team's group size.
        int first_node = node_of_pe[pes[0]];
        int pes_per_node = 0;
        while (pes_per_node < count && node_of_pe[pes[pes_per_node]] == first_node) {
            ++pes_per_node;
        }
        // Requires the same PE count on every node.
        if (count % pes_per_node != 0) {
            return false;
        }
        // Each node must contribute the same local ranks in the same order.
        for (int row = pes_per_node; row < count; row += pes_per_node) {
            int node = node_of_pe[pes[row]];
            for (int column = 0; column < pes_per_node; ++column) {
                int pe = pes[row + column];
                if (node_of_pe[pe] != node || local_rank[pe] != local_rank[pes[column]]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool is_roundrobin_distribution() const {
        if (!is_uniform()) {
            return false;
        }
        for (size_t pe = 0; pe < node_of_pe.size(); ++pe) {
            if (node_of_pe[pe] != static_cast<int>(pe % node_sizes.size())) {
                return false;
            }
        }
        return true;
    }
};

extern nvshmemi_pe_topology_t nvshmemi_pe_topology;

#endif
