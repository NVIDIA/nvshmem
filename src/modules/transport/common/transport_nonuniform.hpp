/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEM_TRANSPORT_NONUNIFORM_HPP
#define NVSHMEM_TRANSPORT_NONUNIFORM_HPP

#include <algorithm>
#include <vector>
#include "internal/host_transport/transport.h"

/* Unequal local PE counts can select different numbers of NICs. Per-device
 * endpoint exchanges must use the same count on every PE. Leave uniform jobs'
 * device selection unchanged. */
static inline int nvshmemt_limit_nonuniform_devices(bootstrap_handle_t *bootstrap,
                                                    std::vector<int> &devices,
                                                    std::vector<int> &ports, bool &nonuniform) {
    std::vector<int> unique_devices, unique_ports;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (std::find(unique_devices.begin(), unique_devices.end(), devices[i]) ==
            unique_devices.end()) {
            unique_devices.push_back(devices[i]);
            unique_ports.push_back(ports[i]);
        }
    }
    int local[] = {bootstrap->npes_node, static_cast<int>(unique_devices.size())};
    std::vector<int> peer_counts(2 * bootstrap->pg_size);
    int status = bootstrap->allgather(local, peer_counts.data(), sizeof(local), bootstrap);
    if (status) {
        return status;
    }

    nonuniform = false;
    int common_count = local[1];
    for (int pe = 0; pe < bootstrap->pg_size; ++pe) {
        nonuniform |= peer_counts[2 * pe] != local[0];
        common_count = std::min(common_count, peer_counts[2 * pe + 1]);
    }
    if (!nonuniform) {
        return 0;
    }
    if (common_count <= 0) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    unique_devices.resize(common_count);
    unique_ports.resize(common_count);
    devices.swap(unique_devices);
    ports.swap(unique_ports);
    return 0;
}

#endif
