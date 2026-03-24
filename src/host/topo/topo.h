/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __TOPO_H
#define __TOPO_H
#include "internal/host/nvshmemi_types.h"  // for nvshmemi_state_t

int nvshmemi_get_devices_by_distance(int *device_arr, int max_dev_per_pe,
                                     struct nvshmem_transport *tcurr);
int nvshmemi_get_netdevs_policy_entity_count(nvshmemi_state_t *state);
int nvshmemi_detect_same_device(nvshmemi_state_t *state);
int nvshmemi_build_transport_map(nvshmemi_state_t *state);
int nvshmemi_set_cpu_affinity(nvshmemi_state_t *state);

#endif
