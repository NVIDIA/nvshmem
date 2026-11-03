/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_DEVICE_STATE_COMPAT_H
#define NVSHMEMI_DEVICE_STATE_COMPAT_H

#include "c_api/nvshmem_version.h"
#include "non_abi/device/common/nvshmemi_runtime_types.h"

/* Released layouts share a prefix and remain fixed within each minor version.
 * Early ABI releases report inter-library version 3.0.0.
 * The host's cudaGetSymbolSize query cannot size callback-owned symbols.
 */
static inline size_t nvshmemi_device_state_size_for_version(nvshmemi_version_t version) {
    static constexpr size_t sizes[] = {
        776, /* 3.0 */
        776, /* 3.1 */
        776, /* 3.2 */
        776, /* 3.3 */
        776, /* 3.4 */
        776, /* 3.5 */
        776, /* 3.6 */
        816, /* 3.7 */
        848, /* 3.8 */
        864, /* 3.9 */
    };
    static_assert(NVSHMEM_VENDOR_MAJOR_VERSION == 3 &&
                      NVSHMEM_VENDOR_MINOR_VERSION == sizeof(sizes) / sizeof(sizes[0]) - 1,
                  "Update the device state size table for this release.");
    static_assert(sizeof(nvshmemi_device_host_state_t) == sizes[NVSHMEM_VENDOR_MINOR_VERSION],
                  "Update the current release's device state size table entry.");

    if (version.major != 3 || version.minor < 0 || version.patch < 0 ||
        static_cast<size_t>(version.minor) >= sizeof(sizes) / sizeof(sizes[0])) {
        return 0;
    }
    return sizes[version.minor];
}

#endif /* NVSHMEMI_DEVICE_STATE_COMPAT_H */
