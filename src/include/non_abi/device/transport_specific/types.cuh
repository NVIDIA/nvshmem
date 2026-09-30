/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NVSHMEMI_TRANSPORT_SPECIFIC_TYPES_CUH_
#define _NVSHMEMI_TRANSPORT_SPECIFIC_TYPES_CUH_

namespace nvshmemx {

enum class threadgroup {
    thread,
    warp,
    block,
};

namespace addressing {
struct Pointer {};
struct CFT {};
struct Auto {};
}  // namespace addressing

namespace transport_policy {
struct Auto {};
struct SM {};
struct TMA {};
struct CE {};
struct Network {};
}  // namespace transport_policy

namespace memory_space {
enum class source {
    shared,
    global,
    generic,
};

enum class dest {
    shared,
    global,
    generic,
};
}  // namespace memory_space

}  // namespace nvshmemx

#endif  // _NVSHMEMI_TRANSPORT_SPECIFIC_TYPES_CUH_
