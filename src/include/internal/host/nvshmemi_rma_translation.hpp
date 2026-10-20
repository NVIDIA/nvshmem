/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_RMA_TRANSLATION_HPP
#define NVSHMEMI_RMA_TRANSLATION_HPP

#include <cstddef>
#include <vector>

#include "internal/host/nvshmemi_types.h"

class nvshmemi_transport_pe {
   public:
    constexpr explicit nvshmemi_transport_pe(int value) noexcept : value_(value) {}

    [[nodiscard]] constexpr int value() const noexcept { return value_; }
    friend constexpr bool operator==(nvshmemi_transport_pe lhs,
                                     nvshmemi_transport_pe rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }

   private:
    int value_;
};

static inline void *nvshmemi_translate_mapped_ptr(void *symmetric_ptr, int target_pe,
                                                  const std::vector<void *> &local_pe_bases) {
    auto *remote_base = static_cast<char *>(local_pe_bases[target_pe]);
    auto *heap_base = static_cast<char *>(nvshmemi_device_state.heap_base);

    return remote_base + (static_cast<char *>(symmetric_ptr) - heap_base);
}

static inline nvshmemi_transport_pe nvshmemi_get_transport_pe(int target_pe) {
    if (!nvshmemi_device_state.enable_rail_opt) {
        return nvshmemi_transport_pe{target_pe};
    }

    return nvshmemi_transport_pe{(target_pe / nvshmemi_state->npes_node) *
                                     nvshmemi_state->npes_node +
                                 nvshmemi_state->mype_node};
}

struct nvshmemi_remote_rma_target {
    void *remote_ptr;
    nvshmemi_transport_pe transport_pe;
};

static inline nvshmemi_remote_rma_target nvshmemi_translate_unmapped_ptr(
    void *symmetric_ptr, int target_pe, const std::vector<void *> &remote_pe_bases) {
    const auto transport_pe = nvshmemi_get_transport_pe(target_pe);
    std::ptrdiff_t rail_offset = 0;

    if (nvshmemi_device_state.enable_rail_opt) {
        rail_offset = static_cast<std::ptrdiff_t>(target_pe % nvshmemi_state->npes_node -
                                                  nvshmemi_state->mype_node) *
                      static_cast<std::ptrdiff_t>(nvshmemi_device_state.heap_size);
    }

    auto *remote_base = static_cast<char *>(remote_pe_bases[transport_pe.value()]);
    auto *heap_base = static_cast<char *>(nvshmemi_device_state.heap_base);
    auto *remote_ptr = remote_base + rail_offset + (static_cast<char *>(symmetric_ptr) - heap_base);

    return {remote_ptr, transport_pe};
}

#endif
