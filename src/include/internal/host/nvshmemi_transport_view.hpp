/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_TRANSPORT_VIEW_HPP
#define NVSHMEMI_TRANSPORT_VIEW_HPP

#include "internal/host_transport/transport.h"

class nvshmemi_transport_view {
   public:
    nvshmemi_transport_view(int num_transports, struct nvshmem_transport **transports,
                            int transport_bitmap, int *transport_map) noexcept
        : num_transports_(num_transports),
          transports_(transports),
          transport_bitmap_(transport_bitmap),
          transport_map_(transport_map) {}

    int num_transports() const { return num_transports_; }
    struct nvshmem_transport *transport(int idx) const { return transports_[idx]; }

    bool is_active(int idx) const { return (transport_bitmap_ & (1 << idx)) != 0; }
    bool has_cap(int idx, int pe, int flag) const {
        return (transports_[idx]->cap[pe] & flag) != 0;
    }
    bool active_has_cap(int idx, int pe, int flag) const {
        return is_active(idx) && has_cap(idx, pe, flag);
    }
    bool is_active_between(int src_pe, int dst_pe, int npes, int idx) const {
        return (transport_map_[src_pe * npes + dst_pe] & (1 << idx)) != 0;
    }
    bool active_between_has_cap(int src_pe, int dst_pe, int npes, int idx, int flag) const {
        return is_active_between(src_pe, dst_pe, npes, idx) && has_cap(idx, dst_pe, flag);
    }
    bool supports_get_mem(int idx) const {
        return transports_[idx]->host_ops.get_mem_handle != nullptr;
    }
    bool supports_release_mem(int idx) const {
        return transports_[idx]->host_ops.release_mem_handle != nullptr;
    }
    bool supports_add_device_remote_mem(int idx) const {
        return transports_[idx]->host_ops.add_device_remote_mem_handles != nullptr;
    }
    void clear_cap(int idx, int pe, int flags) { transports_[idx]->cap[pe] &= ~flags; }

   private:
    int num_transports_;
    struct nvshmem_transport **transports_;
    int transport_bitmap_;
    int *transport_map_;
};

#endif /* NVSHMEMI_TRANSPORT_VIEW_HPP */
