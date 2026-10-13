/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMT_IBGDA_QCOUNTER_HPP
#define NVSHMEMT_IBGDA_QCOUNTER_HPP

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace nvshmemt_ibgda_qcounter {

// mlx5 queue-counter command layout, also described by Linux v6.16 mlx5_ifc.h.
// Only an object created by these callbacks can be queried. Commands never
// clear counters, aggregate another counter set, or address another vport.
struct operations {
    void *(*create)(void *, const void *, size_t, void *, size_t);
    int (*query)(void *, const void *, size_t, void *, size_t);
    int (*destroy)(void *);
};

struct capabilities {
    unsigned max_qp_cnt;
    bool out_of_sequence;
    bool retransmission;
};

enum class state {
    disabled,
    capability_unavailable,
    allocation_failed,
    reserved_zero_id,
    ready,
    destroyed,
    destroy_failed,
};

inline const char *state_name(state value) {
    switch (value) {
        case state::disabled: return "DISABLED";
        case state::capability_unavailable: return "CAPABILITY_UNAVAILABLE";
        case state::allocation_failed: return "ALLOCATION_FAILED";
        case state::reserved_zero_id: return "ZERO_ID_NOT_ASSOCIATED";
        case state::ready: return "OWNED_COUNTER_ALLOCATED";
        case state::destroyed: return "OWNED_COUNTER_DESTROYED";
        case state::destroy_failed: return "COUNTER_DESTROY_FAILED";
    }
    return "UNKNOWN";
}

inline uint32_t read_be32(const unsigned char *p) {
    return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
}

struct snapshot {
    bool valid = false;
    const char *status = "NOT_ATTACHED";
    int error = 0;
    unsigned firmware_status = 0;
    uint32_t syndrome = 0;
    uint64_t sampled_ns = 0;
    std::array<uint32_t, 11> values{};
    std::array<bool, 11> supported{};
};

// One process/device counter shared only by its verified owned RC QPs. The
// counter must outlive every accepted attachment, including an attachment
// whose QPC readback failed. DCI/DCT traffic is outside this counter's scope.
class resource {
 public:
    resource(bool enabled, capabilities caps, void *context, operations callbacks)
        : caps_(caps), ops_(callbacks) {
        if (!enabled) return;
        if (!caps.max_qp_cnt) {
            state_ = state::capability_unavailable;
            return;
        }
        std::array<unsigned char, 16> in{}, out{};
        in[0] = 0x07;
        in[1] = 0x71;  // ALLOC_Q_COUNTER; UID supplied by the owning DEVX context.
        errno = 0;
        handle_ = ops_.create(context, in.data(), in.size(), out.data(), out.size());
        error_ = handle_ ? 0 : errno;
        firmware_status_ = out[0];
        syndrome_ = read_be32(out.data() + 4);
        if (!handle_) {
            state_ = state::allocation_failed;
            return;
        }
        id_ = out[11];
        // A zero QPC counter ID is also the unassociated baseline. Do not
        // interpret it as attribution to this resource, even if returned.
        state_ = firmware_status_ ? state::allocation_failed
                                 : (id_ ? state::ready : state::reserved_zero_id);
    }

    resource(const resource &) = delete;
    resource &operator=(const resource &) = delete;
    // Cleanup is explicit: implicit destruction cannot precede attached QPs.
    ~resource() = default;

    state availability() const { return state_; }
    unsigned id() const { return state_ == state::ready ? id_ : 0; }
    int error() const { return error_; }
    unsigned firmware_status() const { return firmware_status_; }
    uint32_t syndrome() const { return syndrome_; }
    capabilities caps() const { return caps_; }

    void attached(bool verified) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++attached_;
        verified_ += verified;
    }
    bool qp_destroyed(bool verified) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!attached_ || (verified && !verified_)) return false;
        --attached_;
        verified_ -= verified;
        return true;
    }
    unsigned attached_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return attached_;
    }
    unsigned verified_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return verified_;
    }

    snapshot query(uint64_t now_ns, uint64_t interval_ns = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot result;
        if (state_ != state::ready || !handle_) {
            result.status = state_name(state_);
            return result;
        }
        if (!verified_) return result;
        if (interval_ns && sampled_ && now_ns - last_sample_ns_ < interval_ns) {
            result.status = "RATE_LIMITED";
            return result;
        }
        sampled_ = true;
        last_sample_ns_ = now_ns;
        result.sampled_ns = now_ns;
        std::array<unsigned char, 32> in{};
        std::array<unsigned char, 256> out{};
        in[0] = 0x07;
        in[1] = 0x73;  // QUERY_Q_COUNTER
        in[31] = id_;
        errno = 0;
        int rc = ops_.query(handle_, in.data(), in.size(), out.data(), out.size());
        result.error = rc ? errno : 0;
        result.firmware_status = out[0];
        result.syndrome = read_be32(out.data() + 4);
        if (rc || result.firmware_status) {
            result.status = "COUNTER_QUERY_UNAVAILABLE";
            return result;
        }
        result.valid = true;
        result.status = "OWNED_RC_DEVICE_COUNTER_QUERY_AVAILABLE";
        for (size_t i = 0; i < result.values.size(); ++i) {
            result.supported[i] = i < 5 || (i == 5 ? caps_.out_of_sequence : caps_.retransmission);
            if (result.supported[i]) result.values[i] = read_be32(out.data() + 16 + 8 * i);
        }
        return result;
    }

    // A failed destroy is recorded once. The provider's object lifetime after
    // failure is not assumed, and the same opaque handle is never retried.
    bool close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (attached_) return false;
        if (destroy_attempted_) return state_ != state::destroy_failed;
        if (!handle_) return true;
        destroy_attempted_ = true;
        void *handle = handle_;
        handle_ = nullptr;
        errno = 0;
        int rc = ops_.destroy(handle);
        error_ = rc ? errno : 0;
        state_ = rc ? state::destroy_failed : state::destroyed;
        return rc == 0;
    }

 private:
    capabilities caps_;
    operations ops_;
    std::mutex mutex_;
    void *handle_ = nullptr;
    state state_ = state::disabled;
    unsigned id_ = 0, attached_ = 0, verified_ = 0;
    int error_ = 0;
    unsigned firmware_status_ = 0;
    uint32_t syndrome_ = 0;
    uint64_t last_sample_ns_ = 0;
    bool sampled_ = false, destroy_attempted_ = false;
};
}  // namespace nvshmemt_ibgda_qcounter
#endif
