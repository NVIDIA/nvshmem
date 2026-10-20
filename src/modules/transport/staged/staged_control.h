/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_STAGED_CONTROL_H
#define NVSHMEM_STAGED_CONTROL_H

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace nvshmemi {

/* Fixed storage shared by submitters and one FIFO consumer. A blocking submission owns its
 * completion until wait() consumes it. Cancellation may release that ownership, but a running
 * operation's storage remains alive until the consumer calls complete(). */
template <typename Operation, std::size_t Capacity>
class staged_operation_queue_t {
    static_assert(Capacity > 0, "The operation queue needs at least one slot");
    static_assert(std::is_nothrow_move_constructible<Operation>::value,
                  "Moving an operation must not throw");
    static_assert(std::is_nothrow_destructible<Operation>::value,
                  "Releasing an operation must not throw");

   public:
    struct ticket {
        std::size_t index = Capacity;
        std::uint64_t generation = 0;
    };

    struct work_item {
        Operation* operation = nullptr;
        ticket token;
        std::uint64_t sequence = 0;
    };

    explicit staged_operation_queue_t(int cancellation_status)
        : cancellation_status_(cancellation_status != 0 ? cancellation_status : -1) {
        for (std::size_t i = 0; i < Capacity; ++i) {
            free_[i] = i;
        }
    }

    /* Both kinds of submission apply backpressure when full. Only blocking submissions retain
     * a completion for wait(); nonblocking slots return to the free ring on completion. */
    int submit(Operation operation, bool blocking, ticket* result) {
        if (!result) {
            return cancellation_status_;
        }
        *result = {};
        std::unique_lock<std::mutex> lock(mutex_);
        capacity_cv_.wait(lock, [&] { return stopped_ || free_count_ != 0; });
        if (stopped_) {
            return status_locked();
        }

        const std::size_t index = free_[free_head_];
        slot& entry = slots_[index];
        if (submitted_ == std::numeric_limits<std::uint64_t>::max() ||
            entry.generation == std::numeric_limits<std::uint64_t>::max()) {
            fail_locked(cancellation_status_);
            return status_locked();
        }

        free_head_ = (free_head_ + 1) % Capacity;
        --free_count_;
        ++entry.generation;
        entry.sequence = ++submitted_;
        entry.operation.emplace(std::move(operation));
        entry.state = slot_state::QUEUED;
        entry.waiter = blocking;
        entry.status = 0;
        ready_[(ready_head_ + ready_count_) % Capacity] = index;
        ++ready_count_;
        *result = {index, entry.generation};
        ready_cv_.notify_one();
        return 0;
    }

    bool take(work_item* result) {
        if (!result) {
            return false;
        }
        *result = {};
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cv_.wait(lock,
                       [&] { return stopped_ || (ready_count_ != 0 && running_ == Capacity); });
        if (stopped_) {
            return false;
        }
        const std::size_t index = ready_[ready_head_];
        ready_head_ = (ready_head_ + 1) % Capacity;
        --ready_count_;
        slot& entry = slots_[index];
        entry.state = slot_state::RUNNING;
        running_ = index;
        *result = {&*entry.operation, {index, entry.generation}, entry.sequence};
        return true;
    }

    int complete(ticket token, int status) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!matches_locked(token) || slots_[token.index].state != slot_state::RUNNING) {
            return cancellation_status_;
        }

        if (status != 0) {
            fail_locked(status);
        }
        slot& entry = slots_[token.index];
        entry.status = stopped_ ? status_locked() : status;
        entry.operation.reset();
        entry.state = slot_state::DONE;
        running_ = Capacity;
        /* Cancelled queued work must not move the completion frontier past a running operation.
         * Once that operation finishes, every accepted operation is either finished or cancelled.
         */
        completed_ = stopped_ ? submitted_ : entry.sequence;
        if (!entry.waiter) {
            release_locked(token.index);
        }
        done_cv_.notify_all();
        ready_cv_.notify_one();
        return 0;
    }

    /* Each blocking ticket has one owner and must be waited exactly once. */
    int wait(ticket token) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!matches_locked(token) || !slots_[token.index].waiter) {
            return cancellation_status_;
        }
        done_cv_.wait(lock, [&] {
            return !matches_locked(token) || slots_[token.index].state == slot_state::DONE;
        });
        if (!matches_locked(token) || !slots_[token.index].waiter) {
            return cancellation_status_;
        }
        slot& entry = slots_[token.index];
        const int result = entry.state == slot_state::DONE && entry.status == 0
                               ? 0
                               : (stopped_ ? status_locked() : entry.status);
        entry.waiter = false;
        if (entry.state == slot_state::DONE) {
            release_locked(token.index);
        }
        return result;
    }

    int quiet() {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::uint64_t target = submitted_;
        done_cv_.wait(lock, [&] { return completed_ >= target; });
        return status_locked();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cancel_queued_locked();
    }

    void fail(int status) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_locked(status);
    }

    int status() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return status_locked();
    }

   private:
    enum class slot_state { FREE, QUEUED, RUNNING, DONE };

    struct slot {
        std::optional<Operation> operation;
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
        slot_state state = slot_state::FREE;
        bool waiter = false;
        int status = 0;
    };

    bool matches_locked(ticket token) const {
        return token.index < Capacity && slots_[token.index].state != slot_state::FREE &&
               slots_[token.index].generation == token.generation;
    }

    int status_locked() const {
        return failure_status_ != 0 ? failure_status_ : (stopped_ ? cancellation_status_ : 0);
    }

    void release_locked(std::size_t index) {
        slot& entry = slots_[index];
        entry.operation.reset();
        entry.state = slot_state::FREE;
        entry.waiter = false;
        free_[(free_head_ + free_count_) % Capacity] = index;
        ++free_count_;
        capacity_cv_.notify_one();
    }

    void cancel_queued_locked() {
        while (ready_count_ != 0) {
            const std::size_t index = ready_[ready_head_];
            ready_head_ = (ready_head_ + 1) % Capacity;
            --ready_count_;
            slot& entry = slots_[index];
            entry.operation.reset();
            entry.status = status_locked();
            entry.state = slot_state::DONE;
            if (!entry.waiter) {
                release_locked(index);
            }
        }
        if (running_ == Capacity) {
            completed_ = submitted_;
        }
        ready_cv_.notify_all();
        done_cv_.notify_all();
        capacity_cv_.notify_all();
    }

    void fail_locked(int status) {
        if (status != 0 && failure_status_ == 0) {
            failure_status_ = status;
        }
        stopped_ = true;
        cancel_queued_locked();
    }

    const int cancellation_status_;
    mutable std::mutex mutex_;
    std::condition_variable capacity_cv_;
    std::condition_variable ready_cv_;
    std::condition_variable done_cv_;
    std::array<slot, Capacity> slots_{};
    std::array<std::size_t, Capacity> free_{};
    std::array<std::size_t, Capacity> ready_{};
    std::size_t free_head_ = 0;
    std::size_t free_count_ = Capacity;
    std::size_t ready_head_ = 0;
    std::size_t ready_count_ = 0;
    std::size_t running_ = Capacity;
    std::uint64_t submitted_ = 0;
    std::uint64_t completed_ = 0;
    bool stopped_ = false;
    int failure_status_ = 0;
};

/* Responses are reserved before a request is sent, so the receive path never allocates. IDs are
 * never reused; a delayed reply cannot satisfy a new reservation that reuses the same slot. */
template <typename Response, std::size_t Capacity>
class staged_response_table_t {
    static_assert(Capacity > 0, "The response table needs at least one slot");
    static_assert(std::is_nothrow_copy_constructible<Response>::value &&
                      std::is_nothrow_copy_assignable<Response>::value,
                  "Recording and consuming a response must not throw");

   public:
    struct ticket {
        std::size_t index = Capacity;
        std::uint64_t request_id = 0;
    };

    explicit staged_response_table_t(int cancellation_status)
        : cancellation_status_(cancellation_status != 0 ? cancellation_status : -1) {}

    /* A full table is a caller sizing/protocol error, not a wait: the sole requesting worker
     * must consume its pending replies to make room. The last representable ID is usable. */
    int reserve(ticket* result) {
        if (!result) {
            return cancellation_status_;
        }
        *result = {};
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return status_locked();
        }
        if (last_request_id_ == std::numeric_limits<std::uint64_t>::max()) {
            return cancellation_status_;
        }
        for (std::size_t i = 0; i < Capacity; ++i) {
            slot& entry = slots_[i];
            if (entry.request_id != 0) {
                continue;
            }
            entry.request_id = ++last_request_id_;
            *result = {i, entry.request_id};
            return 0;
        }
        return cancellation_status_;
    }

    int complete(std::uint64_t request_id, const Response& response) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return status_locked();
        }
        if (request_id == 0) {
            return cancellation_status_;
        }
        for (slot& entry : slots_) {
            if (entry.request_id != request_id) {
                continue;
            }
            if (entry.response) {
                return cancellation_status_;
            }
            entry.response.emplace(response);
            cv_.notify_all();
            return 0;
        }
        return cancellation_status_;
    }

    /* Return the table status; the caller interprets any status carried inside Response. A
     * response already received remains consumable even when shutdown begins concurrently. */
    int wait(ticket token, Response* result) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!matches_locked(token)) {
            return cancellation_status_;
        }
        cv_.wait(lock, [&] {
            return stopped_ || !matches_locked(token) || slots_[token.index].response.has_value();
        });
        if (!matches_locked(token)) {
            return cancellation_status_;
        }
        slot& entry = slots_[token.index];
        const int status = entry.response ? 0 : status_locked();
        if (result && entry.response) {
            *result = *entry.response;
        }
        entry.response.reset();
        entry.request_id = 0;
        return status;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    void fail(int status) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status != 0 && failure_status_ == 0) {
            failure_status_ = status;
        }
        stopped_ = true;
        cv_.notify_all();
    }

    int status() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return status_locked();
    }

   private:
    struct slot {
        std::uint64_t request_id = 0;
        std::optional<Response> response;
    };

    bool matches_locked(ticket token) const {
        return token.index < Capacity && token.request_id != 0 &&
               slots_[token.index].request_id == token.request_id;
    }

    int status_locked() const {
        return failure_status_ != 0 ? failure_status_ : (stopped_ ? cancellation_status_ : 0);
    }

    const int cancellation_status_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::array<slot, Capacity> slots_{};
    std::uint64_t last_request_id_ = 0;
    bool stopped_ = false;
    int failure_status_ = 0;
};

/* Construct one latch per worker group and declare it before the owned worker threads. */
class staged_startup_latch_t {
   public:
    explicit staged_startup_latch_t(unsigned int expected) : expected_(expected) {}

    void report(int status) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (reports_ == expected_) {
            return;
        }
        if (status != 0 && status_ == 0) {
            status_ = status;
        }
        ++reports_;
        cv_.notify_all();
    }

    int wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return status_ != 0 || reports_ == expected_; });
        return status_;
    }

   private:
    const unsigned int expected_;
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned int reports_ = 0;
    int status_ = 0;
};

}  // namespace nvshmemi

#endif /* NVSHMEM_STAGED_CONTROL_H */
