/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef NVSHMEMI_SCOPE_GUARD_H
#define NVSHMEMI_SCOPE_GUARD_H

#include <type_traits>
#include <utility>

template <typename F>
class scope_guard {
   public:
    scope_guard(const scope_guard&) = delete;
    scope_guard& operator=(const scope_guard&) = delete;

    explicit scope_guard(F&& f) noexcept(std::is_nothrow_move_constructible<F>::value)
        : func_(std::move(f)), active_(true) {}

    explicit scope_guard(const F& f) noexcept(std::is_nothrow_copy_constructible<F>::value)
        : func_(f), active_(true) {}

    scope_guard(scope_guard&& other) noexcept(std::is_nothrow_move_constructible<F>::value)
        : func_(std::move(other.func_)), active_(other.active_) {
        other.dismiss();
    }

    ~scope_guard() noexcept {
        if (active_) {
            func_();
        }
    }

    void dismiss() noexcept { active_ = false; }

   private:
    F func_;
    bool active_;
};

template <typename F>
auto make_scope_guard(F&& f) -> scope_guard<typename std::decay<F>::type> {
    return scope_guard<typename std::decay<F>::type>(std::forward<F>(f));
}

#endif  // NVSHMEMI_SCOPE_GUARD_H
