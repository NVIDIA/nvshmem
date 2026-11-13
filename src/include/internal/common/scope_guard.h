/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef NVSHMEMI_SCOPE_GUARD_H
#define NVSHMEMI_SCOPE_GUARD_H

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#define NVSHMEMI_SCOPE_GUARD_HOST_DEVICE __host__ __device__
namespace nvshmemi_scope_guard_std = cuda::std;
#else
#include <type_traits>
#include <utility>
#define NVSHMEMI_SCOPE_GUARD_HOST_DEVICE
namespace nvshmemi_scope_guard_std = std;
#endif

template <typename F>
class scope_guard {
   public:
    scope_guard(const scope_guard&) = delete;
    scope_guard& operator=(const scope_guard&) = delete;

    NVSHMEMI_SCOPE_GUARD_HOST_DEVICE explicit scope_guard(F&& f) noexcept(
        nvshmemi_scope_guard_std::is_nothrow_move_constructible<F>::value)
        : func_(nvshmemi_scope_guard_std::move(f)), active_(true) {}

    NVSHMEMI_SCOPE_GUARD_HOST_DEVICE explicit scope_guard(const F& f) noexcept(
        nvshmemi_scope_guard_std::is_nothrow_copy_constructible<F>::value)
        : func_(f), active_(true) {}

    NVSHMEMI_SCOPE_GUARD_HOST_DEVICE scope_guard(scope_guard&& other) noexcept(
        nvshmemi_scope_guard_std::is_nothrow_move_constructible<F>::value)
        : func_(nvshmemi_scope_guard_std::move(other.func_)), active_(other.active_) {
        other.dismiss();
    }

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
#pragma nv_exec_check_disable
#endif
    NVSHMEMI_SCOPE_GUARD_HOST_DEVICE ~scope_guard() noexcept {
        if (active_) {
            func_();
        }
    }

    NVSHMEMI_SCOPE_GUARD_HOST_DEVICE void dismiss() noexcept { active_ = false; }

   private:
    F func_;
    bool active_;
};

template <typename F>
NVSHMEMI_SCOPE_GUARD_HOST_DEVICE auto make_scope_guard(F&& f)
    -> scope_guard<typename nvshmemi_scope_guard_std::decay<F>::type> {
    return scope_guard<typename nvshmemi_scope_guard_std::decay<F>::type>(
        nvshmemi_scope_guard_std::forward<F>(f));
}

#undef NVSHMEMI_SCOPE_GUARD_HOST_DEVICE

#endif  // NVSHMEMI_SCOPE_GUARD_H
