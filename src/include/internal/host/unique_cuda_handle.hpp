/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_UNIQUE_CUDA_HANDLE_HPP
#define NVSHMEMI_UNIQUE_CUDA_HANDLE_HPP

#include <cassert>
#include <optional>
#include <utility>
#include "internal/host_transport/cudawrap.h"
#include "non_abi/c/nvshmemi_error_macros.h"

extern nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;

namespace nvshmemi {

class UniqueCudaAllocationHandle {
   public:
    UniqueCudaAllocationHandle() noexcept = default;
    explicit UniqueCudaAllocationHandle(CUmemGenericAllocationHandle handle) noexcept
        : handle_{handle} {}
    ~UniqueCudaAllocationHandle() {
        if (reset() != CUDA_SUCCESS) {
            NVSHMEMI_WARN_PRINT("cuMemRelease failed while releasing an owned allocation handle");
        }
    }

    UniqueCudaAllocationHandle(const UniqueCudaAllocationHandle &) = delete;
    UniqueCudaAllocationHandle &operator=(const UniqueCudaAllocationHandle &) = delete;

    UniqueCudaAllocationHandle(UniqueCudaAllocationHandle &&other) noexcept
        : handle_{std::exchange(other.handle_, std::nullopt)} {}

    UniqueCudaAllocationHandle &operator=(UniqueCudaAllocationHandle &&other) noexcept {
        if (this != &other) {
            UniqueCudaAllocationHandle moved{std::move(other)};
            handle_.swap(moved.handle_);
        }
        return *this;
    }

    CUmemGenericAllocationHandle get() const noexcept {
        assert(handle_.has_value());
        return *handle_;
    }

    CUresult reset() noexcept {
        if (!handle_) {
            return CUDA_SUCCESS;
        }
        const CUresult status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(*handle_));
        if (status == CUDA_SUCCESS) {
            handle_.reset();
        }
        return status;
    }

    CUmemGenericAllocationHandle release() noexcept {
        const CUmemGenericAllocationHandle handle = get();
        handle_.reset();
        return handle;
    }

   private:
    std::optional<CUmemGenericAllocationHandle> handle_;
};

}  // namespace nvshmemi

#endif
