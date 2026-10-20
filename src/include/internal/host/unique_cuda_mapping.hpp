/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMI_UNIQUE_CUDA_MAPPING_HPP
#define NVSHMEMI_UNIQUE_CUDA_MAPPING_HPP

#include <cstddef>
#include <utility>
#include "internal/host_transport/cudawrap.h"
#include "non_abi/c/nvshmemi_error_macros.h"

extern nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;

namespace nvshmemi {

class UniqueCudaMapping {
   public:
    UniqueCudaMapping() noexcept = default;
    explicit UniqueCudaMapping(void *address, size_t size) noexcept
        : address_(address), size_(size) {}
    ~UniqueCudaMapping() {
        if (reset() != CUDA_SUCCESS) {
            NVSHMEMI_WARN_PRINT("cuMemUnmap failed while releasing an owned mapping");
        }
    }

    UniqueCudaMapping(const UniqueCudaMapping &) = delete;
    UniqueCudaMapping &operator=(const UniqueCudaMapping &) = delete;
    UniqueCudaMapping(UniqueCudaMapping &&other) noexcept
        : address_(std::exchange(other.address_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    UniqueCudaMapping &operator=(UniqueCudaMapping &&other) noexcept {
        if (this != &other) {
            UniqueCudaMapping moved{std::move(other)};
            std::swap(address_, moved.address_);
            std::swap(size_, moved.size_);
        }
        return *this;
    }

    CUresult reset() noexcept {
        if (size_ == 0) {
            return CUDA_SUCCESS;
        }
        const CUresult status =
            CUPFN(nvshmemi_cuda_syms, cuMemUnmap(reinterpret_cast<CUdeviceptr>(address_), size_));
        if (status == CUDA_SUCCESS) {
            release();
        }
        return status;
    }

    void release() noexcept {
        address_ = nullptr;
        size_ = 0;
    }

   private:
    void *address_ = nullptr;
    size_t size_ = 0;
};

}  // namespace nvshmemi

#endif
