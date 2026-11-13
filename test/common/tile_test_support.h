/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVSHMEMTEST_TILE_TEST_SUPPORT_H
#define NVSHMEMTEST_TILE_TEST_SUPPORT_H

#include "utils.h"
#include <cstring>
#include "non_abi/device/common/nvshmem_common.cuh"

// This probe runs only without multicast pointers; priority OFF still permits CFT fallback.
static __global__ void nvshmemtest_tile_cft_support(nvshmem_team_t team, bool *supported,
                                                    size_t dynamic_smem_size) {
    NVSHMEM_TEST_GIVE_SMEM(dynamic_smem_size);
#ifdef __CUDA_ARCH__
    if (!threadIdx.x) {
        auto *teami = nvshmemi_device_state_d.team_pool[team];
        supported[0] = nvshmemi_is_multicast_le_implemented(teami->mc_leid_with_flag, 16,
                                                            NVSHMEMI_THREADGROUP_WARP);
        supported[1] = nvshmemi_is_multicast_reduce_le_implemented<NVSHMEMI_THREADGROUP_WARP>(
            teami->mc_leid_with_flag, 16);
    }
#endif
    NVSHMEM_TEST_RELEASE_SMEM(dynamic_smem_size);
}

struct nvshmemtest_tile_support {
    bool multicast_ptr;
    bool cft_multicast;
    bool cft_reduce;
    int executed;
    int skipped;

    bool run(const char *name, const char *scope, bool reduction, int vector_width,
             size_t element_size, size_t major_elements, bool clipped_major) {
        // These tests' V2/V1 layouts deliberately exercise narrow rows and offsets.
        // Predicated tests trim four elements from dimension zero at the final tile.
        const bool aligned_rows =
            (major_elements * element_size) % 16 == 0 &&
            (!clipped_major || ((major_elements - 4) * element_size) % 16 == 0);
        const bool cft = vector_width == 4 && aligned_rows &&
                         (reduction ? (cft_reduce && strcmp(scope, "thread") != 0) : cft_multicast);
        if (multicast_ptr || cft) {
            ++executed;
            return true;
        }
        ++skipped;
        if (nvshmem_team_my_pe(NVSHMEM_TEAM_MC_SHARED) == 0) {
            printf(
                "SKIP %s scope=%s v%d: layout/scope requires multicast pointers or "
                "a supported CFT path\n",
                name, scope, vector_width);
        }
        return false;
    }

    int finish() const {
        if (nvshmem_team_my_pe(NVSHMEM_TEAM_MC_SHARED) == 0) {
            printf("Tile cases: executed=%d skipped=%d\n", executed, skipped);
        }
        return executed ? NVSHMEMX_SUCCESS : NVSHMEMX_ERROR_NOT_SUPPORTED;
    }
};

static nvshmemtest_tile_support nvshmemtest_query_tile_support(nvshmem_team_t team,
                                                               const void *source, const void *dest,
                                                               size_t dynamic_smem_size,
                                                               cudaStream_t stream) {
    nvshmemtest_tile_support result = {};
    result.multicast_ptr = nvshmemx_mc_ptr(team, source) && nvshmemx_mc_ptr(team, dest);
    if (!result.multicast_ptr) {
        bool *supported_d;
        bool supported[2];
        CUDA_CHECK(cudaMalloc(&supported_d, sizeof(supported)));
        CHECK_AND_ENABLE_MAX_DYNAMIC_SMEM(nvshmemtest_tile_cft_support, dynamic_smem_size);
        nvshmemtest_tile_cft_support<<<1, 256, dynamic_smem_size, stream>>>(team, supported_d,
                                                                            dynamic_smem_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(supported, supported_d, sizeof(supported), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(supported_d));
        result.cft_multicast = supported[0];
        result.cft_reduce = supported[1];
    }
    return result;
}
#endif
