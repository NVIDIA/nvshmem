/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include <nvshmem.h>
#include <nvshmemx.h>

namespace cutlass {
namespace comm {
namespace collective {

using namespace cute;

template <class ElementT_, class TileShape_, class StrideMNL_>
class CollectiveAllReduceMulticastWarpSpecialized {
   public:
    using ElementT = ElementT_;
    using TileShape = TileShape_;
    using StrideMNL = StrideMNL_;

    struct Arguments {
        ElementT* ptr_aux = nullptr;  // start pointer of matrix
        ElementT* out_ptr = nullptr;  // start pointer of matrix
        StrideMNL stride;
        int rank;
        int world_size;
        nvshmem_team_t* teams = nullptr;
    };

    struct Params {
        ElementT* ptr_aux = nullptr;
        ElementT* out_ptr = nullptr;
        StrideMNL stride;
        int rank;
        int world_size;
        Layout<Shape<int, int>> tile_layout;
        nvshmem_team_t* teams = nullptr;
    };

    __device__ void debug_print() {
        if (threadIdx.x % 128 == 0) {
            printf(
                "Inside CollectiveAllReduceMulticastWarpSpecialized %lu rank: %d world: %d,teams: "
                "%d \n",
                sizeof(ElementT_), params_ptr->rank, params_ptr->world_size, params_ptr->teams[0]);
        }
    }

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                    Arguments const& args) {
        // Append 1s until problem shape is rank-4
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_mnkl;

        int m_tiles = ceil_div(M, size<0>(TileShape{}));
        int n_tiles = ceil_div(N, size<1>(TileShape{}));
        //  number of tiles in each dimension
        auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

        return {
            args.ptr_aux,    args.out_ptr, args.stride, args.rank,
            args.world_size, tile_layout,  args.teams,
        };
    }

    const Params* params_ptr;

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized() {}

    CUTLASS_HOST_DEVICE
    CollectiveAllReduceMulticastWarpSpecialized(Params const& params) : params_ptr(&params) {}

    template <class ProblemShapeMNKL, class TileCoordMNKL>
    CUTLASS_DEVICE void do_allreduce(ProblemShapeMNKL const& problem_shape,
                                     TileCoordMNKL const& tile_coord) {
        auto [M, N, K, L] = problem_shape;
        auto [m, n, k, l] = tile_coord;

        if (m >= size<0>(params_ptr->tile_layout.shape()) ||
            n >= size<1>(params_ptr->tile_layout.shape())) {
            // early exit if out of bound
            return;
        }

        int tile_index = params_ptr->tile_layout(m, n);
        int tiles_per_rank =
            cute::ceil_div(cute::product(params_ptr->tile_layout.shape()), params_ptr->world_size);
        // only root PE will do reduction for this tile
        int root = tile_index / tiles_per_rank;

        Tensor mAux = make_tensor(params_ptr->ptr_aux,
                                  make_layout(make_shape(M, N, L), params_ptr->stride));  // (M,N,L)
        Tensor mAux_out = make_tensor(
            params_ptr->out_ptr, make_layout(make_shape(M, N, L), params_ptr->stride));  // (M,N,L)
        Tensor gAux =
            local_tile(mAux, take<0, 2>(TileShape{}), make_coord(m, n, l));  // (TILE_M,TILE_N)
        Tensor gAux_out =
            local_tile(mAux_out, take<0, 2>(TileShape{}), make_coord(m, n, l));  // (TILE_M,TILE_N)

        // predication tensor
        Tensor coordAux = make_identity_tensor(shape(mAux));
        Tensor pAux =
            local_tile(coordAux, take<0, 2>(TileShape{}), make_coord(m, n, l));  // (CTA_M,CTA_N)
        auto boundary = nvshmemx::make_shape<int, int>(M, N);
        auto start_coord = nvshmemx::make_shape<int, int>(size<0>(pAux(0, 0)), size<1>(pAux(0, 0)));

        // Call AR
        auto tensor_shape = nvshmemx::make_shape(M, N);
        auto tensor_stride =
            nvshmemx::make_stride(size<0>(params_ptr->stride), size<1>(params_ptr->stride));
        nvshmemx::Tensor srcTensor =
            nvshmemx::Tensor(gAux.data(), nvshmemx::make_layout(tensor_shape, tensor_stride));
        nvshmemx::Tensor dstTensor =
            nvshmemx::Tensor(gAux_out.data(), nvshmemx::make_layout(tensor_shape, tensor_stride));
        int blkId = blockIdx.x + gridDim.x * blockIdx.y;
        nvshmemx::tile_sum_allreduce_warpgroup<decltype(srcTensor), decltype(dstTensor),
                                               decltype(boundary),
                                               nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
            params_ptr->teams[blkId], srcTensor, dstTensor, start_coord, boundary, root, 0);
    }

    CUTLASS_DEVICE
    void tile_collective_wait() {
        int blkId = blockIdx.x + gridDim.x * blockIdx.y;
        nvshmemx::tile_collective_wait_warpgroup<
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(params_ptr->teams[blkId], 0);
    }
};

}  // namespace collective
}  // namespace comm
}  // namespace cutlass
