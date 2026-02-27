/*
 *  Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 *  See License.txt for license information
 */

#include "utils.h"
#include <string>
#include <iostream>
#include <sstream>
#include "test_teams.h"
#include "coll_common.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_tensor.h"
#include "fcollect_common.h"
#include "cuda/std/tuple"

using namespace std;

#define LARGEST_DT uint64_t
#define NVSHMEM_INTERNAL_TEAMS 10

#define V4_SRC_TILE_SIZE_0 8
#define V4_SRC_TILE_SIZE_1 4
#define V4_SRC_MATRIX_SHAPE 32
#define V4_SRC_TENSOR_STRIDE_0 1
#define V4_SRC_TENSOR_STRIDE_1 src_tensor_size_0
#define V4_DST_TENSOR_STRIDE_0 1
#define V4_DST_TENSOR_STRIDE_1 dst_tensor_size_0
#define V4_MAJOR_DIM 0
#define V4_MINOR_DIM 1
#define V4_TILE_SIZE_MAJOR V4_SRC_TILE_SIZE_0
#define V4_TILE_SIZE_MINOR V4_SRC_TILE_SIZE_1
#define V4_SRC_TENSOR_SIZE_MAJOR src_tensor_size_0
#define V4_SRC_TENSOR_SIZE_MINOR src_tensor_size_1
#define V4_DST_TENSOR_SIZE_MAJOR dst_tensor_size_0
#define V4_DST_TENSOR_SIZE_MINOR dst_tensor_size_1
#define V4_TENSOR_STRIDE_MAJOR V4_TENSOR_STRIDE_0
#define V4_TENSOR_STRIDE_MINOR V4_TENSOR_STRIDE_1
#define MAKE_V4_SRC_STRIDE \
    nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, V4_SRC_TENSOR_STRIDE_1);
#define MAKE_V4_DST_STRIDE \
    nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, V4_DST_TENSOR_STRIDE_1);

#define V2_SRC_TILE_SIZE_0 14
#define V2_SRC_TILE_SIZE_1 14
#define V2_SRC_MATRIX_SHAPE 28
#define V2_SRC_TENSOR_STRIDE_0 1
#define V2_SRC_TENSOR_STRIDE_1 src_tensor_size_0
#define V2_DST_TENSOR_STRIDE_0 1
#define V2_DST_TENSOR_STRIDE_1 dst_tensor_size_0
#define V2_MAJOR_DIM 0
#define V2_MINOR_DIM 1
#define V2_TILE_SIZE_MAJOR V2_SRC_TILE_SIZE_0
#define V2_TILE_SIZE_MINOR V2_SRC_TILE_SIZE_1
#define V2_SRC_TENSOR_SIZE_MAJOR src_tensor_size_0
#define V2_SRC_TENSOR_SIZE_MINOR src_tensor_size_1
#define V2_DST_TENSOR_SIZE_MAJOR dst_tensor_size_0
#define V2_DST_TENSOR_SIZE_MINOR dst_tensor_size_1
#define V2_TENSOR_STRIDE_MAJOR V2_TENSOR_STRIDE_0
#define V2_TENSOR_STRIDE_MINOR V2_TENSOR_STRIDE_1
#define MAKE_V2_SRC_STRIDE \
    nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, V2_SRC_TENSOR_STRIDE_1);
#define MAKE_V2_DST_STRIDE \
    nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, V2_DST_TENSOR_STRIDE_1);

#define V1_SRC_TILE_SIZE_0 14
#define V1_SRC_TILE_SIZE_1 14
#define V1_SRC_MATRIX_SHAPE 28
#define V1_SRC_TENSOR_STRIDE_0 1
#define V1_SRC_TENSOR_STRIDE_1 src_tensor_size_0
#define V1_DST_TENSOR_STRIDE_0 1
#define V1_DST_TENSOR_STRIDE_1 dst_tensor_size_0
#define V1_MAJOR_DIM 0
#define V1_MINOR_DIM 1
#define V1_TILE_SIZE_MAJOR V1_SRC_TILE_SIZE_0
#define V1_TILE_SIZE_MINOR V1_SRC_TILE_SIZE_1
#define V1_SRC_TENSOR_SIZE_MAJOR src_tensor_size_0
#define V1_SRC_TENSOR_SIZE_MINOR src_tensor_size_1
#define V1_DST_TENSOR_SIZE_MAJOR dst_tensor_size_0
#define V1_DST_TENSOR_SIZE_MINOR dst_tensor_size_1
#define V1_TENSOR_STRIDE_MAJOR V1_TENSOR_STRIDE_0
#define V1_TENSOR_STRIDE_MINOR V1_TENSOR_STRIDE_1
#define MAKE_V1_SRC_STRIDE \
    nvshmemx::make_stride<int, int>(V1_SRC_TENSOR_STRIDE_0, V1_SRC_TENSOR_STRIDE_1);
#define MAKE_V1_DST_STRIDE \
    nvshmemx::make_stride<int, int>(V1_DST_TENSOR_STRIDE_0, V1_DST_TENSOR_STRIDE_1);

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128

#define warpgroups_per_block ((blockDim.x) / WARPGROUP_SIZE)
#define warps_per_block ((blockDim.x) / WARP_SIZE)
#define threads_per_block (blockDim.x)
#define blocks_per_block (1)

#define GRID_SIZE_BLK_SCOPE 2
#define BLOCK_SIZE_BLK_SCOPE 256

#define GRID_SIZE_WARP_SCOPE 2
#define BLOCK_SIZE_WARP_SCOPE 256

#define GRID_SIZE_WG_SCOPE 2
#define BLOCK_SIZE_WG_SCOPE 256

#define GRID_SIZE_THRD_SCOPE 1
#define BLOCK_SIZE_THRD_SCOPE 2

#define DECL_TYPENAME_ALLGATHER(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)          \
    __global__ void test_##TYPENAME##_tile_allgather_kernel##SC_SUFFIX(                \
        nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t src_tensor_size_0, \
        size_t src_tensor_size_1, size_t dst_tensor_size_0, size_t dst_tensor_size_1, int npes);

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(DECL_TYPENAME_ALLGATHER, )
#undef DECL_TYPENAME_ALLGATHER

#define DEFN_TYPENAME_ALLGATHER(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                 \
    /* This is a persistent kernel which iterated over all the tiles */                            \
    /* Both Grid and block are 1D */                                                               \
    __global__ void test_##TYPENAME##_tile_allgather_kernel_v##VLN##SC_SUFFIX(                     \
        nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t src_tensor_size_0,             \
        size_t src_tensor_size_1, size_t dst_tensor_size_0, size_t dst_tensor_size_1, int npes) {  \
        /* source is already initialized */                                                        \
        size_t src_num_tiles_major, src_num_tiles_minor;                                           \
        size_t dst_num_tiles_major;                                                                \
        /*size_t dst_num_tiles_minor;*/                                                            \
        /* create layout */                                                                        \
        constexpr int major_dim = V##VLN##_MAJOR_DIM;                                              \
        constexpr int minor_dim = V##VLN##_MINOR_DIM;                                              \
        auto src_tile_shape = nvshmemx::make_shape(ConstInt<V##VLN##_SRC_TILE_SIZE_0>{},           \
                                                   ConstInt<V##VLN##_SRC_TILE_SIZE_1>{});          \
        /* allgather along dim 1*/                                                                 \
        auto dst_tile_shape = nvshmemx::make_shape(ConstInt<V##VLN##_SRC_TILE_SIZE_0>{},           \
                                                   V##VLN##_SRC_TILE_SIZE_1 * npes);               \
        src_num_tiles_major = V##VLN##_SRC_TENSOR_SIZE_MAJOR / get<major_dim>(src_tile_shape);     \
        src_num_tiles_minor = V##VLN##_SRC_TENSOR_SIZE_MINOR / get<minor_dim>(src_tile_shape);     \
        dst_num_tiles_major = V##VLN##_DST_TENSOR_SIZE_MAJOR / get<major_dim>(dst_tile_shape);     \
        /*dst_num_tiles_minor = V##VLN##_DST_TENSOR_SIZE_MINOR / get<minor_dim>(dst_tile_shape);   \
        assert((src_num_tiles_major * src_num_tiles_minor) == (dst_num_tiles_major *               \
        dst_num_tiles_minor));*/                                                                   \
        auto src_tile_stride = MAKE_V##VLN##_SRC_STRIDE;                                           \
        auto src_tile_layout = nvshmemx::make_layout(src_tile_shape, src_tile_stride);             \
        auto dst_tile_stride = MAKE_V##VLN##_DST_STRIDE;                                           \
        auto dst_tile_layout = nvshmemx::make_layout(dst_tile_shape, dst_tile_stride);             \
                                                                                                   \
        TYPE *src_tile_start;                                                                      \
        TYPE *dest_tile_start;                                                                     \
                                                                                                   \
        int team_id = (blockIdx.x * SC##s_per_block) + (threadIdx.x / nvshmtest_##SC##_size());    \
                                                                                                   \
        /* grid is 1D */                                                                           \
        /* depending on the scope, we process multiple tiles per block */                          \
        /* (= threads/warps/warpgroups per block) per iteration */                                 \
        /* e.g., block scope --> 1 tile per iteration */                                           \
        /* e.g., warpgroup scope --> blocksize / warpgroup_size tiles per iteration */             \
        for (int i = blockIdx.x * SC##s_per_block; i < src_num_tiles_major * src_num_tiles_minor;  \
             i += gridDim.x * SC##s_per_block) {                                                   \
            int thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                               \
            int my_tile_idx = i + thrd_grp_id;                                                     \
            if (my_tile_idx >= (src_num_tiles_major * src_num_tiles_minor)) {                      \
                continue;                                                                          \
            }                                                                                      \
            size_t src_offset =                                                                    \
                (((my_tile_idx % src_num_tiles_major) * get<major_dim>(src_tile_shape) *           \
                  get<major_dim>(src_tile_stride)) +                                               \
                 ((my_tile_idx / src_num_tiles_major) * get<minor_dim>(src_tile_shape) *           \
                  get<minor_dim>(src_tile_stride)));                                               \
            size_t dst_offset =                                                                    \
                (((my_tile_idx % dst_num_tiles_major) * get<major_dim>(dst_tile_shape) *           \
                  get<major_dim>(dst_tile_stride)) +                                               \
                 ((my_tile_idx / dst_num_tiles_major) * get<minor_dim>(dst_tile_shape) *           \
                  get<minor_dim>(dst_tile_stride)));                                               \
                                                                                                   \
            src_tile_start = source + src_offset;                                                  \
            dest_tile_start = dest + dst_offset;                                                   \
                                                                                                   \
            /* create the tiles */                                                                 \
            auto src_tensor = nvshmemx::Tensor<TYPE, decltype(src_tile_layout)>(src_tile_start,    \
                                                                                src_tile_layout);  \
            auto dest_tensor = nvshmemx::Tensor<TYPE, decltype(dst_tile_layout)>(dest_tile_start,  \
                                                                                 dst_tile_layout); \
                                                                                                   \
            /* tile allgather */                                                                   \
            /* No predication */ struct empty {};                                                  \
            tile_errs_d = nvshmemx::tile_allgather##SC_SUFFIX<                                                   \
                decltype(src_tensor), decltype(dest_tensor), empty,                                \
                nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI>(                               \
                teams_dev[team_id], src_tensor, dest_tensor, empty{}, empty{}, 0);                 \
            if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                  \
                return;                                                                             \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                                 \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI>(teams_dev[team_id], 0);            \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
                                                                                                   \
        /* validate data */                                                                        \
        for (int i = blockIdx.x * SC##s_per_block; i < src_num_tiles_major * src_num_tiles_minor;  \
             i += gridDim.x * SC##s_per_block) {                                                   \
            size_t thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                            \
            size_t my_tile_idx = i + thrd_grp_id;                                                  \
            if (my_tile_idx >= (src_num_tiles_major * src_num_tiles_minor)) {                      \
                continue;                                                                          \
            }                                                                                      \
            size_t src_offset =                                                                    \
                (((my_tile_idx % src_num_tiles_major) * get<major_dim>(src_tile_shape) *           \
                  get<major_dim>(src_tile_stride)) +                                               \
                 ((my_tile_idx / src_num_tiles_major) * get<minor_dim>(src_tile_shape) *           \
                  get<minor_dim>(src_tile_stride)));                                               \
            size_t dst_offset =                                                                    \
                (((my_tile_idx % dst_num_tiles_major) * get<major_dim>(dst_tile_shape) *           \
                  get<major_dim>(dst_tile_stride)) +                                               \
                 ((my_tile_idx / dst_num_tiles_major) * get<minor_dim>(dst_tile_shape) *           \
                  get<minor_dim>(dst_tile_stride)));                                               \
                                                                                                   \
            src_tile_start = source + src_offset;                                                  \
            dest_tile_start = dest + dst_offset;                                                   \
                                                                                                   \
            /* create the tiles */                                                                 \
            auto src_tensor = nvshmemx::Tensor<TYPE, decltype(src_tile_layout)>(src_tile_start,    \
                                                                                src_tile_layout);  \
            auto dest_tensor = nvshmemx::Tensor<TYPE, decltype(dst_tile_layout)>(dest_tile_start,  \
                                                                                 dst_tile_layout); \
            validate_##TYPENAME##_tile_allgather_data##SC_SUFFIX<                                  \
                decltype(src_tensor), decltype(dest_tensor), major_dim, minor_dim>(                \
                teams_dev[team_id], dst_offset, src_tensor, dest_tensor,                           \
                V##VLN##_DST_TENSOR_SIZE_MAJOR, V##VLN##_DST_TENSOR_SIZE_MINOR,                    \
                V##VLN##_DST_TENSOR_SIZE_MAJOR, V##VLN##_DST_TENSOR_SIZE_MINOR);                   \
        }                                                                                          \
                                                                                                   \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                   \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI>(teams_dev[team_id], 0);            \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
        for (int i = blockIdx.x * SC##s_per_block; i < src_num_tiles_major * src_num_tiles_minor;  \
             i += gridDim.x * SC##s_per_block) {                                                   \
            size_t thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                            \
            size_t my_tile_idx = i + thrd_grp_id;                                                  \
            if (my_tile_idx >= (src_num_tiles_major * src_num_tiles_minor)) {                      \
                continue;                                                                          \
            }                                                                                      \
            size_t src_offset =                                                                    \
                (((my_tile_idx % src_num_tiles_major) * get<major_dim>(src_tile_shape) *           \
                  get<major_dim>(src_tile_stride)) +                                               \
                 ((my_tile_idx / src_num_tiles_major) * get<minor_dim>(src_tile_shape) *           \
                  get<minor_dim>(src_tile_stride)));                                               \
            size_t dst_offset =                                                                    \
                (((my_tile_idx % dst_num_tiles_major) * get<major_dim>(dst_tile_shape) *           \
                  get<major_dim>(dst_tile_stride)) +                                               \
                 ((my_tile_idx / dst_num_tiles_major) * get<minor_dim>(dst_tile_shape) *           \
                  get<minor_dim>(dst_tile_stride)));                                               \
                                                                                                   \
            src_tile_start = source + src_offset;                                                  \
            dest_tile_start = dest + dst_offset;                                                   \
                                                                                                   \
            /* create the tiles */                                                                 \
            auto src_tensor = nvshmemx::Tensor<TYPE, decltype(src_tile_layout)>(src_tile_start,    \
                                                                                src_tile_layout);  \
            auto dest_tensor = nvshmemx::Tensor<TYPE, decltype(dst_tile_layout)>(dest_tile_start,  \
                                                                                 dst_tile_layout); \
            reset_##TYPENAME##_tile_data##SC_SUFFIX<decltype(dest_tensor), major_dim, minor_dim>(  \
                dst_offset, dest_tensor);                                                          \
        }                                                                                          \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                   \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI>(teams_dev[team_id], 0);            \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
    }

DEFN_TYPENAME_ALLGATHER(, block, _block, x, half, half, 4)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, half, half, 2)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, half, half, 1)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, bfloat16, __nv_bfloat16, 4)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, bfloat16, __nv_bfloat16, 2)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, bfloat16, __nv_bfloat16, 1)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, float, float, 4)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, float, float, 2)
DEFN_TYPENAME_ALLGATHER(, block, _block, x, float, float, 1)

DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, half, half, 4)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, half, half, 2)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, half, half, 1)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 4)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 2)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 1)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, float, float, 4)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, float, float, 2)
DEFN_TYPENAME_ALLGATHER(, warpgroup, _warpgroup, x, float, float, 1)

DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, half, half, 4)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, half, half, 2)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, half, half, 1)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, bfloat16, __nv_bfloat16, 4)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, bfloat16, __nv_bfloat16, 2)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, bfloat16, __nv_bfloat16, 1)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, float, float, 4)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, float, float, 2)
DEFN_TYPENAME_ALLGATHER(, warp, _warp, x, float, float, 1)

DEFN_TYPENAME_ALLGATHER(, thread, , x, half, half, 4)
DEFN_TYPENAME_ALLGATHER(, thread, , x, half, half, 2)
DEFN_TYPENAME_ALLGATHER(, thread, , x, half, half, 1)
DEFN_TYPENAME_ALLGATHER(, thread, , x, bfloat16, __nv_bfloat16, 4)
DEFN_TYPENAME_ALLGATHER(, thread, , x, bfloat16, __nv_bfloat16, 2)
DEFN_TYPENAME_ALLGATHER(, thread, , x, bfloat16, __nv_bfloat16, 1)
DEFN_TYPENAME_ALLGATHER(, thread, , x, float, float, 4)
DEFN_TYPENAME_ALLGATHER(, thread, , x, float, float, 2)
DEFN_TYPENAME_ALLGATHER(, thread, , x, float, float, 1)

#define DO_ALLGATHER_TEST(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                       \
    init_##TYPENAME##_tile_allgather_data_kernel<<<8, 256, 0, cstrm>>>(                            \
        NVSHMEM_TEAM_WORLD, reinterpret_cast<TYPE *>(source),                                      \
        V##VLN##_SRC_MATRIX_SHAPE /* src size 0*/, V##VLN##_SRC_MATRIX_SHAPE /* src size 1*/,      \
        V##VLN##_SRC_TILE_SIZE_0, V##VLN##_SRC_TILE_SIZE_1, 1 /* dstStride major*/,                \
        V##VLN##_SRC_MATRIX_SHAPE /* dstStride minor*/, 0 /*along major_dim*/);                    \
    CUDA_CHECK(cudaGetLastError());                                                                \
    CUDA_CHECK(cudaStreamSynchronize(cstrm));                                                      \
                                                                                                   \
    assert(V##VLN##_SRC_MATRIX_SHAPE % V##VLN##_SRC_TILE_SIZE_0 == 0);                             \
    assert(V##VLN##_SRC_MATRIX_SHAPE % V##VLN##_SRC_TILE_SIZE_1 == 0);                             \
                                                                                                   \
    DEBUG_PRINT("Launching test with scope: %s %s %s\n", #SC, #TYPENAME, #VLN);                    \
    test_##TYPENAME##_tile_allgather_kernel_v##VLN##SC_SUFFIX<<<grid_size, block_size, 0,          \
                                                                cstrm>>>(                          \
        teams_dev, (TYPE *)dest, (TYPE *)source, V##VLN##_SRC_MATRIX_SHAPE /* src tensor size 0*/, \
        V##VLN##_SRC_MATRIX_SHAPE /* src tensor size 1*/,                                          \
        V##VLN##_SRC_MATRIX_SHAPE /* dst tensor size 0*/,                                          \
        V##VLN##_SRC_MATRIX_SHAPE * npes /* dst tensor size 1*/, npes);                            \
    CUDA_CHECK(cudaGetLastError());                                                                \
    CUDA_CHECK(cudaStreamSynchronize(cstrm));                                                      \
    TILE_CHECK_ERRS();                                                                             \
    nvshmem_barrier_all();

int main(int argc, char **argv) {
    int status = 0;
    size_t max_shape =
        V4_SRC_MATRIX_SHAPE > V2_SRC_MATRIX_SHAPE ? V4_SRC_MATRIX_SHAPE : V2_SRC_MATRIX_SHAPE;
    max_shape = max_shape > V1_SRC_MATRIX_SHAPE ? max_shape : V1_SRC_MATRIX_SHAPE;
    size_t num_elems = max_shape * max_shape;
    size_t alloc_size = num_elems * (8 + 1) * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    cudaStream_t cstrm;
    char size_string[100];
    unsigned long long int errs;
    const char *max_team_env_value = (const char *)getenv("NVSHMEM_MAX_TEAMS");

    size_t grid_size;
    size_t block_size;
    nvshmem_team_t *teams, *teams_dev;

    int num_teams;
    /* teams per scope
     * thread: grid_size * block_size; //creating 1 team per thread
     * warp: grid_size * (block_size/WARP_SIZE); //creating 1 team per warp
     * warpgroup: grid_size * (block_size/WARPGROUP_SIZE); //creating 1 team per warpgroup
     * block: grid_size; //creating 1 team per block
     */
    // determine maximum number of teams across all scopes
    num_teams = (GRID_SIZE_THRD_SCOPE * BLOCK_SIZE_THRD_SCOPE);
    num_teams = num_teams > (GRID_SIZE_WARP_SCOPE * (BLOCK_SIZE_WARP_SCOPE / WARP_SIZE))
                    ? num_teams
                    : (GRID_SIZE_WARP_SCOPE * (BLOCK_SIZE_WARP_SCOPE / WARP_SIZE));
    num_teams = num_teams > (GRID_SIZE_WG_SCOPE * (BLOCK_SIZE_WG_SCOPE / WARPGROUP_SIZE))
                    ? num_teams
                    : (GRID_SIZE_WG_SCOPE * (BLOCK_SIZE_WG_SCOPE / WARPGROUP_SIZE));
    num_teams = num_teams > (GRID_SIZE_BLK_SCOPE) ? num_teams : (GRID_SIZE_BLK_SCOPE);

    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        status = -1;
        goto out;
    }

    DEBUG_PRINT("Requested number of teams: %d\n", num_teams + NVSHMEM_INTERNAL_TEAMS);
    if (!max_team_env_value) {
        sprintf(size_string, "%d", num_teams + NVSHMEM_INTERNAL_TEAMS);
        status = setenv("NVSHMEM_MAX_TEAMS", size_string, 1);
    }
    if (status) {
        ERROR_PRINT("setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
    alloc_size = num_elems * (npes + 1) * sizeof(LARGEST_DT);
    d_buffer = (LARGEST_DT *)nvshmem_calloc(alloc_size, 1);
    if (!d_buffer) {
        ERROR_PRINT("nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    teams = (nvshmem_team_t *)malloc(num_teams * sizeof(nvshmem_team_t));

    // create teams
    for (int i = 0; i < num_teams; i++) {
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &teams[i]);
    }

    CUDA_CHECK(cudaMalloc((void **)&teams_dev, num_teams * sizeof(nvshmem_team_t)));
    CUDA_CHECK(
        cudaMemcpy(teams_dev, teams, num_teams * sizeof(nvshmem_team_t), cudaMemcpyHostToDevice));

    source = d_buffer;
    dest = &source[num_elems];

    nvshmem_barrier_all();

    grid_size = GRID_SIZE_THRD_SCOPE;
    block_size = BLOCK_SIZE_THRD_SCOPE;
    DO_ALLGATHER_TEST(, thread, , x, half, half, 4);
    DO_ALLGATHER_TEST(, thread, , x, half, half, 2);
    DO_ALLGATHER_TEST(, thread, , x, half, half, 1);
    DO_ALLGATHER_TEST(, thread, , x, bfloat16, __nv_bfloat16, 4);
    DO_ALLGATHER_TEST(, thread, , x, bfloat16, __nv_bfloat16, 2);
    DO_ALLGATHER_TEST(, thread, , x, bfloat16, __nv_bfloat16, 1);
    DO_ALLGATHER_TEST(, thread, , x, float, float, 4);
    DO_ALLGATHER_TEST(, thread, , x, float, float, 2);
    DO_ALLGATHER_TEST(, thread, , x, float, float, 1);

    grid_size = GRID_SIZE_WARP_SCOPE;
    block_size = BLOCK_SIZE_WARP_SCOPE;
    DO_ALLGATHER_TEST(, warp, _warp, x, half, half, 4);
    DO_ALLGATHER_TEST(, warp, _warp, x, half, half, 2);
    DO_ALLGATHER_TEST(, warp, _warp, x, half, half, 1);
    DO_ALLGATHER_TEST(, warp, _warp, x, bfloat16, __nv_bfloat16, 4);
    DO_ALLGATHER_TEST(, warp, _warp, x, bfloat16, __nv_bfloat16, 2);
    DO_ALLGATHER_TEST(, warp, _warp, x, bfloat16, __nv_bfloat16, 1);
    DO_ALLGATHER_TEST(, warp, _warp, x, float, float, 4);
    DO_ALLGATHER_TEST(, warp, _warp, x, float, float, 2);
    DO_ALLGATHER_TEST(, warp, _warp, x, float, float, 1);

    grid_size = GRID_SIZE_WG_SCOPE;
    block_size = BLOCK_SIZE_WG_SCOPE;
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, half, half, 4);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, half, half, 2);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, half, half, 1);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 4);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 2);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, bfloat16, __nv_bfloat16, 1);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, float, float, 4);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, float, float, 2);
    DO_ALLGATHER_TEST(, warpgroup, _warpgroup, x, float, float, 1);

    grid_size = GRID_SIZE_BLK_SCOPE;
    block_size = BLOCK_SIZE_BLK_SCOPE;
    DO_ALLGATHER_TEST(, block, _block, x, half, half, 4);
    DO_ALLGATHER_TEST(, block, _block, x, half, half, 2);
    DO_ALLGATHER_TEST(, block, _block, x, half, half, 1);
    DO_ALLGATHER_TEST(, block, _block, x, bfloat16, __nv_bfloat16, 4);
    DO_ALLGATHER_TEST(, block, _block, x, bfloat16, __nv_bfloat16, 2);
    DO_ALLGATHER_TEST(, block, _block, x, bfloat16, __nv_bfloat16, 1);
    DO_ALLGATHER_TEST(, block, _block, x, float, float, 4);
    DO_ALLGATHER_TEST(, block, _block, x, float, float, 2);
    DO_ALLGATHER_TEST(, block, _block, x, float, float, 1);

    COLL_CHECK_ERRS_D();
    nvshmem_barrier_all();
    DEBUG_PRINT("%d Test passed\n", nvshmem_my_pe());

    // free teams
    for (int i = 0; i < num_teams; i++) {
        nvshmem_team_destroy(teams[i]);
    }
    CUDA_CHECK(cudaFree(teams_dev));
    free(teams);

    nvshmem_free(d_buffer);
    CUDA_CHECK(cudaStreamDestroy(cstrm));
    finalize_wrapper();

out:
    if (status) {
        return status;
    }
    return errs;
}
