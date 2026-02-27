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
#include "reduce_common.h"
#include "cuda/std/tuple"

#if CUTLASS_ENABLED == 1
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#endif

using namespace std;

#define V4_TILE_SIZE_0 16
#define V4_TILE_SIZE_1 16
#define V4_MATRIX_SHAPE 32
#define V4_TENSOR_STRIDE_0 1
#define V4_TENSOR_STRIDE_1 tensor_size_0
#define V4_MAJOR_DIM 0
#define V4_MINOR_DIM 1
#define V4_TILE_SIZE_MAJOR V4_TILE_SIZE_0
#define V4_TILE_SIZE_MINOR V4_TILE_SIZE_1
#define V4_TENSOR_SIZE_MAJOR tensor_size_0
#define V4_TENSOR_SIZE_MINOR tensor_size_1
#define V4_TENSOR_STRIDE_MAJOR V4_TENSOR_STRIDE_0
#define V4_TENSOR_STRIDE_MINOR V4_TENSOR_STRIDE_1
#define MAKE_V4_STRIDE nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, V4_TENSOR_STRIDE_1);

#define V2_TILE_SIZE_0 14  // divisible by 2 but not by 4
#define V2_TILE_SIZE_1 14
#define V2_MATRIX_SHAPE 28
#define V2_TENSOR_STRIDE_0 tensor_size_1
#define V2_TENSOR_STRIDE_1 1
#define V2_MAJOR_DIM 1
#define V2_MINOR_DIM 0
#define V2_TILE_SIZE_MAJOR V2_TILE_SIZE_1
#define V2_TILE_SIZE_MINOR V2_TILE_SIZE_0
#define V2_TENSOR_SIZE_MAJOR tensor_size_1
#define V2_TENSOR_SIZE_MINOR tensor_size_0
#define V2_TENSOR_STRIDE_MAJOR V2_TENSOR_STRIDE_1
#define V2_TENSOR_STRIDE_MINOR V2_TENSOR_STRIDE_0
#define MAKE_V2_STRIDE nvshmemx::make_stride<int, ConstInt<1>>(V2_TENSOR_STRIDE_0, ConstInt<1>{});

#define V1_TILE_SIZE_0 14
#define V1_TILE_SIZE_1 14
#define V1_MATRIX_SHAPE 28
#define V1_TENSOR_STRIDE_0 1
#define V1_TENSOR_STRIDE_1 tensor_size_0
#define V1_MAJOR_DIM 0
#define V1_MINOR_DIM 1
#define V1_TILE_SIZE_MAJOR V1_TILE_SIZE_0
#define V1_TILE_SIZE_MINOR V1_TILE_SIZE_1
#define V1_TENSOR_SIZE_MAJOR tensor_size_0
#define V1_TENSOR_SIZE_MINOR tensor_size_1
#define V1_TENSOR_STRIDE_MAJOR V1_TENSOR_STRIDE_0
#define V1_TENSOR_STRIDE_MINOR V1_TENSOR_STRIDE_1
#define MAKE_V1_STRIDE nvshmemx::make_stride<int, int>(V1_TENSOR_STRIDE_0, V1_TENSOR_STRIDE_1);

#define LARGEST_DT uint64_t

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128

#define warpgroups_per_block ((blockDim.x) / WARPGROUP_SIZE)
#define warps_per_block ((blockDim.x) / WARP_SIZE)
#define threads_per_block (blockDim.x)
#define blocks_per_block (1)

#define GRID_SIZE_BLK_SCOPE 2
#define BLOCK_SIZE_BLK_SCOPE 256

#define GRID_SIZE_WARP_SCOPE 2
#define BLOCK_SIZE_WARP_SCOPE WARP_SIZE

#define GRID_SIZE_WG_SCOPE 2
#define BLOCK_SIZE_WG_SCOPE 256

#define GRID_SIZE_THRD_SCOPE 1
#define BLOCK_SIZE_THRD_SCOPE 2

#define DECL_TYPENAME_OP_REDUCE(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                     \
    __global__ void test_##TYPENAME##_##OP##_tile_allreduce_kernel##SC_SUFFIX(                    \
        nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t nelems, size_t tensor_size_0, \
        size_t tensor_size_1, int npes);

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES_AND_OPS(DECL_TYPENAME_OP_REDUCE)
#undef DECL_TYPENAME_OP_REDUCE

#define DEFN_TYPENAME_OP_REDUCE(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                \
    /* This is a persistent kernel which iterated over all the tiles */                           \
    /* Both Grid and block are 1D */                                                              \
    __global__ void test_##TYPENAME##_##OP##_tile_reduce_kernel_v##VLN##SC_SUFFIX(                \
        nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t nelems, size_t tensor_size_0, \
        size_t tensor_size_1, int npes) {                                                         \
        /* source is already initialized */                                                       \
        size_t num_tiles_major, num_tiles_minor;                                                  \
        /* create layout */                                                                       \
        constexpr int major_dim = V##VLN##_MAJOR_DIM;                                             \
        constexpr int minor_dim = V##VLN##_MINOR_DIM;                                             \
        num_tiles_major = V##VLN##_TENSOR_SIZE_MAJOR / V##VLN##_TILE_SIZE_MAJOR;                  \
        num_tiles_minor = V##VLN##_TENSOR_SIZE_MINOR / V##VLN##_TILE_SIZE_MINOR;                  \
        auto tile_shape = nvshmemx::make_shape(ConstInt<V##VLN##_TILE_SIZE_0>{},                  \
                                               ConstInt<V##VLN##_TILE_SIZE_1>{});                 \
        size_t tensor_stride_major_dim = V##VLN##_TENSOR_STRIDE_MAJOR;                            \
        size_t tensor_stride_minor_dim = V##VLN##_TENSOR_STRIDE_MINOR;                            \
        auto tile_stride = MAKE_V##VLN##_STRIDE;                                                  \
        auto tile_layout = nvshmemx::make_layout(tile_shape, tile_stride);                        \
        TYPE *src_tile_start;                                                                     \
        TYPE *dest_tile_start;                                                                    \
                                                                                                  \
        int team_id = (blockIdx.x * SC##s_per_block) + (threadIdx.x / nvshmtest_##SC##_size());   \
        int root = 0;                                                                             \
        int mype = nvshmem_team_my_pe(teams_dev[team_id]);                                        \
                                                                                                  \
        /* grid is 1D */                                                                          \
        /* depending on the scope, we process multiple tiles per block */                         \
        /* (= threads/warps/warpgroups per block) per iteration */                                \
        /* e.g., block scope --> 1 tile per iteration */                                          \
        /* e.g., warpgroup scope --> blocksize / warpgroup_size tiles per iteration */            \
        for (int i = blockIdx.x * SC##s_per_block; i < num_tiles_major * num_tiles_minor;         \
             i += gridDim.x * SC##s_per_block) {                                                  \
            int thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                              \
            int my_tile_idx = i + thrd_grp_id;                                                    \
            if (my_tile_idx >= (num_tiles_major * num_tiles_minor)) {                             \
                continue;                                                                         \
            }                                                                                     \
            size_t offset = (((my_tile_idx % num_tiles_major) * get<major_dim>(tile_shape) *      \
                              tensor_stride_major_dim) +                                          \
                             ((my_tile_idx / num_tiles_major) * get<minor_dim>(tile_shape) *      \
                              tensor_stride_minor_dim));                                          \
                                                                                                  \
            src_tile_start = source + offset;                                                     \
            dest_tile_start = dest + offset;                                                      \
                                                                                                  \
            /* create the tiles */                                                                \
            auto src_tensor =                                                                     \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(src_tile_start, tile_layout);       \
            auto dest_tensor =                                                                    \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(dest_tile_start, tile_layout);      \
                                                                                                  \
            /* tile reduce */                                                                     \
            /* No predicate */ struct empty {};                                                   \
            tile_errs_d = nvshmemx::tile_##OP##_rooted_reduce##SC_SUFFIX<                                              \
                decltype(src_tensor), decltype(dest_tensor), empty,                               \
                nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(                              \
                teams_dev[team_id], src_tensor, dest_tensor, empty{}, empty{}, root, 0);          \
            if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                  \
                return;                                                                             \
            }                                                                                      \
        }                                                                                         \
                                                                                                  \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                                \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams_dev[team_id], 0);           \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
        /* validate data */                                                                       \
        for (int i = blockIdx.x * SC##s_per_block; i < num_tiles_major * num_tiles_minor;         \
             i += gridDim.x * SC##s_per_block) {                                                  \
            size_t thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                           \
            size_t my_tile_idx = i + thrd_grp_id;                                                 \
            size_t offset = (((my_tile_idx % num_tiles_major) * get<major_dim>(tile_shape) *      \
                              tensor_stride_major_dim) +                                          \
                             ((my_tile_idx / num_tiles_major) * get<minor_dim>(tile_shape) *      \
                              tensor_stride_minor_dim));                                          \
                                                                                                  \
            src_tile_start = source + offset;                                                     \
            dest_tile_start = dest + offset;                                                      \
                                                                                                  \
            /* create the tiles */                                                                \
            auto src_tensor =                                                                     \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(src_tile_start, tile_layout);       \
            auto dest_tensor =                                                                    \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(dest_tile_start, tile_layout);      \
            /* No predicate */ struct empty {};                                                   \
            if (mype == root) {                                                                   \
                validate_##TYPENAME##_##OP##_allreduce_tile_data##SC_SUFFIX<                      \
                    decltype(src_tensor), decltype(dest_tensor), major_dim, minor_dim>(           \
                    teams_dev[team_id], offset, src_tensor, dest_tensor,                          \
                    V##VLN##_TENSOR_SIZE_MAJOR, V##VLN##_TENSOR_SIZE_MINOR,                       \
                    V##VLN##_TENSOR_SIZE_MAJOR, V##VLN##_TENSOR_SIZE_MINOR);                      \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                                \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams_dev[team_id], 0);           \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
        /* reset data */                                                                          \
        for (int i = blockIdx.x * SC##s_per_block; i < num_tiles_major * num_tiles_minor;         \
             i += gridDim.x * SC##s_per_block) {                                                  \
            size_t thrd_grp_id = threadIdx.x / nvshmtest_##SC##_size();                           \
            size_t my_tile_idx = i + thrd_grp_id;                                                 \
            size_t offset = (((my_tile_idx % num_tiles_major) * get<major_dim>(tile_shape) *      \
                              tensor_stride_major_dim) +                                          \
                             ((my_tile_idx / num_tiles_major) * get<minor_dim>(tile_shape) *      \
                              tensor_stride_minor_dim));                                          \
                                                                                                  \
            src_tile_start = source + offset;                                                     \
            dest_tile_start = dest + offset;                                                      \
                                                                                                  \
            auto src_tensor =                                                                     \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(src_tile_start, tile_layout);       \
            auto dest_tensor =                                                                    \
                nvshmemx::Tensor<TYPE, decltype(tile_layout)>(dest_tile_start, tile_layout);      \
            reset_##TYPENAME##_tile_data##SC_SUFFIX<decltype(dest_tensor), major_dim, minor_dim>( \
                offset, dest_tensor);                                                             \
        }                                                                                         \
        tile_errs_d = nvshmemx::tile_collective_wait##SC_SUFFIX<                                                \
            nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(teams_dev[team_id], 0);           \
        if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                      \
            return;                                                                                \
        }                                                                                          \
    }

#if CUTLASS_ENABLED == 1
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 1)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)

DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 1)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)

DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 1)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)

DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_half_t, cutlass::half_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_half_t, cutlass::half_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_half_t, cutlass::half_t, 1)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)
#endif

DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, half, half, 4)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, half, half, 2)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, half, half, 1)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, float, float, 4)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, float, float, 2)
DEFN_TYPENAME_OP_REDUCE(sum, block, _block, x, float, float, 1)

DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, half, half, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, half, half, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, half, half, 1)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, float, float, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, float, float, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warpgroup, _warpgroup, x, float, float, 1)

DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, half, half, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, half, half, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, half, half, 1)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, float, float, 4)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, float, float, 2)
DEFN_TYPENAME_OP_REDUCE(sum, warp, _warp, x, float, float, 1)

DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, half, half, 4)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, half, half, 2)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, half, half, 1)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, float, float, 4)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, float, float, 2)
DEFN_TYPENAME_OP_REDUCE(sum, thread, , x, float, float, 1)

#define DO_RDXN_TEST(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                            \
    init_##TYPENAME##_tile_data_kernel<<<8, 256, 0, cstrm>>>(                                      \
        NVSHMEM_TEAM_WORLD, reinterpret_cast<TYPE *>(source), num_elems);                          \
    CUDA_CHECK(cudaGetLastError());                                                                \
    CUDA_CHECK(cudaStreamSynchronize(cstrm));                                                      \
                                                                                                   \
    assert(V##VLN##_MATRIX_SHAPE % V##VLN##_TILE_SIZE_0 == 0);                                     \
    assert(V##VLN##_MATRIX_SHAPE % V##VLN##_TILE_SIZE_1 == 0);                                     \
                                                                                                   \
    DEBUG_PRINT("Launching test with scope: %s %s %s\n", #SC, #TYPENAME, #VLN);                    \
    test_##TYPENAME##_##OP##_tile_reduce_kernel_v##VLN##SC_SUFFIX<<<grid_size, block_size, 0,      \
                                                                    cstrm>>>(                      \
        teams_dev, (TYPE *)dest, (TYPE *)source, num_elems,                                        \
        V##VLN##_MATRIX_SHAPE /* tensor size 0*/, V##VLN##_MATRIX_SHAPE /* tensor size 1*/, npes); \
                                                                                                   \
    CUDA_CHECK(cudaGetLastError());                                                                \
    CUDA_CHECK(cudaStreamSynchronize(cstrm));                                                      \
    TILE_CHECK_ERRS();                                                                             \
    nvshmem_barrier_all();

int main(int argc, char **argv) {
    int status = 0;
    size_t max_shape = V4_MATRIX_SHAPE > V2_MATRIX_SHAPE ? V4_MATRIX_SHAPE : V2_MATRIX_SHAPE;
    max_shape = max_shape > V1_MATRIX_SHAPE ? max_shape : V1_MATRIX_SHAPE;
    size_t num_elems = max_shape * max_shape;
    size_t alloc_size = num_elems * 2 * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    cudaStream_t cstrm;
    char size_string[100];
    unsigned long long int errs;

    size_t grid_size;
    size_t block_size;
    nvshmem_team_t *teams, *teams_dev;

    int num_teams;

    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
    d_buffer = (LARGEST_DT *)nvshmem_calloc(alloc_size, 1);
    if (!d_buffer) {
        ERROR_PRINT("nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

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
    DO_RDXN_TEST(sum, thread, , x, half, half, 4)
    DO_RDXN_TEST(sum, thread, , x, half, half, 2)
    DO_RDXN_TEST(sum, thread, , x, half, half, 1)
    DO_RDXN_TEST(sum, thread, , x, float, float, 4)
    DO_RDXN_TEST(sum, thread, , x, float, float, 2)
    DO_RDXN_TEST(sum, thread, , x, float, float, 1)

#if CUTLASS_ENABLED == 1
    DO_RDXN_TEST(sum, thread, , x, cutlass_half_t, cutlass::half_t, 4)
    DO_RDXN_TEST(sum, thread, , x, cutlass_half_t, cutlass::half_t, 2)
    DO_RDXN_TEST(sum, thread, , x, cutlass_half_t, cutlass::half_t, 1)
    DO_RDXN_TEST(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
    DO_RDXN_TEST(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
    DO_RDXN_TEST(sum, thread, , x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)
#endif

    grid_size = GRID_SIZE_WARP_SCOPE;
    block_size = BLOCK_SIZE_WARP_SCOPE;
    DO_RDXN_TEST(sum, warp, _warp, x, half, half, 4)
    DO_RDXN_TEST(sum, warp, _warp, x, half, half, 2)
    DO_RDXN_TEST(sum, warp, _warp, x, half, half, 1)
    DO_RDXN_TEST(sum, warp, _warp, x, float, float, 4)
    DO_RDXN_TEST(sum, warp, _warp, x, float, float, 2)
    DO_RDXN_TEST(sum, warp, _warp, x, float, float, 1)

#if CUTLASS_ENABLED == 1
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 4)
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 2)
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_half_t, cutlass::half_t, 1)
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
    DO_RDXN_TEST(sum, warp, _warp, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)
#endif

    grid_size = GRID_SIZE_WG_SCOPE;
    block_size = BLOCK_SIZE_WG_SCOPE;
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, half, half, 4)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, half, half, 2)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, half, half, 1)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, float, float, 4)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, float, float, 2)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, float, float, 1)

#if CUTLASS_ENABLED == 1
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 4)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 2)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_half_t, cutlass::half_t, 1)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
    DO_RDXN_TEST(sum, warpgroup, _warpgroup, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)
#endif

    grid_size = GRID_SIZE_BLK_SCOPE;
    block_size = BLOCK_SIZE_BLK_SCOPE;
    DO_RDXN_TEST(sum, block, _block, x, half, half, 4)
    DO_RDXN_TEST(sum, block, _block, x, half, half, 2)
    DO_RDXN_TEST(sum, block, _block, x, half, half, 1)
    DO_RDXN_TEST(sum, block, _block, x, float, float, 4)
    DO_RDXN_TEST(sum, block, _block, x, float, float, 2)
    DO_RDXN_TEST(sum, block, _block, x, float, float, 1)

#if CUTLASS_ENABLED == 1
    DO_RDXN_TEST(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 4)
    DO_RDXN_TEST(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 2)
    DO_RDXN_TEST(sum, block, _block, x, cutlass_half_t, cutlass::half_t, 1)
    DO_RDXN_TEST(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 4)
    DO_RDXN_TEST(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 2)
    DO_RDXN_TEST(sum, block, _block, x, cutlass_bfloat16_t, cutlass::bfloat16_t, 1)
#endif

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
