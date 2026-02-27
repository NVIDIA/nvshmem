/*
 *  Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 *  See License.txt for license information
 */

 #include "utils.h"
 #include <string>
 #include <iostream>
 #include <sstream>
 #include "nvshmem.h"
 #include "nvshmemx.h"
 #include "test_teams.h"
 #include "coll_common.h"
 #include "device_host/nvshmem_common.cuh"
 #include "device_host/nvshmem_tensor.h"
 #include "reduce_common.h"
 #include "cuda/std/tuple"

 using namespace std;

 #define V4_TILE_SIZE_0 8
 #define V4_TILE_SIZE_1 16
 #define V4_MATRIX_SHAPE 32  // should not overflow fp16, bfloat16
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
 #define V2_MATRIX_SHAPE 28  // should not overflow fp16, bfloat16
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

 #define V1_TILE_SIZE_0 14  // not divisible by 2 and 4
 #define V1_TILE_SIZE_1 14
 #define V1_MATRIX_SHAPE 28  // should not overflow fp16, bfloat16
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

 #define DECL_TYPENAME_TILE_PUT(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                     \
     __global__ void test_##TYPENAME##_tile_put_kernel##SC_SUFFIX(                    \
         nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t nelems, size_t tensor_size_0, \
         size_t tensor_size_1, int npes);

 NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(DECL_TYPENAME_TILE_PUT, NA)
 #undef DECL_TYPENAME_TILE_PUT

#define VALIDATE_TILE_PUT_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)               \
    template <typename src_tensor_t, typename dst_tensor_t, int major_dim, int minor_dim>        \
    __device__ void validate_##TYPENAME##_copy_tile_data##SC_SUFFIX(                 \
        nvshmem_team_t team, size_t tile_start_elem_idx, src_tensor_t src, dst_tensor_t dest,    \
        size_t tensor_size_major, size_t tensor_size_minor, size_t boundary_major,               \
        size_t boundary_minor) {                                                                 \
        size_t flat_elem_idx;                                                                    \
        size_t idx_in_tile;                                                                      \
        int npes = nvshmem_team_n_pes(team);                                                     \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        size_t tot_elem_in_tile =                                                                \
            cuda::std::get<0>(dest.shape()) * cuda::std::get<1>(dest.shape());                   \
        TYPE expected = assign<TYPE>(0);                                                         \
        TYPE temp = assign<TYPE>(0);                                                             \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {                        \
            idx_in_tile = ((i % cuda::std::get<major_dim>(dest.shape())) *                       \
                           cuda::std::get<major_dim>(dest.stride())) +                           \
                          ((i / cuda::std::get<major_dim>(dest.shape())) *                       \
                           cuda::std::get<minor_dim>(dest.stride()));                            \
            flat_elem_idx = (tile_start_elem_idx + idx_in_tile);                                 \
            if constexpr (std::is_same_v<TYPE, half>) {                                          \
                /* data is received from my_pe-1 */                                              \
                expected = assign<TYPE>((my_pe >= 2049) ? 0 : 1);                                \
                if (!my_pe) {                                                                    \
                    expected = assign<TYPE>(((npes-1) >= 2048) ? 0 : 1);                         \
                }                                                                                \
            } else if constexpr (std::is_same_v<TYPE, __nv_bfloat16>) {                          \
                expected = assign<TYPE>((my_pe >= 257) ? 0 : 1);                                 \
                if (!my_pe) {                                                                    \
                    expected = assign<TYPE>(((npes-1) >= 256) ? 0 : 1);                          \
                }                                                                                \
            } else {                                                                             \
                if (!my_pe) {                                                                    \
                    expected = assign<TYPE>(flat_elem_idx + npes - 1);                           \
                } else {                                                                         \
                    expected = assign<TYPE>(flat_elem_idx + my_pe - 1);                          \
                }                                                                                \
            }                                                                                    \
            if (((flat_elem_idx % tensor_size_major) < boundary_major) &&                        \
                ((flat_elem_idx / tensor_size_major) < boundary_minor) &&                        \
                (*(dest.data() + idx_in_tile) != expected)) {                                    \
                print_err<TYPE>(*(dest.data() + idx_in_tile), expected, flat_elem_idx,           \
                                tot_elem_in_tile, team,                                          \
                                NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC));                  \
                atomicAdd(&errs_d, 1);                                                           \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(VALIDATE_TILE_PUT_DATA, NA)
#undef VALIDATE_TILE_PUT_DATA


 #define DEFN_TYPENAME_TILE_PUT(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                \
     /* This is a persistent kernel which iterated over all the tiles */                           \
     /* Both Grid and block are 1D */                                                              \
     __global__ void test_##TYPENAME##_tile_put_kernel_v##VLN##SC_SUFFIX(                       \
         nvshmem_team_t *teams_dev, TYPE *dest, TYPE *source, size_t nelems, size_t tensor_size_0, \
         size_t tensor_size_1, bool only_p2p, int npes, int my_pe) {                               \
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
         int dest_pe = (my_pe + 1) % npes;                                                         \
         int team_id = (blockIdx.x * SC##s_per_block) + (threadIdx.x / nvshmtest_##SC##_size());   \
                                                                                                   \
         auto boundary = nvshmemx::make_shape(int(tensor_size_0 - 4), int(tensor_size_1));         \
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
             /* tile put */                                                                        \
             /* predicate */                                                                       \
             nvshmemx::shape<int, int> start_coord;                                                \
             if (major_dim == 0) {                                                                 \
                 start_coord = nvshmemx::make_shape(                                               \
                     int((my_tile_idx % num_tiles_major) * V##VLN##_TILE_SIZE_MAJOR),              \
                     int((my_tile_idx / num_tiles_major) * V##VLN##_TILE_SIZE_MINOR));             \
             } else {                                                                              \
                 start_coord = nvshmemx::make_shape(                                               \
                     int((my_tile_idx / num_tiles_major) * V##VLN##_TILE_SIZE_MINOR),              \
                     int((my_tile_idx % num_tiles_major) * V##VLN##_TILE_SIZE_MAJOR));             \
             }                                                                                     \
             if (only_p2p) {                                                                       \
                tile_errs_d = nvshmemx::tile_put##SC_SUFFIX<                                                     \
                    decltype(src_tensor), decltype(dest_tensor), decltype(boundary),               \
                    nvshmemx::tile_algo_t::PEER_PUSH_NBI>(                                         \
                    src_tensor, dest_tensor, start_coord, boundary,                                \
                    dest_pe, 0);                                                                   \
             } else {                                                                              \
                tile_errs_d = nvshmemx::tile_put##SC_SUFFIX<                                                     \
                    decltype(src_tensor), decltype(dest_tensor), decltype(boundary),               \
                    nvshmemx::tile_algo_t::REMOTE_PUSH_NBI>(                                       \
                    src_tensor, dest_tensor, start_coord, boundary,                                \
                    dest_pe, 0);                                                                   \
                                                                                                   \
             }                                                                                     \
             if (tile_errs_d != NVSHMEMX_SUCCESS) {                                                  \
                return;                                                                             \
             }                                                                                      \
         }                                                                                         \
         nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(teams_dev[team_id]);                              \
                                                                                                   \
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
             validate_##TYPENAME##_copy_tile_data##SC_SUFFIX<                                      \
                 decltype(src_tensor), decltype(dest_tensor), major_dim, minor_dim>(               \
                 teams_dev[team_id], offset, src_tensor, dest_tensor, V##VLN##_TENSOR_SIZE_MAJOR,  \
                 V##VLN##_TENSOR_SIZE_MINOR, get<major_dim>(boundary), get<minor_dim>(boundary));  \
         }                                                                                         \
         nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(teams_dev[team_id]);                              \
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
         nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(teams_dev[team_id]);                              \
     }

 NVSHMEMTEST_TILE_REPT_SCOPES_AND_VLEN(DEFN_TYPENAME_TILE_PUT, NA, float, float)
 NVSHMEMTEST_TILE_REPT_SCOPES_AND_VLEN(DEFN_TYPENAME_TILE_PUT, NA, half, half)
 NVSHMEMTEST_TILE_REPT_SCOPES_AND_VLEN(DEFN_TYPENAME_TILE_PUT, NA, bfloat16, __nv_bfloat16)

 #define DO_PUT_TEST(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, VLN)                             \
     init_##TYPENAME##_tile_data_kernel<<<8, 256, 0, cstrm>>>(                                      \
         NVSHMEM_TEAM_WORLD, reinterpret_cast<TYPE *>(source), num_elems);                          \
     CUDA_CHECK(cudaGetLastError());                                                                \
     CUDA_CHECK(cudaStreamSynchronize(cstrm));                                                      \
                                                                                                    \
     assert(V##VLN##_MATRIX_SHAPE % V##VLN##_TILE_SIZE_0 == 0);                                     \
     assert(V##VLN##_MATRIX_SHAPE % V##VLN##_TILE_SIZE_1 == 0);                                     \
                                                                                                    \
     DEBUG_PRINT("Launching test with scope: %s %s %s\n", #SC, #TYPENAME, #VLN);                    \
     test_##TYPENAME##_tile_put_kernel_v##VLN##SC_SUFFIX<<<grid_size, block_size, 0,                \
                                                                        cstrm>>>(                   \
         teams_dev, (TYPE *)dest, (TYPE *)source, num_elems,                                        \
         V##VLN##_MATRIX_SHAPE /* tensor size 0*/, V##VLN##_MATRIX_SHAPE /* tensor size 1*/,        \
         p2p_only, npes, mype);                                                                    \
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
     unsigned long long int errs = 0;

     size_t grid_size;
     size_t block_size;
     nvshmem_team_t *teams, *teams_dev;
     int num_teams;
     bool p2p_only;

     read_args(argc, argv);
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
     mype = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
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
         if (nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &teams[i]) != 0) {
             ERROR_PRINT("nvshmem_team_split_strided failed %d\n",i);
             status = -1;
             goto out;
         }
     }

     CUDA_CHECK(cudaMalloc((void **)&teams_dev, num_teams * sizeof(nvshmem_team_t)));
     CUDA_CHECK(
         cudaMemcpy(teams_dev, teams, num_teams * sizeof(nvshmem_team_t), cudaMemcpyHostToDevice));

     source = d_buffer;
     dest = &source[num_elems];

     p2p_only = true;
     for (int i=0; i<npes; ++i) {
        if (!nvshmem_ptr(d_buffer, i)) {
            p2p_only = false;
            break;
        }
     }
     printf("[%d] Running with p2p only option: %d\n", mype, p2p_only);
     nvshmem_barrier_all();

     if (threadgroup_scope.type == NVSHMEM_THREAD || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
         grid_size = GRID_SIZE_THRD_SCOPE;
         block_size = BLOCK_SIZE_THRD_SCOPE;
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, thread, , , float, float)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, thread, , , half, half)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, thread, , , bfloat16, __nv_bfloat16)
     }

     if (threadgroup_scope.type == NVSHMEM_WARP || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
         grid_size = GRID_SIZE_WARP_SCOPE;
         block_size = BLOCK_SIZE_WARP_SCOPE;
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warp, _warp, x, float, float)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warp, _warp, x, half, half)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warp, _warp, x, bfloat16, __nv_bfloat16)
     }

     if (threadgroup_scope.type == NVSHMEM_WARPGROUP || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
         grid_size = GRID_SIZE_WG_SCOPE;
         block_size = BLOCK_SIZE_WG_SCOPE;
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warpgroup, _warpgroup, x, float, float)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warpgroup, _warpgroup, x, half, half)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, warpgroup, _warpgroup, x, bfloat16,
         __nv_bfloat16)
     }

     if (threadgroup_scope.type == NVSHMEM_BLOCK || threadgroup_scope.type == NVSHMEM_ALL_SCOPES) {
         grid_size = GRID_SIZE_BLK_SCOPE;
         block_size = BLOCK_SIZE_BLK_SCOPE;
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, block, _block, x, float, float)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, block, _block, x, half, half)
         NVSHMEMTEST_TILE_REPT_VLEN(DO_PUT_TEST, NA, block, _block, x, bfloat16, __nv_bfloat16)
     }

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
