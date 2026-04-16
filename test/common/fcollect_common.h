/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEMTEST_FCOLLECT_COMMON_H
#define NVSHMEMTEST_FCOLLECT_COMMON_H

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_tensor.h"
#include "utils.h"
#include "coll_common.h"
#include "cuda/std/tuple"
#include <iostream>

__device__ unsigned long long int errs_d;

#define DECL_INIT_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                         \
    __device__ void init_##TYPENAME##_fcollect_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                               size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_INIT_FCOLLECT_DATA)
#undef DECL_INIT_FCOLLECT_DATA

#define DECL_VALIDATE_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)           \
    __device__ void validate_##TYPENAME##_fcollect_data##SC_SUFFIX(nvshmem_team_t team, \
                                                                   TYPE *dest, size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_VALIDATE_FCOLLECT_DATA)
#undef DECL_VALIDATE_FCOLLECT_DATA

#define DECL_RESET_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                        \
    __device__ void reset_##TYPENAME##_fcollect_data_##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                 size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(DECL_RESET_FCOLLECT_DATA)
#undef DECL_RESET_COLLECT_DATA

#define DECL_INIT_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                        \
    __global__ void init_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                           size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_INIT_FCOLLECT_DATA_KERNEL)
#undef DECL_INIT_FCOLLECT_DATA_KERNEL

#define DECL_VALIDATE_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void validate_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                               size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_VALIDATE_FCOLLECT_DATA_KERNEL)
#undef DECL_VALIDATE_FCOLLECT_DATA_KERNEL

#define DECL_RESET_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                      \
    __global__ void reset_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                            size_t nelems);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_RESET_FCOLLECT_DATA_KERNEL)
#undef DECL_RESET_FCOLLECT_DATA_KERNEL

#define INIT_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                              \
    __device__ void init_##TYPENAME##_fcollect_data##SC_SUFFIX(nvshmem_team_t team, TYPE *source, \
                                                               size_t nelems) {                   \
        int my_pe = nvshmem_team_my_pe(team);                                                     \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
                                                                                                  \
        for (size_t i = myIdx; i < nelems; i += groupSize) {                                      \
            source[i] = assign<TYPE>(my_pe * nelems + i);                                         \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(INIT_FCOLLECT_DATA)
#undef INIT_FCOLLECT_DATA

#define INIT_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                             \
    __global__ void init_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *source, \
                                                           size_t nelems) {                   \
        init_##TYPENAME##_fcollect_data_block(team, source, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(INIT_FCOLLECT_DATA_KERNEL)
#undef INIT_FCOLLECT_DATA_KERNEL

#define VALIDATE_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                          \
    __device__ void validate_##TYPENAME##_fcollect_data##SC_SUFFIX(nvshmem_team_t team,           \
                                                                   TYPE *dest, size_t nelems) {   \
        int n_pes = nvshmem_team_n_pes(team);                                                     \
        int my_pe = nvshmem_team_my_pe(team);                                                     \
        int myIdx = nvshmtest_thread_id_in_##SC();                                                \
        int groupSize = nvshmtest_##SC##_size();                                                  \
                                                                                                  \
        for (size_t i = 0; i < n_pes; i++) {                                                      \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                  \
                TYPE expected = assign<TYPE>(i * nelems + j);                                     \
                if (dest[i * nelems + j] != expected) {                                           \
                    print_err<TYPE>(dest[i * nelems + j], expected, i * nelems + j, nelems, team, \
                                    NVSHMEMTEST_ERRSTR_FORMAT_1(TYPENAME, SC));                   \
                    atomicAdd(&errs_d, 1);                                                        \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        nvshmtest_##SC##_sync();                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(VALIDATE_FCOLLECT_DATA)
#undef VALIDATE_FCOLLECT_DATA

#define VALIDATE_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void validate_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                               size_t nelems) {                 \
        validate_##TYPENAME##_fcollect_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(VALIDATE_FCOLLECT_DATA_KERNEL)
#undef VALIDATE_FCOLLECT_DATA_KERNEL

#define RESET_FCOLLECT_DATA(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                            \
    __device__ void reset_##TYPENAME##_fcollect_data##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, \
                                                                size_t nelems) {                 \
        int n_pes = nvshmem_team_n_pes(team);                                                    \
        int my_pe = nvshmem_team_my_pe(team);                                                    \
        int myIdx = nvshmtest_thread_id_in_##SC();                                               \
        int groupSize = nvshmtest_##SC##_size();                                                 \
        for (size_t i = 0; i < n_pes; i++) {                                                     \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                 \
                dest[i * nelems + j] = assign<TYPE>(0);                                          \
            }                                                                                    \
        }                                                                                        \
        nvshmtest_##SC##_sync();                                                                 \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(RESET_FCOLLECT_DATA)
#undef RESET_FCOLLECT_DATA

#define RESET_FCOLLECT_DATA_KERNEL(TYPENAME, TYPE)                                           \
    __global__ void reset_##TYPENAME##_fcollect_data_kernel(nvshmem_team_t team, TYPE *dest, \
                                                            size_t nelems) {                 \
        reset_##TYPENAME##_fcollect_data_block(team, dest, nelems);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(RESET_FCOLLECT_DATA_KERNEL)
#undef RESET_FCOLLECT_DATA_KERNEL

/*********** TILE Variants ***********/

#define DECL_INIT_TILE_ALLGATHER_DATA_KERNEL(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __global__ void init_##TYPENAME##_tile_allgather_data_kernel(nvshmem_team_t team,      \
                                                                 TYPE *source, size_t nelems);
NVSHMEMTEST_TILE_REPT_TYPES(DECL_INIT_TILE_ALLGATHER_DATA_KERNEL, , , , )
#undef DECL_INIT_TILE_ALLGATHER_DATA_KERNEL

#define INIT_TILE_ALLGATHER_DATA_KERNEL(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)        \
    __global__ void init_##TYPENAME##_tile_allgather_data_kernel(                            \
        nvshmem_team_t team, TYPE *source, size_t size_major, size_t size_minor,             \
        size_t tile_size_major, size_t tile_size_minor, size_t dst_stride_major,             \
        size_t dst_stride_minor, bool along_major_dim) {                                     \
        int my_pe = nvshmem_team_my_pe(team);                                                \
        int npes = nvshmem_team_n_pes(team);                                                 \
        int tileId_major, tileId_minor, final_coord_major, final_coord_minor;                \
        int coord_in_tile_major, coord_in_tile_minor;                                        \
        int myIdx = threadIdx.x + blockIdx.x * blockDim.x;                                   \
        int groupSize = blockDim.x * gridDim.x;                                              \
        for (size_t i = myIdx; i < size_major * size_minor; i += groupSize) {                \
            tileId_major = (i % size_major) / tile_size_major;                               \
            tileId_minor = (i / size_major) / tile_size_minor;                               \
            coord_in_tile_major = (i % size_major) % tile_size_major;                        \
            coord_in_tile_minor = (i / size_major) % tile_size_minor;                        \
            if (along_major_dim) {                                                           \
                final_coord_major =                                                          \
                    (((tileId_major * npes * tile_size_major) + (my_pe * tile_size_major)) + \
                     coord_in_tile_major);                                                   \
                final_coord_minor = (i / size_major);                                        \
            } else {                                                                         \
                final_coord_major = (i % size_major);                                        \
                final_coord_minor =                                                          \
                    (((tileId_minor * npes * tile_size_minor) + (my_pe * tile_size_minor)) + \
                     coord_in_tile_minor);                                                   \
            }                                                                                \
            source[i] = assign<TYPE>((final_coord_major * dst_stride_major) +                \
                                     (final_coord_minor * dst_stride_minor));                \
        }                                                                                    \
    }

NVSHMEMTEST_TILE_REPT_TYPES(INIT_TILE_ALLGATHER_DATA_KERNEL, , , , )
#undef INIT_TILE_ALLGATHER_DATA_KERNEL

#define DECL_VALIDATE_TILE_ALLGATHER_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    template <typename src_tensor_t, typename dst_tensor_t>                             \
    __device__ void validate_##TYPENAME##_tile_allgather_data##SC_SUFFIX(               \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dest);

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(DECL_VALIDATE_TILE_ALLGATHER_DATA, )
#undef DECL_VALIDATE_TILE_ALLGATHER_DATA

#define VALIDATE_TILE_ALLGATHER_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)            \
    template <typename src_tensor_t, typename dst_tensor_t, int major_dim, int minor_dim>     \
    __device__ void validate_##TYPENAME##_tile_allgather_data##SC_SUFFIX(                     \
        nvshmem_team_t team, size_t tile_start_elem_idx, src_tensor_t src, dst_tensor_t dest, \
        size_t tensor_size_major, size_t tensor_size_minor, size_t boundary_major,            \
        size_t boundary_minor) {                                                              \
        size_t flat_elem_idx;                                                                 \
        size_t idx_in_tile;                                                                   \
        int npes = nvshmem_team_n_pes(team);                                                  \
        int myIdx = nvshmtest_thread_id_in_##SC();                                            \
        int groupSize = nvshmtest_##SC##_size();                                              \
        size_t tot_elem_in_tile =                                                             \
            cuda::std::get<0>(dest.shape()) * cuda::std::get<1>(dest.shape());                \
        TYPE expected = assign<TYPE>(0);                                                      \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {                     \
            idx_in_tile = ((i % cuda::std::get<major_dim>(dest.shape())) *                    \
                           cuda::std::get<major_dim>(dest.stride())) +                        \
                          ((i / cuda::std::get<major_dim>(dest.shape())) *                    \
                           cuda::std::get<minor_dim>(dest.stride()));                         \
            flat_elem_idx = (tile_start_elem_idx + idx_in_tile);                              \
            expected = assign<TYPE>(flat_elem_idx);                                           \
                                                                                              \
            if (((flat_elem_idx % tensor_size_major) < boundary_major) &&                     \
                ((flat_elem_idx / tensor_size_major) < boundary_minor) &&                     \
                (*(dest.data() + idx_in_tile) != expected)) {                                 \
                print_err<TYPE>(*(dest.data() + idx_in_tile), expected, flat_elem_idx,        \
                                tot_elem_in_tile, team,                                       \
                                NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC));               \
                atomicAdd(&errs_d, 1);                                                        \
            }                                                                                 \
        }                                                                                     \
        nvshmtest_##SC##_sync();                                                              \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(VALIDATE_TILE_ALLGATHER_DATA, )
#undef VALIDATE_TILE_ALLGATHER_DATA

#define VALIDATE_TILE_ALLGATHER_1D_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)         \
    template <typename src_tensor_t, typename dst_tensor_t, int major_dim, int minor_dim>     \
    __device__ void validate_##TYPENAME##_tile_allgather_1D_data##SC_SUFFIX(                  \
        nvshmem_team_t team, size_t tile_start_elem_idx, src_tensor_t src, dst_tensor_t dest, \
        size_t tensor_size_major, size_t boundary_major) {                                    \
        size_t flat_elem_idx;                                                                 \
        size_t idx_in_tile;                                                                   \
        int npes = nvshmem_team_n_pes(team);                                                  \
        int myIdx = nvshmtest_thread_id_in_##SC();                                            \
        int groupSize = nvshmtest_##SC##_size();                                              \
        size_t tot_elem_in_tile = cuda::std::get<0>(dest.shape());                            \
        TYPE expected = assign<TYPE>(0);                                                      \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {                     \
            idx_in_tile = (i * cuda::std::get<major_dim>(dest.stride()));                     \
            flat_elem_idx = (tile_start_elem_idx + idx_in_tile);                              \
            expected = assign<TYPE>(flat_elem_idx);                                           \
                                                                                              \
            assert(flat_elem_idx < tensor_size_major);                                        \
            if (((flat_elem_idx) < boundary_major) &&                                         \
                (*(dest.data() + idx_in_tile) != expected)) {                                 \
                print_err<TYPE>(*(dest.data() + idx_in_tile), expected, flat_elem_idx,        \
                                tot_elem_in_tile, team,                                       \
                                NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC));               \
                atomicAdd(&errs_d, 1);                                                        \
            }                                                                                 \
        }                                                                                     \
        nvshmtest_##SC##_sync();                                                              \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(VALIDATE_TILE_ALLGATHER_1D_DATA, )
#undef VALIDATE_TILE_ALLGATHER_1D_DATA

#define DECL_RESET_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    template <typename dst_tensor_t>                                       \
    __device__ void reset_##TYPENAME##_tile_data##SC_SUFFIX(dst_tensor_t dest);
NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(DECL_RESET_TILE_DATA, )
#undef DECL_RESET_TILE_DATA

#define RESET_ALLREDUCE_TILE_DATA(OP, SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)         \
    template <typename dst_tensor_t, int major_dim, int minor_dim>                      \
    __device__ void reset_##TYPENAME##_tile_data##SC_SUFFIX(size_t tile_start_elem_idx, \
                                                            dst_tensor_t dest) {        \
        size_t idx_in_tile;                                                             \
        int myIdx = nvshmtest_thread_id_in_##SC();                                      \
        int groupSize = nvshmtest_##SC##_size();                                        \
        size_t tot_elem_in_tile =                                                       \
            cuda::std::get<0>(dest.shape()) * cuda::std::get<1>(dest.shape());          \
        for (size_t i = myIdx; i < tot_elem_in_tile; i = i + groupSize) {               \
            idx_in_tile = ((i % cuda::std::get<major_dim>(dest.shape())) *              \
                           cuda::std::get<major_dim>(dest.stride())) +                  \
                          ((i / cuda::std::get<major_dim>(dest.shape())) *              \
                           cuda::std::get<minor_dim>(dest.stride()));                   \
            *(dest.data() + idx_in_tile) = assign<TYPE>(0);                             \
        }                                                                               \
        nvshmtest_##SC##_sync();                                                        \
    }

NVSHMEMTEST_TILE_REPT_TYPES_AND_SCOPES(RESET_ALLREDUCE_TILE_DATA, )
#undef RESET_ALLREDUCE_TILE_DATA

#endif /* NVSHMEMTEST_FCOLLECT_COMMON_H */
