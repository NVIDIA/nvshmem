/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include "utils.h"
#include "tile_coll_test.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_tensor.h"
#define LARGEST_DT int64_t

#define VLEN 4

size_t round_up(int val, int gran) { return ((val + gran - 1) / gran) * gran; }

#define CALL_TILE_ALLGATHER(TG_PRE, TG, TYPENAME, TYPE, THREAD_COMP, ELEM_COMP)              \
                                                                                             \
    template <nvshmemx::tile_coll_algo_t algo, typename src_tensor_t, typename dst_tensor_t> \
    __global__ void test_tile_##TYPENAME##_allgather_kern##TG(                               \
        nvshmem_team_t team, src_tensor_t *src, dst_tensor_t *dst, int nelems, int iter) {   \
        int i;                                                                               \
        struct empty {};                                                                     \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {            \
            for (i = 0; i < iter; i++) {                                                     \
                nvshmemx::tile_allgather##TG<src_tensor_t, dst_tensor_t, empty, algo>(       \
                    team, *src, *dst, {}, {}, 0);                                            \
                nvshmemx::tile_collective_wait##TG<algo>(team, 0);                           \
            }                                                                                \
        }                                                                                    \
    }

#define CALL_TILE_AG_KERNEL(TYPENAME, TG, BLOCKS, THREADS, ALGO, TEAM, SRC_TENSOR, DST_TENSOR, \
                            STREAM, ITER)                                                      \
    test_tile_##TYPENAME##_allgather_kern##TG<ALGO, src_tensor_t, dst_tensor_t>                \
        <<<BLOCKS, THREADS, 0, STREAM>>>(TEAM, SRC_TENSOR, DST_TENSOR, num_elems, ITER);

#define CALL_TILE_ALLGATHER_ALL_TG(TYPENAME, TYPE)                   \
    CALL_TILE_ALLGATHER(x, _block, TYPENAME, TYPE, INT_MAX, INT_MAX) \
    CALL_TILE_ALLGATHER(x, _warpgroup, TYPENAME, TYPE, 128, 65536)   \
    CALL_TILE_ALLGATHER(x, _warp, TYPENAME, TYPE, warpSize, 4096)    \
    CALL_TILE_ALLGATHER(, , TYPENAME, TYPE, 1, 512)

CALL_TILE_ALLGATHER_ALL_TG(half, half)
CALL_TILE_ALLGATHER_ALL_TG(float, float)

#define SET_SIZE_ARR(TYPE, ELEM_COMP)                                                   \
    do {                                                                                \
        j = 0;                                                                          \
        for (num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) { \
            if (num_elems < ELEM_COMP) {                                                \
                size_arr[j] = num_elems * sizeof(TYPE);                                 \
            } else {                                                                    \
                size_arr[j] = 0;                                                        \
            }                                                                           \
            j++;                                                                        \
        }                                                                               \
    } while (0)

#define RUN_ITERS(TYPENAME, TYPE, GROUP, ELEM_COMP, ALGO)                                       \
    do {                                                                                        \
        float milliseconds;                                                                     \
        cudaEvent_t start, stop;                                                                \
        cudaEventCreate(&start);                                                                \
        cudaEventCreate(&stop);                                                                 \
        SET_SIZE_ARR(TYPE, ELEM_COMP);                                                          \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
        j = 0;                                                                                  \
        for (num_elems = min_elems; num_elems < ELEM_COMP; num_elems *= step_factor) {          \
            /* we keep shape and stride along dim 0 same, while increase size along dim 1 */    \
            /* init team and tensors */                                                         \
            /*nvshmem_team_t team = NVSHMEM_TEAM_WORLD;*/                                       \
            constexpr int tsize_0 = (VLEN * sizeof(uint32_t)) / sizeof(TYPE);                   \
            auto src_tile_shape = nvshmemx::make_shape<ConstInt<tsize_0>, int>(                 \
                ConstInt<tsize_0>{}, num_elems / tsize_0);                                      \
            auto dst_tile_shape = nvshmemx::make_shape<ConstInt<tsize_0>, int>(                 \
                ConstInt<tsize_0>{}, (npes * num_elems) / tsize_0);                             \
            /* keeping src and dst with same stride*/                                           \
            auto tile_stride = nvshmemx::make_stride<ConstInt<1>, int>(ConstInt<1>{}, tsize_0); \
            auto src_tile_layout = nvshmemx::make_layout(src_tile_shape, tile_stride);          \
            auto dst_tile_layout = nvshmemx::make_layout(dst_tile_shape, tile_stride);          \
            auto src_tensor = nvshmemx::Tensor<TYPE, decltype(src_tile_layout)>(                \
                reinterpret_cast<TYPE *>(source), src_tile_layout);                             \
            auto dst_tensor = nvshmemx::Tensor<TYPE, decltype(dst_tile_layout)>(                \
                reinterpret_cast<TYPE *>(dest), dst_tile_layout);                               \
                                                                                                \
            using src_tensor_t = decltype(src_tensor);                                          \
            using dst_tensor_t = decltype(dst_tensor);                                          \
            src_tensor_t *src_tensor_dev;                                                       \
            dst_tensor_t *dst_tensor_dev;                                                       \
            CUDA_CHECK(cudaMalloc((void **)&src_tensor_dev, sizeof(src_tensor_t)));             \
            CUDA_CHECK(cudaMalloc((void **)&dst_tensor_dev, sizeof(dst_tensor_t)));             \
            CUDA_CHECK(cudaMemcpy(src_tensor_dev, &src_tensor, sizeof(src_tensor_t),            \
                                  cudaMemcpyHostToDevice));                                     \
            CUDA_CHECK(cudaMemcpy(dst_tensor_dev, &dst_tensor, sizeof(dst_tensor_t),            \
                                  cudaMemcpyHostToDevice));                                     \
                                                                                                \
            CALL_TILE_AG_KERNEL(TYPENAME, GROUP, num_blocks, nvshm_test_num_tpb, ALGO, team,    \
                                src_tensor_dev, dst_tensor_dev, stream, skip);                  \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                          \
            nvshmem_barrier_all();                                                              \
                                                                                                \
            cudaEventRecord(start, stream);                                                     \
            CALL_TILE_AG_KERNEL(TYPENAME, GROUP, num_blocks, nvshm_test_num_tpb, ALGO, team,    \
                                src_tensor_dev, dst_tensor_dev, stream, iter);                  \
            cudaEventRecord(stop, stream);                                                      \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                          \
                                                                                                \
            if (!mype) {                                                                        \
                cudaEventElapsedTime(&milliseconds, start, stop);                               \
                h_lat[j] = (milliseconds * 1000.0) / (float)iter;                               \
            }                                                                                   \
            nvshmem_barrier_all();                                                              \
            j++;                                                                                \
        }                                                                                       \
    } while (0)

int tile_AG_calling_kernel(nvshmem_team_t team, void *dest, void *source, int mype, int npes,
                           cudaStream_t stream, run_opt_t run_options, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = threads_per_block;
    int num_blocks = 1;
    size_t num_elems = 1, min_elems, max_elems;
    int iter = iters;
    int skip = warmup_iters;
    int j = 0;
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_lat = (double *)h_tables[1];
    // operations of reduction API in us\n");

    //  ensuring v4 collective variant gets used
    if (run_options.run_thread) {
        min_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        round_up(min_size / sizeof(half), VLEN));
        max_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        max_size / sizeof(half));
        RUN_ITERS(half, half, , 512, nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "fp16-AG-t", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }

        min_elems = max(static_cast<size_t>(VLEN), round_up(min_size / sizeof(float), VLEN));
        max_elems = max(static_cast<size_t>(VLEN), max_size / sizeof(float));
        RUN_ITERS(float, float, , 512, nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "float-AG-t", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }
    }

    if (run_options.run_warp) {
        min_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        round_up(min_size / sizeof(half), VLEN));
        max_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        max_size / sizeof(half));
        RUN_ITERS(half, half, _warp, 4096, nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "fp16-AG-w", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }

        min_elems = max(static_cast<size_t>(VLEN), round_up(min_size / sizeof(float), VLEN));
        max_elems = max(static_cast<size_t>(VLEN), max_size / sizeof(float));
        RUN_ITERS(float, float, _warp, 4096, nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "float-AG-w", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }
    }

    if (run_options.run_warpgroup) {
        min_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        round_up(min_size / sizeof(half), VLEN));
        max_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        max_size / sizeof(half));
        RUN_ITERS(half, half, _warpgroup, 65536,
                  nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "fp16-AG-g", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }

        min_elems = max(static_cast<size_t>(VLEN), round_up(min_size / sizeof(float), VLEN));
        max_elems = max(static_cast<size_t>(VLEN), max_size / sizeof(float));
        RUN_ITERS(float, float, _warpgroup, 65536,
                  nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "float-AG-g", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }
    }

    if (run_options.run_block) {
        min_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        round_up(min_size / sizeof(half), VLEN));
        max_elems = max(static_cast<size_t>((VLEN * sizeof(uint32_t)) / sizeof(half)),
                        max_size / sizeof(half));
        RUN_ITERS(half, half, _block, max_elems,
                  nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "fp16-AG-b", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }

        min_elems = max(static_cast<size_t>(VLEN), round_up(min_size / sizeof(float), VLEN));
        max_elems = max(static_cast<size_t>(VLEN), max_size / sizeof(float));
        RUN_ITERS(float, float, _block, max_elems,
                  nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI);
        if (!mype) {
            print_table_v1("fcollect_device", "float-AG-b", "size (Bytes)", "latency", "us", '-',
                           size_arr, h_lat, j);
        }
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes, array_size;
    size_t size = 0;

    read_args(argc, argv);
    int *h_buffer = NULL;
    int *d_source, *d_dest;
    int *h_source, *h_dest;
    char size_string[100];
    cudaStream_t cstrm;
    run_opt_t run_options;
    void **h_tables;

    run_options.run_thread = run_options.run_warp = run_options.run_warpgroup =
        run_options.run_block = 1;

    size = page_size_roundoff(max_size);       // send buf
    size += page_size_roundoff(max_size * 8);  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }
    array_size = max_size_log;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 2, array_size);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    CUDA_CHECK(cudaHostAlloc(&h_buffer, max_size * (npes + 1), cudaHostAllocDefault));
    h_source = (int32_t *)h_buffer;
    h_dest = (int32_t *)&h_source[max_size / sizeof(int32_t)];

    d_source = (int32_t *)nvshmem_align(getpagesize(), max_size);
    d_dest = (int32_t *)nvshmem_align(getpagesize(), max_size * npes);

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, max_size, cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, max_size * npes, cudaMemcpyHostToDevice, cstrm));

    tile_AG_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, npes, cstrm, run_options,
                           h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, max_size, cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, max_size * npes, cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
