/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <cuda.h>
#include <algorithm>
#include <numeric>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cassert>
#include <vector>
#include <getopt.h>
#include <errno.h>

#include "utils.h"

using int64 = long long int;

struct pe_info_t {
    enum class dest_kind_t {
        ib = 0,  // should be 0 for the std::stable_sort below
        nvl = 1,
    };
    dest_kind_t kind;
    int pe;
};

constexpr size_t MSG_SIZE_NVL = 256;
constexpr size_t THREADS_PER_BLOCK = 256;

template <typename T>
T div_up(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

double refval(size_t src_pe, size_t src_i) {
    return (double)(((size_t)src_pe) * 314 + (((size_t)src_pe) % 19) * src_i);
}

__global__ void all_to_all(double *src, double *dst, int my_pe, size_t size_per_dst,
                           size_t num_pes_ib, size_t num_pes_nvl,
                           // IB message == 1 nvshmem_double_put_nbi of msg_size_ib elements
                           size_t num_blocks_ib, size_t num_msg_ib, size_t msg_size_ib,
                           // NVL message == 1 nvshmem_double_put_nbi_block of a fixed size
                           size_t num_blocks_nvl, size_t num_msg_nvl, size_t msg_size_nvl,
                           pe_info_t *info) {
    // IB path
    if ((size_t)blockIdx.x < num_blocks_ib) {
        size_t tid = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
        if (tid < num_msg_ib) {
            size_t pe_id = tid % num_pes_ib;
            size_t msg_id = tid / num_pes_ib;

            int pe_dst = info[pe_id].pe;
            // assert(info[pe_id].kind == dest_kind_t::ib)

            double *src_start = src + pe_dst * size_per_dst + msg_id * msg_size_ib;
            double *dst_start = dst + my_pe * size_per_dst + msg_id * msg_size_ib;
            nvshmem_double_put_nbi(dst_start, src_start, msg_size_ib, pe_dst);
            // printf("IB: %d (%d %d), %p -> %p (%d)\n", my_pe, threadIdx.x, blockIdx.x, src_start,
            // dst_start, pe_dst);
        }

        // NVLink path
    } else {
        size_t blockid = (size_t)blockIdx.x - num_blocks_ib;
        if (blockid < num_msg_nvl) {
            size_t pe_id = num_pes_ib + (blockid % num_pes_nvl);
            size_t msg_id = blockid / num_pes_nvl;

            int pe_dst = info[pe_id].pe;
            // assert(info[pe_id].kind == dest_kind_t::nvl)

            double *src_start = src + pe_dst * size_per_dst + msg_id * msg_size_nvl;
            double *dst_start = dst + my_pe * size_per_dst + msg_id * msg_size_nvl;
            nvshmemx_double_put_nbi_block(dst_start, src_start, msg_size_nvl, pe_dst);
            // if(threadIdx.x == 0) {
            //     printf("NV: %d (%d %d), %p -> %p (%d)\n", my_pe, threadIdx.x, blockIdx.x,
            //     src_start, dst_start, pe_dst);
            // }
        }
    }
}

static void print_help(void) {
    printf("Usage: {srun/mpirun/oshrun} ./alltoall_bw -n -N\n");
    printf(
        "  Runs a bandwidth benchmark that simulates the communication required by an FFT "
        "calculation. The only inputs are the minimum and maximum dimension size.\n");
    printf(
        "  You can imagine the data shared between nodes in an FFT as a 3-dimensional array with "
        "all dimensions having equal size (N).\n");
    printf(
        "  with knowledge of the FFT dimension (N) and number of GPUs (G), the number of messages "
        "sent by each GPU to each other (M) and the size of each message (S) can be calculated as "
        "follows:\n");
    printf(
        "  M = N/G, S = (Type_Size) * N^2 / G. Total Symmetric Heap requirement is equal to "
        "(Type_size) * N^3 / G.");
    printf("  Note: All *_LOG arguments should be entered as decimals from 0-63.");
    printf(
        "  -n FFT_DIM_MIN_LOG is the start of the interval of the one-dimensional FFT slice size, "
        "represented as the integer log2 of a power of 2\n");
    printf(
        "  -N FFT_DIM_MAX_LOG is the end of the interval of the one-dimensional FFT slice size, "
        "represented as the integer log2 of a power of 2\n");
}

int main(int argc, char *argv[]) {
    int ndevices;
    int mype, npes;
    int status = 0;

    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    cudaStream_t stream;

    std::vector<double> host_input;
    std::vector<double> host_output;
    std::vector<pe_info_t> host_info;

    pe_info_t *info = NULL;
    double *src = NULL;
    double *dst = NULL;

    size_t num_pes_ib = 0;
    size_t num_pes_nvl = 0;
    size_t size_max = 0;
    size_t array_dim_min_log = 1;
    size_t array_dim_max_log = 0;
    size_t array_dim_min = 1;
    size_t array_dim_max = 0;
    size_t warmup = 0;
    size_t repeat = 1;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    while (1) {
        int c;
        c = getopt(argc, argv, "n:N:w:r:h");
        if (c == -1) {
            break;
        }

        switch (c) {
            case 'n':
                array_dim_min_log = strtol(optarg, NULL, 0);
                if (array_dim_min_log > 63) {
                    print_help();
                    status = -EINVAL;
                    goto out;
                }
                array_dim_min = (1 << array_dim_min_log);
                break;
            case 'N':
                array_dim_max_log = strtol(optarg, NULL, 0);
                if (array_dim_max_log > 63) {
                    print_help();
                    status = -EINVAL;
                    goto out;
                }
                array_dim_max = (1 << array_dim_max_log);
                break;
            case 'w':
                warmup = strtol(optarg, NULL, 0);
                break;
            case 'r':
                repeat = strtol(optarg, NULL, 0);
                break;
            case 'h':
            default:
                print_help();
                goto out;
                break;
        }
    }

    if (array_dim_min < (size_t)npes) {
        fprintf(stderr,
                "array_dim_min_log is too small. With %d gpus, the minimum dimension size must be "
                "at least log2(%d)\n",
                npes, npes);
        status = -EINVAL;
        goto out;
    }

    assert(array_dim_max >= array_dim_min);
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));

    if (mype == 0) {
        printf(
            "all_to_all_rem: num_pes = %d, array_dim_min = %lu, array_dim_max = %lu, skip = %zu, "
            "repeat = %zu\n",
            npes, array_dim_min, array_dim_max, warmup, repeat);
    }

    /* Allocate the src and dest array. */
    if ((array_dim_max_log * 3) > 63) {
        fprintf(stderr,
                "array_dim_max_log (%lu) is too large and would result in overflow. Exiting.\n",
                array_dim_max_log);
        status = -EINVAL;
        goto out;
    }
    size_max = 1LLU << (array_dim_max_log * 3);
    size_max = size_max / npes;
    if (size_max > (SIZE_MAX / sizeof(double))) {
        fprintf(stderr,
                "array_dim_max_log (%lu) is too large and would result in overflow. Exiting.\n",
                array_dim_max_log);
        status = -EINVAL;
        goto out;
    }

    host_input.insert(host_input.begin(), size_max, 0.0);
    host_output.insert(host_output.begin(), size_max, 0.0);
    for (size_t i = 0; i < size_max; i++) {
        host_input[i] = refval(mype, i);
        // printf("in[%zu] = %f\n", i, host_input[i]);
    }

    src = (double *)nvshmem_malloc(size_max * sizeof(double));
    dst = (double *)nvshmem_malloc(size_max * sizeof(double));

    // Check who is accessible over remtoe transports vs NVLINK
    // Create a list of npes "info" struct, with IB first and NVLink second
    for (int pe = 0; pe < npes; pe++) {
        if (std::getenv("NVSHMEMTEST_ALL_IB")) {
            host_info.push_back({pe_info_t::dest_kind_t::ib, pe});
            num_pes_ib++;
        } else if (std::getenv("NVSHMEMTEST_RANDOMISH")) {
            if (pe % 2 == 0) {
                host_info.push_back({pe_info_t::dest_kind_t::ib, pe});
                num_pes_ib++;
            } else {
                host_info.push_back({pe_info_t::dest_kind_t::nvl, pe});
                num_pes_nvl++;
            }
        } else {
            if (nvshmem_ptr(src, pe) == nullptr) {
                host_info.push_back({pe_info_t::dest_kind_t::ib, pe});
                num_pes_ib++;
            } else {
                host_info.push_back({pe_info_t::dest_kind_t::nvl, pe});
                num_pes_nvl++;
            }
        }
    }
    std::stable_sort(host_info.begin(), host_info.end(),
                     [](const pe_info_t &lhs, const pe_info_t &rhs) {
                         return static_cast<int>(lhs.kind) < static_cast<int>(rhs.kind);
                     });
    if (mype == 0) {
        printf("num_pes_ib %zu, num_pes_nvl %zu\n", num_pes_ib, num_pes_nvl);
        for (size_t i = 0; i < host_info.size(); i++) {
            printf("info[%zu] = {%s, %d}\n", i,
                   host_info[i].kind == pe_info_t::dest_kind_t::ib ? "ib" : "nvl", host_info[i].pe);
        }
    }

    CUDA_CHECK(cudaMalloc(&info, sizeof(host_info[0]) * host_info.size()));
    CUDA_CHECK(cudaMemcpy(info, host_info.data(), sizeof(host_info[0]) * host_info.size(),
                          cudaMemcpyDefault));

    // Create cuda stuff
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Iterate over sizes
    if (mype == 0)
        printf(
            "         array_dim,         errors,        checked, "
            "         num_B,  av_time_ms, av_bw_GB_s_pe,       num_B_ib, av_bw_ib_GB_s_pe\n");
    for (size_t array_dim = array_dim_min; array_dim <= array_dim_max; array_dim *= 2) {
        // This models a cuFFTMp all-to-all
        // See
        // https://docs.google.com/document/d/1oSzxXAhHvLks5nLy-t2U-5USp_y-dAGl-tNjoulEWgE/edit#bookmark=id.bnqukyk7rw56
        // array_dim is the FFT size "N" and npes is "G". Let Gib the number of PEs accessible over
        // IB, and Gnvl the number of PEs accessible over NVLink.

        // Global data is an array_dim^3 tensor, with X (slow), Y, Z (fast) the three dimensions
        // Data is initially distributed along X and, after the all-to-all, it is split along Y

        // Communication is performed differently over peer-to-peer (aka NVLink) or IB
        // Over peer-to-peer, we use 1 thread per element and nvshmemx_<type>_put_nbi_block
        // Over IB, we use 1 thread per message, where message is the longuest contiguous buffer at
        // both the source and destination. Each message is sent using nvshmem_<type>_put_nbi.

        // Reset device memory
        CUDA_CHECK(cudaMemset(src, 0, size_max * sizeof(double)));
        CUDA_CHECK(cudaMemset(dst, 0, size_max * sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());  // cudaMemset is async

        // Total size per PE is N^3 / G
        const size_t size = array_dim * array_dim * array_dim / npes;

        // Total amount exchanged between pairs of PEs is N^3 / G^2.
        const size_t size_per_dst = size / npes;

        if (size % npes != 0 || array_dim % npes != 0) {
            fprintf(stderr, "Number of PEs should be a power of 2\n");
            status = -EINVAL;
            goto out;
        }

        // Input -> src
        cudaMemcpy(src, host_input.data(), size * sizeof(double), cudaMemcpyDefault);

        // Figure out arguments

        // Over IB, message sizes (== "size" of an NVSHMEM API call) are N^2 / G
        // We have Gib * (N / G) to send
        size_t msg_size_ib = array_dim * (array_dim / npes);
        size_t num_msg_ib = num_pes_ib * (array_dim / npes);

        // Over NVLink, message size (== "size" of an NVSHMEM API call) is constant
        size_t msg_size_nvl = std::min(MSG_SIZE_NVL, msg_size_ib);
        size_t num_msg_nvl = (num_pes_nvl * size_per_dst) / msg_size_nvl;

        if (num_msg_nvl * msg_size_nvl + num_msg_ib * msg_size_ib != size) {
            fprintf(stderr, "Everthing should be divisible...\n");
            status = -EINVAL;
            goto out;
        }

        // 1 thread sends one message over IB
        // 1 block sends one message over NVLink
        size_t num_blocks_ib = div_up(num_msg_ib, THREADS_PER_BLOCK);
        size_t num_blocks_nvl = num_msg_nvl;
        size_t num_blocks = num_blocks_ib + num_blocks_nvl;

        // printf("size_per_dst %zu num_pes_ib %zu num_pes_nvl %zu num_blocks_ib %zu num_msg_ib %zu
        // msg_size_ib %zu num_blocks_nvl %zu num_msg_nvl %zu msg_size_nvl %zu\n", size_per_dst,
        // num_pes_ib, num_pes_nvl, num_blocks_ib, num_msg_ib, msg_size_ib, num_blocks_nvl,
        // num_msg_nvl, msg_size_nvl);

        // Skip first ones
        nvshmem_barrier_all();
        for (size_t i = 0; i < warmup; i++) {
            all_to_all<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                src, dst, mype, size_per_dst, num_pes_ib, num_pes_nvl, num_blocks_ib, num_msg_ib,
                msg_size_ib, num_blocks_nvl, num_msg_nvl, msg_size_nvl, info);
            nvshmemx_barrier_all_on_stream(stream);
        }

        // Run and time
        cudaEventRecord(start);
        for (size_t i = 0; i < repeat; i++) {
            all_to_all<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                src, dst, mype, size_per_dst, num_pes_ib, num_pes_nvl, num_blocks_ib, num_msg_ib,
                msg_size_ib, num_blocks_nvl, num_msg_nvl, msg_size_nvl, info);
            nvshmemx_barrier_all_on_stream(stream);
        }
        cudaEventRecord(stop);

        // Measure and display time
        cudaEventSynchronize(stop);
        cudaStreamSynchronize(stream);
        float time_ms = 0;
        cudaEventElapsedTime(&time_ms, start, stop);
        float average_time_ms = time_ms / repeat;
        size_t num_B = (array_dim * array_dim * array_dim) / npes * sizeof(double);
        float average_bw_GB_s = (num_B / 1e9) / (average_time_ms * 1e-3);

        // Assume NVLink time is negligeable
        size_t num_B_ib = (num_B / npes) * num_pes_ib;
        float average_bw_ib_GB_s = (num_B_ib / 1e9) / (average_time_ms * 1e-3);

        // Check correctness
        size_t errors = 0;
        size_t checked = 0;
        if (!std::getenv("NVSHMEMTEST_SKIP_CHECK")) {
            cudaMemcpy(host_output.data(), dst, size * sizeof(double), cudaMemcpyDefault);
            for (size_t i = 0; i < size; i += 1) {
                // printf("out[%zu] = %f\n", i, host_output[i]);
                size_t src_pe = i / size_per_dst;
                size_t src_i = mype * size_per_dst + i % size_per_dst;
                double ref = refval(src_pe, src_i);
                if (ref != host_output[i]) {
                    printf("pe %d expected %f got %f at i %zu\n", mype, ref, host_output[i], i);
                    errors++;
                }
                checked++;
            }
        }

        if (mype == 0)
            printf("    %14zu, %14zu, %14zu, %14zu,   %5.3e,     %5.3e, %14zu,        %5.3e\n",
                   array_dim, errors, checked, num_B, average_time_ms, average_bw_GB_s, num_B_ib,
                   average_bw_ib_GB_s);

        if (errors != 0) {
            fprintf(stderr, "ERROR in the alltoall\n");
            status = -1;
            goto out;
        }
    }

out:
    if (info) {
        cudaFree(info);
    }
    if (stream) {
        cudaStreamDestroy(stream);
    }
    if (start) {
        cudaEventDestroy(start);
    }
    if (stop) {
        cudaEventDestroy(stop);
    }

    if (src) {
        nvshmem_free(src);
    }
    if (dst) {
        nvshmem_free(dst);
    }

    finalize_wrapper();
    return status;
}
