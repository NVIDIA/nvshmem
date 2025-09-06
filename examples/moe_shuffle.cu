/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include <algorithm>
#include <chrono>
#include <functional>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"

/* Types */
typedef enum {
    MOE_BARE_TWO_STEP_ALLPUSH = 0x0,
    MOE_COLL_LAUNCH_TWO_STEP_ALLPUSH = 0x1,
    MOE_BARE_TWO_STEP_BY_PEER = 0x2,
    MOE_COLL_LAUNCH_TWO_STEP_BY_PEER = 0x3,
    MOE_ONE_STEP_LAUNCH_MIN = 0x4,
    MOE_BARE_ONE_STEP_ALLPUSH = 0x4,
    MOE_COLL_LAUNCH_ONE_STEP_ALLPUSH = 0x5,
    MOE_BARE_ONE_STEP_BY_PEER = 0x6,
    MOE_COLL_LAUNCH_ONE_STEP_BY_PEER = 0x7
} moe_comms_profile_t;

/* Constants */
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

constexpr int num_experts = 16;

/* Start of initialization helpers */
void set_comms_profile(int *profile, bool do_by_peer, bool do_one_step, bool do_collective_launch) {
    if (do_by_peer) {
        *profile = MOE_BARE_TWO_STEP_BY_PEER;
    }

    *profile += do_collective_launch;
    *profile += do_one_step ? MOE_ONE_STEP_LAUNCH_MIN : 0;
}

int validate_input_params(int *num_blocks, int num_baseline_blocks, int threads_per_block,
                          int num_rows, int num_elems) {
    int rc = 0;
    int num_warps_per_block = threads_per_block / 32;
    int num_elems_per_row = num_rows / num_elems;

    *num_blocks = std::min(num_baseline_blocks, num_rows);  // best for 16 ranks
    if (*num_blocks < num_baseline_blocks) {
        std::cout << "number of blocks requested (" << num_baseline_blocks << ")"
                  << "is greater than the number of rows. (" << num_rows
                  << ") decreasing block count to " << num_blocks << "\n";
    }

    /* for bench simplicity */
    if (num_elems % threads_per_block) {
        std::cout << "num_elems (" << num_elems << ") is not evenly divisible by "
                  << "threads_per_block (" << threads_per_block << "). Cannot continue\n";
        rc = -1;
    }

    if (threads_per_block % 32) {
        std::cout << "num_threads_per_block (" << threads_per_block
                  << ") is not evenly divisible by num_threads_per_warp"
                  << " (32). Cannot continue\n";
        rc = -1;
    }

    if (num_elems_per_row % num_warps_per_block) {
        std::cout << "num_elems_per_row (" << num_elems_per_row
                  << ") is not evenly divisible by num_warps_per_block (" << num_warps_per_block
                  << "). Cannot continue.\n";
    }

    /* for bench simplicity */
    if (num_rows % *num_blocks) {
        std::cout << "num_rows (" << num_rows << ") is not evenly divisible by num_blocks ("
                  << *num_blocks << "). Cannot continue\n";
        rc = -1;
    }

    return rc;
}

void usage(void) {
    std::cout
        << "USAGE -b [number of blocks] -t [threads per block] -c -h -l -o -p -r -v\n"
        << "-c use cudaMalloc to instantiate expert to row mappings rather than malloc\n"
        << "-h display this help message\n"
        << "-l use nvshmemx_collective_launch to launch the MoE alltoallv kernel\n"
        << "-o perform the allgather of offsets in the same kernel as the alltoallv\n"
        << "-p use a peer based communication pattern rather than writing to all peers at once\n"
        << "-r randomize the selection of experts per rows, creating an uneven alltoall pattern\n"
        << "-v display verbose output about the selected parameters\n";
}

int setup_test_parameters(int argc, char **argv, int *num_blocks, int *threads_per_block,
                          bool *use_cuda_malloc, bool *randomize_output, bool *warp_specialized,
                          moe_comms_profile_t *profile, int num_rows, int num_elems) {
    int num_baseline_blocks = 96;
    int rc = 0;
    bool do_by_peer = 0;
    bool do_one_step = 0;
    bool do_collective_launch = 0;
    bool verbose = 0;

    while (1) {
        int c;
        c = getopt(argc, argv, "b:t:chloprvw");
        if (c == -1) break;

        switch (c) {
            case 'b':
                num_baseline_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                *threads_per_block = strtol(optarg, NULL, 0);
                break;
            case 'c':
                *use_cuda_malloc = true;
                break;
            case 'l':
                do_collective_launch = true;
                break;
            case 'o':
                do_one_step = true;
                break;
            case 'p':
                do_by_peer = true;
                break;
            case 'r':
                *randomize_output = true;
                break;
            case 'v':
                verbose = true;
            case 'w':
                *warp_specialized = true;
                break;
            default:
                std::cout << "Received unknown argument: -" << c
                          << " Displaying help and exiting\n";
            case 'h':
                usage();
                rc = -1;
                goto finalize;
        }
    }

    set_comms_profile((int *)profile, do_by_peer, do_one_step, do_collective_launch);
    rc = validate_input_params(num_blocks, num_baseline_blocks, *threads_per_block, num_rows,
                               num_elems);
    if (rc) {
        goto finalize;
    }

    if (verbose) {
        std::cout
            << "performing moe comm pattern simulation with the following parameters: \n"
            << "number of blocks:                  " << *num_blocks << "\n"
            << "number of threads per block:       " << *threads_per_block << "\n"
            << "comms pattern:                     "
            << (do_by_peer ? "by peer                   " : "one shot                  ") << "\n"
            << "expert selection pattern:          "
            << (*randomize_output ? "randomized                " : "static                    ")
            << "\n"
            << "expert map allocation strategy:    "
            << (*use_cuda_malloc ? "cudaMalloc               " : "malloc                   ")
            << "\n"
            << "operation compostion:              "
            << (do_one_step ? "single kernel            " : "split kernels            ") << "\n"
            << "alltoall launch strategy:          "
            << (do_collective_launch ? "nvshmem collective launch" : "direct launch            ")
            << "\n"
            << "warp specialization strategy:      "
            << (*warp_specialized ? "warp APIs                " : "block APIs               ")
            << "\n";
    }

finalize:
    return rc;
}
/* End of initialization helpers */

/* Start of offset exchange code */
static __forceinline__ __device__ void _exchange_offsets(int64_t *local_expert_counts,
                                                         int64_t *symmetric_expert_counts,
                                                         int64_t *accumulated_expert_positions,
                                                         int npes) {
    const int src_rank = threadIdx.x / npes;
    const int expert = threadIdx.x % npes;
    const int num_experts_per_rank = num_experts / npes;
    const int base = npes * num_experts_per_rank * threadIdx.x;

    int64_t prev = 0;

    // get counts from every node for each expert
    if (threadIdx.x < npes * num_experts) {
        local_expert_counts[threadIdx.x] =
            nvshmem_int64_g((int64_t *)symmetric_expert_counts + expert, src_rank);
    }
    __syncthreads();

    if (threadIdx.x < npes) {
#pragma unroll 4
        for (int i = 0; i < npes * num_experts_per_rank; ++i) {
            prev += local_expert_counts[base + i];
            accumulated_expert_positions[base + i] = prev;
        }
    }
}

__global__ void exchange_offsets(int64_t *expert_counts, int64_t *expert_pos_out, int npes) {
    extern __shared__ int64_t expert_count_in[];
    _exchange_offsets(expert_count_in, expert_counts, expert_pos_out, npes);
}

/* End of offset exchange code */

/* Start of alltoall code */
/* Start of allpush code */
template <bool WARP_SCOPED>
static __forceinline__ __device__ void _token_shuffle_allpush(
    float *send_data, float *recv_data, int *src_rows, int *src_experts, int64_t *expert_offsets,
    int k, int num_rows, int mype, int npes, int num_copies_per_block, int hidden_dim,
    int64_t *expert_pos_out) {
    const int num_experts_per_rank = num_experts / npes;
    const int num_warps_per_block = (blockDim.x * blockDim.y * blockDim.z) / 32;
    const int num_elems_per_warp = hidden_dim / num_warps_per_block;
    const int my_warp_idx = threadIdx.x / 32;

    int block_offset = blockIdx.x * num_copies_per_block;

    for (int i = 0; i < num_copies_per_block; ++i) {
        if (block_offset >= num_rows) {
            return;
        }
        /*
         * Copies one token (row)
         * All threads in the block call API with same arguments
         */
        auto src_row = src_rows[block_offset] % (num_rows / k);
        auto expert = src_experts[block_offset];
        auto peer = expert / num_experts_per_rank;
        bool first_expert_in_rank = expert % num_experts_per_rank == 0;

        // expert position on peer for this rank on the destination rank
        auto expert_start =
            (mype == 0 && first_expert_in_rank) ? 0 : expert_pos_out[mype + expert * npes - 1];
        // relative position in expert
        auto pos_in_expert = block_offset - (expert > 0 ? expert_offsets[expert - 1] : 0);

        /*
        if (threadIdx.x == 0) {
            printf("%3d %3d %3d %3ld %3ld \n", peer, expert, mype, expert_start, pos_in_expert);
        }
        */
        if (WARP_SCOPED) {
            nvshmemx_float_put_nbi_warp(
                recv_data + (expert_start + pos_in_expert) * hidden_dim +
                    my_warp_idx * num_elems_per_warp,
                send_data + src_row * hidden_dim + my_warp_idx * num_elems_per_warp,
                num_elems_per_warp, peer);
        } else {
            nvshmemx_float_put_nbi_block(recv_data + (expert_start + pos_in_expert) * hidden_dim,
                                         send_data + src_row * hidden_dim, hidden_dim, peer);
        }

        block_offset += 1;
    }
}

template <bool WARP_SCOPED>
__global__ void token_shuffle_two_step_allpush(float *send_data, float *recv_data, int *src_rows,
                                               int *src_experts, int64_t *expert_offsets, int k,
                                               int num_rows, int mype, int npes, int hidden_dim,
                                               int num_copies_per_block, int64_t *expert_pos_out) {
    _token_shuffle_allpush<WARP_SCOPED>(send_data, recv_data, src_rows, src_experts, expert_offsets,
                                        k, num_rows, mype, npes, num_copies_per_block, hidden_dim,
                                        expert_pos_out);
}

template <bool WARP_SCOPED>
__global__ void token_shuffle_one_step_allpush(float *send_data, float *recv_data, int *src_rows,
                                               int *src_experts, int64_t *expert_offsets, int k,
                                               int num_rows, int mype, int npes, int hidden_dim,
                                               int num_copies_per_block, int64_t *expert_counts) {
    extern __shared__ int64_t expert_count_positions[];

    _exchange_offsets(expert_count_positions, (expert_counts + num_experts * blockIdx.x),
                      &expert_count_positions[npes * num_experts], npes);
    __syncthreads();
    _token_shuffle_allpush<WARP_SCOPED>(send_data, recv_data, src_rows, src_experts, expert_offsets,
                                        k, num_rows, mype, npes, num_copies_per_block, hidden_dim,
                                        &expert_count_positions[npes * num_experts]);
}
/* End of allpush code */

/* Start of py peer code */
template <bool WARP_SCOPED>
static __forceinline__ __device__ void _token_shuffle_by_peer(
    float *send_data, float *recv_data, int *src_rows, int *src_experts, int64_t *expert_offsets,
    int k, int num_rows, int mype, int npes, int hidden_dim, int64_t *expert_positions) {
    const int num_experts_per_rank = num_experts / npes;
    const int num_warps_per_block = (blockDim.x * blockDim.y * blockDim.z) / 32;
    const int num_elems_per_warp = hidden_dim / num_warps_per_block;
    const int my_warp_idx = threadIdx.x / 32;

    int block_offset = blockIdx.x;
    int num_blocks = gridDim.x * gridDim.y * gridDim.z;
    int rows_per_expert = num_rows / num_experts;
    int true_block_offset = rows_per_expert * mype * num_experts_per_rank + block_offset;
    int rounded_block_offset = true_block_offset % num_rows;

    for (; block_offset < num_rows; block_offset += num_blocks) {
        if (block_offset >= num_rows) {
            return;
        }
        /*
         * Copies one token (row)
         * All threads in the block call API with same arguments
         */
        auto src_row = src_rows[rounded_block_offset] % (num_rows / k);
        auto expert = src_experts[rounded_block_offset];
        auto peer = expert / num_experts_per_rank;
        bool first_expert_in_rank = expert % num_experts_per_rank == 0;

        // expert position on peer for this rank on the destination rank
        auto expert_start =
            (mype == 0 && first_expert_in_rank) ? 0 : expert_positions[mype + expert * npes - 1];
        // relative position in expert
        auto pos_in_expert = rounded_block_offset - (expert > 0 ? expert_offsets[expert - 1] : 0);

        /* if (threadIdx.x == 0) {
            printf("mype: %3d block: %3d peer: %3d expert: %3d expert_start: %3ld pos_in_expert:
        %3ld \n", mype, blockIdx.x, peer, expert, expert_start, pos_in_expert);
        } */

        if (WARP_SCOPED) {
            nvshmemx_float_put_nbi_warp(
                recv_data + (expert_start + pos_in_expert) * hidden_dim +
                    my_warp_idx * num_elems_per_warp,
                send_data + src_row * hidden_dim + my_warp_idx * num_elems_per_warp,
                num_elems_per_warp, peer);
        } else {
            nvshmemx_float_put_nbi_block(recv_data + (expert_start + pos_in_expert) * hidden_dim,
                                         send_data + src_row * hidden_dim, hidden_dim, peer);
        }
        true_block_offset += num_blocks;
        rounded_block_offset = true_block_offset % num_rows;
    }
}

template <bool WARP_SCOPED>
__global__ void token_shuffle_two_step_by_peer(float *send_data, float *recv_data, int *src_rows,
                                               int *src_experts, int64_t *expert_offsets, int k,
                                               int num_rows, int mype, int npes, int hidden_dim,
                                               int64_t *expert_pos_out) {
    _token_shuffle_by_peer<WARP_SCOPED>(send_data, recv_data, src_rows, src_experts, expert_offsets,
                                        k, num_rows, mype, npes, hidden_dim, expert_pos_out);
}

template <bool WARP_SCOPED>
__global__ void token_shuffle_one_step_by_peer(float *send_data, float *recv_data, int *src_rows,
                                               int *src_experts, int64_t *expert_offsets, int k,
                                               int num_rows, int mype, int npes, int hidden_dim,
                                               int64_t *expert_counts) {
    extern __shared__ int64_t expert_count_positions[];

    _exchange_offsets(expert_count_positions, (expert_counts + num_experts * blockIdx.x),
                      &expert_count_positions[npes * num_experts], npes);
    __syncthreads();
    _token_shuffle_by_peer<WARP_SCOPED>(send_data, recv_data, src_rows, src_experts, expert_offsets,
                                        k, num_rows, mype, npes, hidden_dim,
                                        &expert_count_positions[npes * num_experts]);
}
/* End of py peer code */
/* End of alltoall code */

/* helper kernel for setting initial values */
__global__ void set_counts(float *pointer, int base, int hidden_dim) {
    int row = blockIdx.x;
    for (int col = threadIdx.x; col < hidden_dim; col += hidden_dim / blockDim.x) {
        pointer[row * hidden_dim + col] = row + base * 1000 + (float)col / hidden_dim;
    }
}

int main(int argc, char *argv[]) {
    constexpr int iterations = 50;     // number of iterations to time
    constexpr int num_src_rows = 384;  // local rows to start
    constexpr int expert_factor = 2;   // max input elements - unsafe

    int k = 2;                              // top K
    int num_rows = num_src_rows * k;        // total rows sent
    int hidden_dim = 16384 / 4;             // because we're sending floats
    int num_elems = num_rows * hidden_dim;  // total elements sent

    std::vector<std::pair<int, int>> expertForSrcRow(num_rows);
    std::vector<int64_t> expertCount(num_experts);
    std::vector<int64_t> totalExpertCount(num_experts);
    std::vector<int64_t> expertOffsetsCpu(num_experts);
    std::vector<int> expandedSrcRow(num_rows);
    std::vector<int> expertForExpandedSrcRow(num_rows);
    std::function<void()> run;

    void *cooperative_launch_args[13];
    int64_t *expert_counts_gpu_tmp;
    int64_t *expert_counts_gpu;
    int64_t *expert_offsets_gpu;
    int64_t *expert_pos_out_gpu;
    float *send_data, *recv_data;
    int *expandedSrcRow_gpu;
    int *expertForExpandedSrcRow_gpu;

    size_t offset_exchange_shared_memory;
    size_t one_step_shared_memory;
    float milliseconds;
    int gridsize;
    int mype, mype_node, npes;
    int rc;
    int nvshmem_rc;
    int num_blocks;
    int num_copies_per_block;

    cudaStream_t stream;
    cudaEvent_t start, stop;

    /* command line controlled variables */
    int threads_per_block = 1024;
    bool use_cuda_malloc = 0;
    bool randomize_output = 0;
    bool warp_specialized = 0;
    moe_comms_profile_t comms_profile = MOE_BARE_TWO_STEP_ALLPUSH;

    rc = setup_test_parameters(argc, argv, &num_blocks, &threads_per_block, &use_cuda_malloc,
                               &randomize_output, &warp_specialized, &comms_profile, num_rows,
                               num_elems);

    if (rc) {
        return -1;
    }

    num_copies_per_block = num_rows / num_blocks;

    nvshmem_init();
    npes = nvshmem_n_pes();
    mype = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    if (randomize_output) {
        srand(mype * 1000);
    }

    /* kernel shared memory calculations */
    offset_exchange_shared_memory = sizeof(int64_t) * npes * num_experts;
    one_step_shared_memory = sizeof(int64_t) * npes * num_experts * 2;

    /* CUDA state initialization */
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* symmetric memory allocation */
    expert_counts_gpu = (int64_t *)nvshmem_malloc(sizeof(int64_t) * num_experts * num_blocks);
    expert_offsets_gpu = (int64_t *)nvshmem_malloc(sizeof(int64_t) * num_experts);
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_src_rows * hidden_dim);
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_rows * hidden_dim * expert_factor);

    /* user buffer allocation*/
    CUDA_CHECK(cudaMalloc(&expert_pos_out_gpu, sizeof(int64_t) * npes * num_experts * num_blocks));
    if (use_cuda_malloc) {
        CUDA_CHECK(cudaMalloc(&expandedSrcRow_gpu, sizeof(int) * num_rows));
        CUDA_CHECK(cudaMalloc(&expertForExpandedSrcRow_gpu, sizeof(int) * num_rows));
    } else {
        expandedSrcRow_gpu = (int *)malloc(sizeof(int) * num_rows);
        expertForExpandedSrcRow_gpu = (int *)malloc(sizeof(int) * num_rows);
    }

    int cur_expert = 0;
    for (int i = 0; i < expertForSrcRow.size() / k; ++i) {
        for (int j = 0; j < k; ++j) {
            int selected_expert = -1;
            if (randomize_output) {
                do {
                    selected_expert = rand() % num_experts;
                } while (expertCount[selected_expert] > 0);
            } else {
                selected_expert = cur_expert % num_experts;
                cur_expert += 1;
            }
            expertForSrcRow[i + j * num_src_rows] = {selected_expert, i + j * num_src_rows};
            expertCount[selected_expert] += 1;
            totalExpertCount[selected_expert] += 1;
        }
        for (int j = 0; j < k; ++j) {
            expertCount[std::get<0>(expertForSrcRow[i + j * num_src_rows])] = 0;
        }
    }
    expertOffsetsCpu[0] = totalExpertCount[0];
    for (int i = 1; i < num_experts; ++i) {
        expertOffsetsCpu[i] = expertOffsetsCpu[i - 1] + totalExpertCount[i];
    }

    std::vector<std::pair<int, int>> sortedByExpert(expertForSrcRow);

    std::sort(sortedByExpert.begin(), sortedByExpert.end());

    /*
    for (int i = 0; i < num_experts; ++i) {
        std::cout << totalExpertCount[i] << " ";
    }
    std::cout << "\n";
    */

    /*
     std::cout << "Rank " << mype << ": ";
     for (int i = 0; i < sortedByExpert.size(); ++i) {
         std::cout << "("<< std::get<0>(sortedByExpert[i]) << " ," << std::get<1>(sortedByExpert[i])
     << ")";
     }
     */

    for (int i = 0; i < sortedByExpert.size(); ++i) {
        expertForExpandedSrcRow[i] = std::get<0>(sortedByExpert[i]);
        expandedSrcRow[i] = std::get<1>(sortedByExpert[i]);
    }

    expert_counts_gpu_tmp = expert_counts_gpu;
    for (int i = 0; i < num_blocks; i++) {
        cudaMemcpy(expert_counts_gpu_tmp, totalExpertCount.data(), num_experts * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        expert_counts_gpu_tmp += num_experts;
    }
    cudaMemcpy(expert_offsets_gpu, expertOffsetsCpu.data(), num_experts * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(expandedSrcRow_gpu, expandedSrcRow.data(), num_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(expertForExpandedSrcRow_gpu, expertForExpandedSrcRow.data(), num_rows * sizeof(int),
               cudaMemcpyHostToDevice);

    set_counts<<<num_src_rows, 1024, 0, stream>>>((float *)send_data, mype, hidden_dim);

    cooperative_launch_args[0] = &send_data;
    cooperative_launch_args[1] = &recv_data;
    cooperative_launch_args[2] = &expandedSrcRow_gpu;
    cooperative_launch_args[3] = &expertForExpandedSrcRow_gpu;
    cooperative_launch_args[4] = &expert_counts_gpu;
    cooperative_launch_args[5] = &expert_offsets_gpu;
    cooperative_launch_args[6] = &k;
    cooperative_launch_args[7] = &num_rows;
    cooperative_launch_args[8] = &mype;
    cooperative_launch_args[9] = &npes;
    cooperative_launch_args[10] = &hidden_dim;
    if (comms_profile < MOE_ONE_STEP_LAUNCH_MIN) {
        if (comms_profile == MOE_COLL_LAUNCH_TWO_STEP_ALLPUSH) {
            cooperative_launch_args[11] = &num_copies_per_block;
            cooperative_launch_args[12] = &expert_pos_out_gpu;
        } else {
            cooperative_launch_args[11] = &expert_pos_out_gpu;
        }
    } else {
        if (comms_profile == MOE_COLL_LAUNCH_ONE_STEP_ALLPUSH) {
            cooperative_launch_args[11] = &num_copies_per_block;
            cooperative_launch_args[12] = &expert_counts_gpu;
        } else {
            cooperative_launch_args[11] = &expert_counts_gpu;
        }
    }

    switch (comms_profile) {
        case MOE_COLL_LAUNCH_ONE_STEP_ALLPUSH:
        case MOE_BARE_ONE_STEP_ALLPUSH:
            if (warp_specialized) {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_one_step_allpush<true>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            } else {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_one_step_allpush<false>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            }
            break;
        case MOE_COLL_LAUNCH_ONE_STEP_BY_PEER:
        case MOE_BARE_ONE_STEP_BY_PEER:
            if (warp_specialized) {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_one_step_by_peer<true>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            } else {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_one_step_by_peer<false>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            }
            break;
        case MOE_BARE_TWO_STEP_ALLPUSH:
        case MOE_COLL_LAUNCH_TWO_STEP_ALLPUSH:
            if (warp_specialized) {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_two_step_allpush<true>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            } else {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_two_step_allpush<false>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            }
            break;
        case MOE_BARE_TWO_STEP_BY_PEER:
        case MOE_COLL_LAUNCH_TWO_STEP_BY_PEER:
            if (warp_specialized) {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_two_step_by_peer<true>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            } else {
                nvshmem_rc = nvshmemx_collective_launch_query_gridsize(
                    (const void *)token_shuffle_two_step_by_peer<false>, threads_per_block,
                    cooperative_launch_args, 2048, &gridsize);
            }
            break;
        default:
            std::cout << "invalid comms profile (" << comms_profile
                      << ") detected. Cannot continue.\n";
            return -1;
    }

    if (nvshmem_rc != NVSHMEMX_SUCCESS) {
        std::cout << "Failed to query for the gridsize of a collective launch API.\n";
        return -1;
    }

    if (gridsize < num_blocks) {
        std::cout << "gridsize (" << gridsize
                  << ") from collective launch query is smaller than requested blocks ("
                  << num_blocks << "). Cannot continue\n";
        return -1;
    }

    switch (comms_profile) {
        case MOE_BARE_TWO_STEP_ALLPUSH:
            if (warp_specialized) {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    token_shuffle_two_step_allpush<true>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            num_copies_per_block, expert_pos_out_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    token_shuffle_two_step_allpush<false>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            num_copies_per_block, expert_pos_out_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_COLL_LAUNCH_TWO_STEP_ALLPUSH:
            if (warp_specialized) {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    nvshmemx_collective_launch((const void *)token_shuffle_two_step_allpush<true>,
                                               num_blocks, threads_per_block,
                                               cooperative_launch_args, 0, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    nvshmemx_collective_launch((const void *)token_shuffle_two_step_allpush<false>,
                                               num_blocks, threads_per_block,
                                               cooperative_launch_args, 0, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_BARE_TWO_STEP_BY_PEER:
            if (warp_specialized) {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    token_shuffle_two_step_by_peer<true>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    token_shuffle_two_step_by_peer<false>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_COLL_LAUNCH_TWO_STEP_BY_PEER:
            if (warp_specialized) {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    nvshmemx_collective_launch((const void *)token_shuffle_two_step_by_peer<true>,
                                               num_blocks, threads_per_block,
                                               cooperative_launch_args, 0, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    exchange_offsets<<<1, num_experts * npes, offset_exchange_shared_memory,
                                       stream>>>(expert_counts_gpu, expert_pos_out_gpu, npes);
                    nvshmemx_collective_launch((const void *)token_shuffle_two_step_by_peer<false>,
                                               num_blocks, threads_per_block,
                                               cooperative_launch_args, 0, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_BARE_ONE_STEP_ALLPUSH:
            if (warp_specialized) {
                run = [&]() {
                    token_shuffle_one_step_allpush<true>
                        <<<num_blocks, threads_per_block, one_step_shared_memory, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            num_copies_per_block, expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    token_shuffle_one_step_allpush<false>
                        <<<num_blocks, threads_per_block, one_step_shared_memory, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            num_copies_per_block, expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_COLL_LAUNCH_ONE_STEP_ALLPUSH:
            if (warp_specialized) {
                run = [&]() {
                    nvshmemx_collective_launch(
                        (const void *)token_shuffle_one_step_allpush<true>, num_blocks,
                        threads_per_block, cooperative_launch_args, one_step_shared_memory, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    nvshmemx_collective_launch(
                        (const void *)token_shuffle_one_step_allpush<false>, num_blocks,
                        threads_per_block, cooperative_launch_args, one_step_shared_memory, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_BARE_ONE_STEP_BY_PEER:
            if (warp_specialized) {
                run = [&]() {
                    token_shuffle_one_step_by_peer<true>
                        <<<num_blocks, threads_per_block, one_step_shared_memory, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    token_shuffle_one_step_by_peer<false>
                        <<<num_blocks, threads_per_block, one_step_shared_memory, stream>>>(
                            send_data, recv_data, expandedSrcRow_gpu, expertForExpandedSrcRow_gpu,
                            expert_offsets_gpu, k, num_rows, mype, npes, hidden_dim,
                            expert_counts_gpu);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        case MOE_COLL_LAUNCH_ONE_STEP_BY_PEER:
            if (warp_specialized) {
                run = [&]() {
                    nvshmemx_collective_launch(
                        (const void *)token_shuffle_one_step_by_peer<true>, num_blocks,
                        threads_per_block, cooperative_launch_args, one_step_shared_memory, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            } else {
                run = [&]() {
                    nvshmemx_collective_launch(
                        (const void *)token_shuffle_one_step_by_peer<false>, num_blocks,
                        threads_per_block, cooperative_launch_args, one_step_shared_memory, stream);
                    nvshmemx_barrier_all_on_stream(stream);
                };
            }
            break;
        default:
            std::cout << "invalid comms profile (" << comms_profile
                      << ") detected. Cannot continue.\n";
            return -1;
    }

    for (int i = 0; i < 5; ++i) {
        run();
    }
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; ++i) {
        run();
    }
    cudaEventRecord(stop, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventElapsedTime(&milliseconds, start, stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Rank: " << mype << " Time: " << (milliseconds * 1000) / iterations << "\n";

    std::vector<float> recv_data_cpu(expert_factor * num_rows * hidden_dim);

    cudaMemcpy(recv_data_cpu.data(), recv_data, num_rows * hidden_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    /*
    std::cout << "Rank " << mype << ": ";
    for (int i = 0; i < num_rows; ++i) {
        std::cout << std::setprecision(10) << recv_data_cpu[i * hidden_dim + 512] << " ";
    }
    std::cout << "\n";
    */

    nvshmem_free(send_data);
    nvshmem_free(recv_data);

    nvshmem_finalize();

    return 0;
}
