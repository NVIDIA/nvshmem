/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_TEST_NONUNIFORM_TEAMS_CUH
#define NVSHMEM_TEST_NONUNIFORM_TEAMS_CUH

#include "utils.h"
#include <algorithm>
#include <climits>
#include <string>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "nonuniform_pes: %s\n", message);
        nvshmem_global_exit(1);
    }
}

__global__ void check_collectives(nvshmem_team_t team, int *src, int *dst, double2 *max_src,
                                  double2 *max_dst, int *errors) {
    int me = nvshmem_my_pe();
    int size = nvshmem_team_n_pes(team);
    int team_me = nvshmem_team_my_pe(team);
    int expected_sum = 0, expected_max = -1;
    for (int i = 0; i < size; ++i) {
        int pe = nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD);
        expected_sum += pe + 1;
        expected_max = max(expected_max, pe);
    }
    if (threadIdx.x == 0) {
        src[0] = src[1] = me + 1;
        max_src[0] = make_double2(me + 1, me);
    }
    __syncthreads();
    nvshmemx_barrier_block(team);
    nvshmemx_int_sum_reduce_block(team, dst, src, 2);
    if (threadIdx.x == 0 && (dst[0] != expected_sum || dst[1] != expected_sum)) {
        atomicAdd(errors, 1);
    }
    nvshmemx_barrier_block(team);
    nvshmemx_int_fcollect_block(team, dst, src, 2);
    if (threadIdx.x == 0) {
        for (int i = 0; i < size; ++i) {
            int pe = nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD);
            if (dst[2 * i] != pe + 1 || dst[2 * i + 1] != pe + 1) {
                atomicAdd(errors, 1);
            }
        }
    }
    nvshmemx_barrier_block(team);
    for (int root = 0; root < size; ++root) {
        nvshmemx_int_broadcast_block(team, dst, src, 2, root);
        int pe = nvshmem_team_translate_pe(team, root, NVSHMEM_TEAM_WORLD);
        if (threadIdx.x == 0 && (dst[0] != pe + 1 || dst[1] != pe + 1)) {
            atomicAdd(errors, 1);
        }
        nvshmemx_barrier_block(team);
    }
    nvshmemx_double2_maxloc_reduce_block(team, max_dst, max_src, 1);
    if (threadIdx.x == 0 && (max_dst[0].x != expected_max + 1 || max_dst[0].y != expected_max)) {
        atomicAdd(errors, 1);
    }
    nvshmemx_barrier_block(team);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        src[i] = me * 1000 + i;
    }
    __syncthreads();
    nvshmemx_int_alltoall_block(team, dst, src, 1);
    if (threadIdx.x == 0) {
        for (int i = 0; i < size; ++i) {
            int pe = nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD);
            if (dst[i] != pe * 1000 + team_me) {
                atomicAdd(errors, 1);
            }
        }
    }
    nvshmemx_barrier_block(team);
}

__global__ void check_device_rma(int *src, int *dst, int *errors) {
    int me = nvshmem_my_pe(), npes = nvshmem_n_pes();
    int next = (me + 1) % npes, prev = (me + npes - 1) % npes;
    if (threadIdx.x == 0) {
        src[0] = me + 17;
    }
    nvshmemx_barrier_all_block();
    if (threadIdx.x == 0) {
        nvshmem_int_p(dst, me + 29, next);
        nvshmem_quiet();
    }
    nvshmemx_barrier_all_block();
    if (threadIdx.x == 0 && (dst[0] != prev + 29 || nvshmem_int_g(src, next) != next + 17)) {
        atomicAdd(errors, 1);
    }
    nvshmemx_barrier_all_block();
}

static void check_team(nvshmem_team_t team, const std::vector<int> &expected, int me) {
    require(nvshmem_team_n_pes(team) == static_cast<int>(expected.size()), "team size mismatch");
    auto it = std::find(expected.begin(), expected.end(), me);
    require(it != expected.end(), "caller missing from expected team");
    require(nvshmem_team_my_pe(team) == it - expected.begin(), "team local rank mismatch");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD) == expected[i],
                "team-to-world translation mismatch");
        require(
            nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD, expected[i], team) == static_cast<int>(i),
            "world-to-team translation mismatch");
    }
}

static void check_nonuniform_teams() {
    int me = nvshmem_my_pe(), npes = nvshmem_n_pes();
    // Hostnames provide an independent oracle for the public topology teams.
    constexpr size_t name_size = 256;
    char hostname[name_size] = {};
    require(gethostname(hostname, name_size - 1) == 0, "gethostname failed");
    char *name_src = static_cast<char *>(nvshmem_malloc(name_size));
    char *name_dst = static_cast<char *>(nvshmem_malloc(name_size * npes));
    require(name_src && name_dst, "hostname allocation failed");
    CUDA_CHECK(cudaMemcpy(name_src, hostname, name_size, cudaMemcpyHostToDevice));
    nvshmem_barrier_all();
    require(nvshmem_char_fcollect(NVSHMEM_TEAM_WORLD, name_dst, name_src, name_size) == 0,
            "hostname exchange failed");
    std::vector<char> names(name_size * npes);
    CUDA_CHECK(cudaMemcpy(names.data(), name_dst, names.size(), cudaMemcpyDeviceToHost));
    nvshmem_free(name_dst);
    nvshmem_free(name_src);
    std::vector<int> pe_local(npes), node_pes, same_rank_pes;
    for (int pe = 0; pe < npes; ++pe) {
        for (int j = 0; j < pe; ++j) {
            pe_local[pe] += !strcmp(&names[pe * name_size], &names[j * name_size]);
        }
        if (!strcmp(&names[pe * name_size], hostname)) {
            node_pes.push_back(pe);
        }
    }
    for (int pe = 0; pe < npes; ++pe) {
        if (pe_local[pe] == pe_local[me]) {
            same_rank_pes.push_back(pe);
        }
    }
    check_team(NVSHMEMX_TEAM_NODE, node_pes, me);
    check_team(NVSHMEMX_TEAM_SAME_MYPE_NODE, same_rank_pes, me);
    for (nvshmem_team_t parent :
         {NVSHMEM_TEAM_WORLD, NVSHMEMX_TEAM_NODE, NVSHMEMX_TEAM_SAME_MYPE_NODE}) {
        int size = nvshmem_team_n_pes(parent);
        auto reject_split = [&](int start, int stride, int count) {
            nvshmem_team_t child = NVSHMEM_TEAM_WORLD;
            require(nvshmem_team_split_strided(parent, start, stride, count, nullptr, 0, &child) ==
                        NVSHMEMX_ERROR_INVALID_VALUE,
                    "invalid split was not rejected");
            require(child == NVSHMEM_TEAM_INVALID, "invalid split returned a child team");
        };
        reject_split(size, 1, 1);
        if (size > 1) {
            reject_split(size - 1, 2, 2);
            reject_split(0, INT_MAX, size);
        }
    }
    int *src = static_cast<int *>(nvshmem_malloc(std::max(8, 2 * npes) * sizeof(int)));
    int *dst = static_cast<int *>(nvshmem_malloc(std::max(8, 2 * npes) * sizeof(int)));
    double2 *max_src = static_cast<double2 *>(nvshmem_malloc(sizeof(double2)));
    double2 *max_dst = static_cast<double2 *>(nvshmem_malloc(sizeof(double2)));
    int *errors;
    CUDA_CHECK(cudaMalloc(&errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(int)));
    require(src && dst && max_src && max_dst, "symmetric allocation failed");
    int value;
    if (me == 0) {
        printf("phase=device-rma\n");
        fflush(stdout);
    }
    check_device_rma<<<1, 128>>>(src, dst, errors);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto run_on_stream = [&](nvshmem_team_t team) {
        if (me == 0) {
            printf("phase=on-stream team=%d\n", team);
            fflush(stdout);
        }
        int input[] = {me + 1, me + 1}, output[2], expected = 0;
        int size = nvshmem_team_n_pes(team);
        for (int i = 0; i < size; ++i) {
            expected += nvshmem_team_translate_pe(team, i, NVSHMEM_TEAM_WORLD) + 1;
        }
        CUDA_CHECK(cudaMemcpyAsync(src, input, sizeof(input), cudaMemcpyHostToDevice, stream));
        nvshmemx_barrier_on_stream(team, stream);
        require(nvshmemx_int_sum_reduce_on_stream(team, dst, src, 2, stream) == 0,
                "on-stream reduction failed");
        CUDA_CHECK(cudaMemcpyAsync(output, dst, sizeof(output), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        require(output[0] == expected && output[1] == expected, "on-stream reduction mismatch");
        nvshmem_barrier_all();
    };
    run_on_stream(NVSHMEM_TEAM_WORLD);
    run_on_stream(NVSHMEMX_TEAM_NODE);
    run_on_stream(NVSHMEMX_TEAM_SAME_MYPE_NODE);
    CUDA_CHECK(cudaStreamDestroy(stream));
    auto run_collectives = [&](nvshmem_team_t team) {
        if (me == 0) {
            printf("phase=device-collectives team=%d\n", team);
            fflush(stdout);
        }
        check_collectives<<<1, 128>>>(team, src, dst, max_src, max_dst, errors);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
    };
    run_collectives(NVSHMEM_TEAM_WORLD);
    run_collectives(NVSHMEMX_TEAM_NODE);
    run_collectives(NVSHMEMX_TEAM_SAME_MYPE_NODE);
    for (int iteration = 0; iteration < 3; ++iteration) {
        nvshmem_team_t team;
        require(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &team) == 0,
                "world duplicate creation failed");
        run_collectives(team);
        nvshmem_team_destroy(team);
    }
    if (npes > 1) {
        nvshmem_team_t team;
        require(
            nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 1, 1, npes - 1, nullptr, 0, &team) == 0,
            "world subset creation failed");
        require((team != NVSHMEM_TEAM_INVALID) == (me > 0), "world subset membership mismatch");
        if (me > 0) {
            std::vector<int> pes;
            for (int pe = 1; pe < npes; ++pe) {
                pes.push_back(pe);
            }
            check_team(team, pes, me);
            check_collectives<<<1, 128>>>(team, src, dst, max_src, max_dst, errors);
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_team_destroy(team);
        }
        nvshmem_barrier_all();
    }
    if (me == 0) {
        printf("phase=reserved-team-splits\n");
        fflush(stdout);
    }
    for (nvshmem_team_t parent : {NVSHMEMX_TEAM_NODE, NVSHMEMX_TEAM_SAME_MYPE_NODE}) {
        int size = nvshmem_team_n_pes(parent), parent_me = nvshmem_team_my_pe(parent);
        for (int stride : {1, 2}) {
            int start = stride == 1 || size == 1 ? 0 : 1;
            int count = (size - start + stride - 1) / stride;
            nvshmem_team_t team;
            require(
                nvshmem_team_split_strided(parent, start, stride, count, nullptr, 0, &team) == 0,
                "reserved team split failed");
            bool member = parent_me >= start && (parent_me - start) % stride == 0;
            require((team != NVSHMEM_TEAM_INVALID) == member, "reserved split membership mismatch");
            if (member) {
                std::vector<int> pes;
                for (int i = 0; i < count; ++i) {
                    pes.push_back(
                        nvshmem_team_translate_pe(parent, start + stride * i, NVSHMEM_TEAM_WORLD));
                }
                check_team(team, pes, me);
                check_collectives<<<1, 128>>>(team, src, dst, max_src, max_dst, errors);
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_team_t duplicate;
                require(nvshmem_team_split_strided(team, 0, 1, count, nullptr, 0, &duplicate) == 0,
                        "nested split failed");
                nvshmem_team_destroy(team);
                check_team(duplicate, pes, me);
                check_collectives<<<1, 128>>>(duplicate, src, dst, max_src, max_dst, errors);
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_team_destroy(duplicate);
            }
            nvshmem_barrier_all();
        }
    }
    CUDA_CHECK(cudaMemcpy(&value, errors, sizeof(int), cudaMemcpyDeviceToHost));
    require(value == 0, "device RMA or collective result mismatch");
    printf("nonuniform_pes PASS rank=%d npes=%d errors=%d\n", me, npes, value);
    CUDA_CHECK(cudaFree(errors));
    nvshmem_free(max_dst);
    nvshmem_free(max_src);
    nvshmem_free(dst);
    nvshmem_free(src);
}

#endif
