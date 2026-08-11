/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "test_teams.h"
#include "barrier_common.h"
#include "coll_common.h"
#include "utils.h"

#define MAX_ITERS 50
#define MAX_NPES 128

void set_remote_data(nvshmem_team_t team, int *buf_d) {
    int team_n_pes = nvshmem_team_n_pes(team);
    int team_my_pe = nvshmem_team_my_pe(team);
    for (int pe = 0; pe < team_n_pes; pe++) {
        nvshmem_int_p(&buf_d[team_my_pe], team_my_pe,
                      nvshmem_team_translate_pe(team, pe, NVSHMEM_TEAM_WORLD));
    }
}

void validate_data(nvshmem_team_t team, int *buf_h, int *buf_d, unsigned long long int *errs) {
    int team_n_pes = nvshmem_team_n_pes(team);
    cudaMemcpy(buf_h, buf_d, team_n_pes * sizeof(int), cudaMemcpyDeviceToHost);
    for (int pe = 0; pe < team_n_pes; pe++) {
        if (buf_h[pe] != pe) {
            printf("error: buf_idx = %d, found = %d, expected = %d, team = %s\n", pe, buf_h[pe], pe,
                   map_team_to_string[team].c_str());
            (*errs)++;
        }
    }
}

void reset_data(nvshmem_team_t team, int *buf_d) {
    int team_n_pes = nvshmem_team_n_pes(team);
    if (use_egm) {
        memset(buf_d, 0, team_n_pes * sizeof(int));
    } else {
        cudaMemset(buf_d, 0, team_n_pes * sizeof(int));
    }
    cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    size_t alloc_size = MAX_NPES * sizeof(int);
    int *buf_d, *buf_h;
    char size_string[100];
    cudaStream_t stream;
    unsigned long long int errs = 0;
#ifdef _NVSHMEM_DEBUG
    nvshmem_team_t team;
    int team_n_pes;
    int team_my_pe;
#endif

    read_args(argc, argv);
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }
    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    int npes;
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
#endif

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (use_mmap) {
        buf_d = (int *)allocate_mmap_buffer(alloc_size, _mem_handle_type, use_egm);
    } else {
        buf_d = (int *)nvshmem_malloc(alloc_size);
    }
    if (!buf_d) {
        fprintf(stderr, "shmem_malloc failed \n");
        status = -1;
        goto out;
    }
    if (status) {
        ERROR_PRINT("[%d] barrier failed \n", mype);
    }

    cudaMallocHost((void **)&buf_h, alloc_size);

#ifdef _NVSHMEM_DEBUG
    team = NVSHMEM_TEAM_WORLD;
    team_n_pes = nvshmem_team_n_pes(team);
    team_my_pe = nvshmem_team_my_pe(team);
#endif

    DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                map_team_to_string[team].c_str(), team_my_pe, team_n_pes);

    /* Test nvshmem_barrier_all */
    for (int i = 0; i < MAX_ITERS; i++) {
        set_remote_data(NVSHMEM_TEAM_WORLD, buf_d);
        nvshmem_barrier_all();
        validate_data(NVSHMEM_TEAM_WORLD, buf_h, buf_d, &errs);
        reset_data(NVSHMEM_TEAM_WORLD, buf_d);
        nvshmem_barrier_all();
        COLL_CHECK_ERRS();
    }
    DEBUG_PRINT("Completed nvshmem_barrier_all loop\n");

    /* Test nvshmemx_barrier_all_on_stream */
    for (int i = 0; i < MAX_ITERS; i++) {
        DEBUG_PRINT("In barrier_all_on_stream loop, iter: %d\n", i);
        init_barrier_data_kernel<<<1, 32, 0, stream>>>(NVSHMEM_TEAM_WORLD, buf_d);
        nvshmemx_barrier_all_on_stream(stream);
        validate_barrier_data_kernel<<<1, 32, 0, stream>>>(NVSHMEM_TEAM_WORLD, buf_d);
        reset_barrier_data_kernel<<<1, 32, 0, stream>>>(NVSHMEM_TEAM_WORLD, buf_d);
        nvshmemx_barrier_all_on_stream(stream);
    }
    cudaStreamSynchronize(stream);
    COLL_CHECK_ERRS_D();
    DEBUG_PRINT("Completed both barrier loops\n");

    CUDA_CHECK(cudaFreeHost(buf_h));
    if (use_mmap) {
        free_mmap_buffer(buf_d);
    } else {
        nvshmem_free(buf_d);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    finalize_wrapper();

out:
    return errs;
}
