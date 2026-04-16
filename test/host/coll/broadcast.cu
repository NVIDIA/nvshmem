/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "test_teams.h"
#include "coll_common.h"
#include "broadcast_common.h"
#define MAX_ITERS 1
#undef MAX_ELEMS
#define MAX_ELEMS (1024 * 1024)
#define MAX_NPES 128
#define LARGEST_DT uint64_t

#define DO_BROADCAST_TEST(TYPENAME, TYPE)                                                          \
    do {                                                                                           \
        init_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)source,         \
                                                                     nelems);                      \
        nvshmemx_barrier_on_stream(team, stream);                                                  \
        cudaStreamSynchronize(stream);                                                             \
        nvshmem_##TYPENAME##_broadcast(team, (TYPE *)dest, (const TYPE *)source, nelems, PE_root); \
        validate_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest,       \
                                                                         nelems);                  \
        reset_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest, nelems); \
        cudaStreamSynchronize(stream);                                                             \
    } while (0);

#define DO_BROADCAST_ON_STREAM_TEST(TYPENAME, TYPE)                                                \
    do {                                                                                           \
        init_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)source,         \
                                                                     nelems);                      \
        nvshmemx_barrier_on_stream(team, stream);                                                  \
        nvshmemx_##TYPENAME##_broadcast_on_stream(team, (TYPE *)dest, (const TYPE *)source,        \
                                                  nelems, PE_root, stream);                        \
        validate_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest,       \
                                                                         nelems);                  \
        reset_##TYPENAME##_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest, nelems); \
        cudaStreamSynchronize(stream);                                                             \
    } while (0);

#define DO_BROADCASTMEM_ON_STREAM_TEST()                                                      \
    do {                                                                                      \
        init_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)source, nelems);   \
        nvshmemx_barrier_on_stream(team, stream);                                             \
        nvshmemx_broadcastmem_on_stream(team, dest, source, nelems, PE_root, stream);         \
        validate_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)dest, nelems); \
        reset_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)dest, nelems);    \
        cudaStreamSynchronize(stream);                                                        \
    } while (0);

#define DO_BROADCASTMEM_TEST()                                                                \
    do {                                                                                      \
        init_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)source, nelems);   \
        cudaStreamSynchronize(stream);                                                        \
        nvshmem_barrier(team);                                                                \
        nvshmem_broadcastmem(team, dest, source, nelems, PE_root);                            \
        validate_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)dest, nelems); \
        reset_char_broadcast_data_kernel<<<1, 1, 0, stream>>>(team, (char *)dest, nelems);    \
        cudaStreamSynchronize(stream);                                                        \
    } while (0);

int main(int argc, char **argv) {
    int status = 0;
    size_t alloc_size = MAX_ELEMS * 2 * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    char size_string[100];
    cudaStream_t stream;
    unsigned long long int errs;
    int PE_root = 0;
#ifdef _NVSHMEM_DEBUG
    int mype;
#endif

    DEBUG_PRINT("symmetric size requested %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

#ifdef _NVSHMEM_DEBUG
    int npes;
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
#endif

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "shmem_malloc failed d_buffer %lu \n", alloc_size);
        status = -1;
        goto out;
    }

    source = (LARGEST_DT *)d_buffer;
    dest = (LARGEST_DT *)&source[MAX_ELEMS];

    init_test_teams();
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));
            for (size_t nelems = 1; nelems <= MAX_ELEMS; nelems *= 2) {
                DEBUG_PRINT("nlemes = %zu\n", nelems);
                for (int iter = 0; iter < MAX_ITERS; iter++) {
                    /* Limiting to two types because this test takes a long time to run. */
                    DO_BROADCAST_TEST(char, char)
                    DO_BROADCAST_TEST(int32, int32_t)
                    DO_BROADCAST_TEST(uint64, uint64_t)
                    DO_BROADCAST_ON_STREAM_TEST(char, char)
                    DO_BROADCAST_ON_STREAM_TEST(int32, int32_t)
                    DO_BROADCAST_ON_STREAM_TEST(uint64, uint64_t)
                    COLL_CHECK_ERRS_D();
                }
            }
        }
        nvshmem_barrier_all();
    }

    team = NVSHMEM_TEAM_WORLD;

    for (int nelems = 1; nelems <= MAX_ELEMS; nelems *= 2) {
        for (int iter = 0; iter < MAX_ITERS; iter++) {
            DO_BROADCASTMEM_TEST()
            DO_BROADCASTMEM_ON_STREAM_TEST()
        }
    }
    COLL_CHECK_ERRS_D();

    finalize_test_teams();
    nvshmem_free(d_buffer);
    CUDA_CHECK(cudaStreamDestroy(stream));
    finalize_wrapper();

out:
    return errs;
}
