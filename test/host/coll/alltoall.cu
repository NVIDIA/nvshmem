/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "test_teams.h"
#include "coll_common.h"
#include "alltoall_common.h"
#define LARGEST_DT uint64_t
#undef MAX_ELEMS
#define MAX_ELEMS (1024 * 16)
#define MAX_NPES 128

#define DO_ALLTOALL_ON_STREAM_TEST(TYPENAME, TYPE)                                                 \
    do {                                                                                           \
        init_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)source,        \
                                                                      nelems);                     \
        nvshmemx_barrier_on_stream(team, stream);                                                  \
        nvshmemx_##TYPENAME##_alltoall_on_stream(team, (TYPE *)dest, (const TYPE *)source, nelems, \
                                                 stream);                                          \
        validate_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)dest,      \
                                                                          nelems);                 \
        reset_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)dest,         \
                                                                       nelems);                    \
        cudaStreamSynchronize(stream);                                                             \
    } while (0);

#define DO_ALLTOALL_TEST(TYPENAME, TYPE)                                                      \
    do {                                                                                      \
        init_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)source,   \
                                                                      nelems);                \
        cudaStreamSynchronize(stream);                                                        \
        nvshmem_barrier(team);                                                                \
        nvshmem_##TYPENAME##_alltoall(team, (TYPE *)dest, (const TYPE *)source, nelems);      \
        validate_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)dest, \
                                                                          nelems);            \
        reset_##TYPENAME##_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (TYPE *)dest,    \
                                                                       nelems);               \
        cudaStreamSynchronize(stream);                                                        \
    } while (0);

#define DO_ALLTOALLMEM_ON_STREAM_TEST()                                                        \
    do {                                                                                       \
        init_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)source, nelems);   \
        nvshmemx_barrier_on_stream(team, stream);                                              \
        nvshmemx_alltoallmem_on_stream(team, dest, source, nelems, stream);                    \
        validate_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)dest, nelems); \
        reset_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)dest, nelems);    \
        cudaStreamSynchronize(stream);                                                         \
    } while (0);

#define DO_ALLTOALLMEM_TEST()                                                                  \
    do {                                                                                       \
        init_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)source, nelems);   \
        cudaStreamSynchronize(stream);                                                         \
        nvshmem_barrier(team);                                                                 \
        nvshmem_alltoallmem(team, dest, source, nelems);                                       \
        validate_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)dest, nelems); \
        reset_char_alltoall_data_kernel<<<1, 256, 0, stream>>>(team, (char *)dest, nelems);    \
        cudaStreamSynchronize(stream);                                                         \
    } while (0);

int main(int argc, char **argv) {
    int status = 0;
    size_t alloc_size = MAX_ELEMS * MAX_NPES * 2 * sizeof(LARGEST_DT);
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    char size_string[100];
    cudaStream_t stream;
    unsigned long long int errs;
    int iters;
    size_t nelems;
#ifdef _NVSHMEM_DEBUG
    int mype;
#endif

    read_args(argc, argv);
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }
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

    if (use_mmap) {
        d_buffer = (LARGEST_DT *)allocate_mmap_buffer(alloc_size, _mem_handle_type, use_egm);
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    }
    if (!d_buffer) {
        fprintf(stderr, "shmem_malloc failed d_buffer %lu \n", alloc_size);
        status = -1;
        goto out;
    }

    source = (LARGEST_DT *)d_buffer;
    dest = (LARGEST_DT *)&source[MAX_ELEMS * MAX_NPES];

    init_test_teams();
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            DEBUG_PRINT("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                        map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                        nvshmem_team_n_pes(team));
            nelems = 1024;
            iters = 1;
            DEBUG_PRINT("nlemes = %zu\n", nelems);
            for (int iter = 0; iter < iters; iter++) {
                NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DO_ALLTOALL_TEST)
                NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DO_ALLTOALL_ON_STREAM_TEST)
                COLL_CHECK_ERRS_D();
            }
            nvshmem_barrier_all();
        } else {
            nvshmem_barrier_all();
        }
    }
    iters = 10;
    team = NVSHMEM_TEAM_WORLD;
    for (int nelems = 1; nelems <= MAX_ELEMS; nelems *= 2) {
        for (int iter = 0; iter < iters; iter++) {
            DO_ALLTOALL_TEST(float, float)
            DO_ALLTOALL_ON_STREAM_TEST(float, float)
        }
    }
    COLL_CHECK_ERRS_D();

    for (int nelems = 1; nelems <= MAX_ELEMS; nelems *= 2) {
        for (int iter = 0; iter < iters; iter++) {
            DO_ALLTOALLMEM_TEST()
            DO_ALLTOALLMEM_ON_STREAM_TEST()
        }
    }
    COLL_CHECK_ERRS_D();

    finalize_test_teams();
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    finalize_wrapper();

out:
    return errs;
}
