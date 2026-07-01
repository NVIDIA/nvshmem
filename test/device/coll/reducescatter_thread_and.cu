/*
 * Copyright (c) 2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define CUMODULE_NAME "reducescatter_thread_and.cubin"

#include "utils.h"
#include "coll_test.h"
#include <string>
#include <iostream>
#include <sstream>
#include "test_teams.h"
#include "coll_common.h"
#include "reducescatter_common.h"

using namespace std;

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DECL_TYPENAME_OP_REDUCESCATTER, thread, ,
                                                      , and)

NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DEFN_TYPENAME_OP_REDUCESCATTER, thread, ,
                                                      , and)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    unsigned num_threads;
    size_t alloc_size = MAX_ELEMS * sizeof(LARGEST_DT) /* For dest */ +
                        MAX_ELEMS * sizeof(LARGEST_DT) * MAX_NPES /* For source */;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *source, *dest;
    cudaStream_t cstrm;
    size_t nelems;
    char size_string[100];
    unsigned long long int errs;

    read_args(argc, argv);
    if (use_mmap) {
        alloc_size = pad_up(alloc_size);
    }
    DEBUG_PRINT("symmetric size %zu\n", alloc_size);
    sprintf(size_string, "%zu", alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failted \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
    if (use_mmap) {
        d_buffer =
            (LARGEST_DT *)allocate_mmap_buffer(alloc_size * 1, _mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocating mmaped buffer\n");
    } else {
        d_buffer = (LARGEST_DT *)nvshmem_calloc(alloc_size, 1);
    }
    if (!d_buffer) {
        ERROR_PRINT("nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }
    assert(nvshmem_n_pes() <= MAX_NPES); /* alloc_size needs to be adjusted to increase n_pes */
    source = d_buffer;
    dest = &source[MAX_ELEMS * nvshmem_n_pes()];

    nvshmem_barrier_all();

    init_test_teams();
    nvshmem_team_t team;
    while (get_next_team(&team)) {
        if (team != NVSHMEM_TEAM_INVALID) {
            printf("%d: Testing team: %s (my_pe = %d, n_pes = %d)\n", mype,
                   map_team_to_string[team].c_str(), nvshmem_team_my_pe(team),
                   nvshmem_team_n_pes(team));
            num_threads = 1;
            for (nelems = 1; nelems <= ELEMS_PER_THREAD * num_threads; nelems *= 2) {
                DEBUG_PRINT("thread version, num elements = %ld\n", nelems);
                NVSHMEMTEST_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DO_REDUCESCATTER_DEVICE_TEST,
                                                                      thread, , , and)
            }

            COLL_CHECK_ERRS_D();
            nvshmem_barrier_all();
        } else {
            nvshmem_barrier_all();
        }
    }

    finalize_test_teams();
    if (use_mmap) {
        free_mmap_buffer(d_buffer);
    } else {
        nvshmem_free(d_buffer);
    }
    CUDA_RUNTIME_CHECK(cudaStreamDestroy(cstrm));
    finalize_wrapper();

out:
    return errs;
}
