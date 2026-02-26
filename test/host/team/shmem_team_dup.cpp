/*
 * * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See License.txt for license information
 * */

#include <stdio.h>
#include <stdlib.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define NUM_TEAMS 16

unsigned int ret = 0;
unsigned int dest_ret = 0;

int main(int argc, char **argv) {
    int me, npes, i, j;
    int team_count = 0;
    cudaStream_t cstrm;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    me = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    nvshmem_team_t new_team[NUM_TEAMS];

    unsigned int *ret_d;
    unsigned int *dest_ret_d;
    if (use_mmap) {
        ret_d = (unsigned int *)allocate_mmap_buffer(sizeof(unsigned int) * 1, _mem_handle_type,
                                                     use_egm, true);
        dest_ret_d =
            (unsigned int *)allocate_mmap_buffer(sizeof(unsigned int), _mem_handle_type, use_egm);
    } else {
        ret_d = (unsigned int *)nvshmem_calloc(1, sizeof(unsigned int));
        dest_ret_d = (unsigned int *)nvshmem_malloc(sizeof(unsigned int));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    /* Make duplicate of TEAM_WORLD */
    for (i = j = 0; i < NUM_TEAMS / 2;) {
        /* Make a duplicate team of indentical set */
        ret = nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, NULL, 0, &new_team[i]);
        /* Wait for team PEs to fill in ret before starting the reduction */
        nvshmem_team_sync(new_team[i]);
        CUDA_CHECK(
            cudaMemcpyAsync(ret_d, &ret, sizeof(unsigned int), cudaMemcpyHostToDevice, cstrm));
        CUDA_CHECK(cudaStreamSynchronize(cstrm));
        nvshmem_uint_or_reduce(new_team[i], dest_ret_d, ret_d, 1);
        cudaMemcpy(&dest_ret, dest_ret_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        /* If success was not global, free a team and retry */
        if (dest_ret != 0) {
            if (ret == 0) {
                printf("%d: Local success and global failure on iteration %d\n", me, i);
                exit(1);
            }

            /* No more teams to free */
            if (i == j) break;

            nvshmem_team_destroy(new_team[j]);
            j++;
        } else {
            i++;
            team_count++;
        }
    }

    printf("The number of teams created for PE %d is : %d\n", me, team_count);

    npes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    /* Make duplicate of TEAM_NODE */
    for (i = j = NUM_TEAMS / 2; i < NUM_TEAMS;) {
        /* Make a duplicate team of indentical set */
        ret = nvshmem_team_split_strided(NVSHMEMX_TEAM_NODE, 0, 1, npes, NULL, 0, &new_team[i]);
        /* Wait for team PEs to fill in ret before starting the reduction */
        nvshmem_team_sync(new_team[i]);
        CUDA_CHECK(
            cudaMemcpyAsync(ret_d, &ret, sizeof(unsigned int), cudaMemcpyHostToDevice, cstrm));
        CUDA_CHECK(cudaStreamSynchronize(cstrm));
        nvshmem_uint_or_reduce(new_team[i], dest_ret_d, ret_d, 1);
        cudaMemcpy(&dest_ret, dest_ret_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        /* If success was not global, free a team and retry */
        if (dest_ret != 0) {
            if (ret == 0) {
                printf("%d: Local success and global failure on iteration %d\n", me, i);
                exit(1);
            }

            /* No more teams to free */
            if (i == j) break;

            nvshmem_team_destroy(new_team[j]);
            j++;
        } else {
            i++;
            team_count++;
        }
    }

    printf("The number of teams created for PE %d is : %d\n", me, team_count);

    finalize_wrapper();

    return 0;
}
