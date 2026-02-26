/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#include <stdio.h>

int main(int argc, char **argv) {
    int status = 0;
    int npes;
    constexpr int num_qp_1 = 1;
    constexpr int num_qp_2 = 4;
    constexpr int num_qp_3 = 1;
    constexpr int num_total_qps = num_qp_1 + num_qp_2 + num_qp_3;
    int num_qps_assigned = 0;
    int qp_base_index;
    nvshmemx_qp_handle_t *qp_handle_1 = NULL;
    nvshmemx_qp_handle_t *qp_handle_2 = NULL;
    nvshmemx_qp_handle_t *qp_handle_3 = NULL;
    nvshmemx_qp_handle_t *qp_handle_all = NULL;

    init_wrapper(&argc, &argv);

    qp_handle_all = (nvshmemx_qp_handle_t *)calloc(num_total_qps, sizeof(nvshmemx_qp_handle_t));
    if (qp_handle_all == NULL) {
        fprintf(stderr, "Failed to allocate qp_handle_all\n");
        goto out;
    }

    npes = nvshmem_n_pes();

    status = nvshmemx_qp_create(num_qp_1, &qp_handle_1);
    if (status != NVSHMEMX_SUCCESS || qp_handle_1 == NULL) {
        fprintf(stderr, "Failed to create qp_handle_1\n");
        goto out;
    }
    assert(qp_handle_1 != NULL);
    printf("qp_handle_1 allocated\n");

    nvshmem_barrier_all();
    for (int i = 0; i < num_qp_1; i++) {
        qp_handle_all[num_qps_assigned] = qp_handle_1[i];
        num_qps_assigned++;
    }
    status = nvshmemx_qp_create(num_qp_2, &qp_handle_2);
    if (status) {
        fprintf(stderr, "Failed to create qp_handle_2\n");
        goto out;
    }
    assert(qp_handle_2 != NULL);
    printf("qp_handle_2 allocated\n");

    nvshmem_barrier_all();
    for (int i = 0; i < num_qp_2; i++) {
        qp_handle_all[num_qps_assigned] = qp_handle_2[i];
        num_qps_assigned++;
    }
    status = nvshmemx_qp_create(num_qp_3, &qp_handle_3);
    if (status) {
        fprintf(stderr, "Failed to create qp_handle_3\n");
        goto out;
    }
    assert(qp_handle_3 != NULL);
    printf("qp_handle_3 allocated\n");

    nvshmem_barrier_all();
    for (int i = 0; i < num_qp_3; i++) {
        qp_handle_all[num_qps_assigned] = qp_handle_3[i];
        num_qps_assigned++;
    }

    if (qp_handle_all[0] == NVSHMEMX_QP_DEFAULT) {
        if (qp_handle_all[1] == NVSHMEMX_QP_DEFAULT && qp_handle_all[2] == NVSHMEMX_QP_DEFAULT) {
            printf(
                "qp_handle_1 is the default qp, assuming the transport doesn't support multiple "
                "qps\n");
            status = 0;
            goto out;
        } else {
            fprintf(stderr,
                    "the first qp of each call is set to the default qp. This indicates the code "
                    "is not working as expected\n");
            status = 1;
            goto out;
        }
    }

    qp_base_index = qp_handle_all[0];
    for (int i = 1; i < num_total_qps; i++) {
        if (qp_handle_all[i] != qp_base_index + i * npes) {
            fprintf(
                stderr,
                "qpair indices are not strided as expected. Expected difference is %d, got %d\n",
                npes, qp_handle_all[i] - qp_handle_all[i - 1]);
            status = 1;
            goto out;
        }
    }
    printf("qp indices are strided as expected\n");

    finalize_wrapper();

out:
    free(qp_handle_all);
    return status;
}