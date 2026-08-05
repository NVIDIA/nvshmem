/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

static int check_status(const char *name, int got, int expected) {
    if (got != expected) {
        printf("[PE %d] FAIL: %s returned %d, expected %d\n", nvshmem_my_pe(), name, got, expected);
        return 1;
    }
    return 0;
}

static int check_active(uint32_t hints, int expected) {
    int active = -1;
    int status = nvshmemx_region_is_active(hints, &active);
    if (status != NVSHMEMX_SUCCESS || active != expected) {
        printf("[PE %d] FAIL: nvshmemx_region_is_active(0x%x) status=%d active=%d expected=%d\n",
               nvshmem_my_pe(), hints, status, active, expected);
        return 1;
    }
    return 0;
}

static int verify_data(const char *name, int *device_data, std::vector<int> &host_data,
                       int expected_pe, int nelems) {
    CUDA_CHECK(
        cudaMemcpy(host_data.data(), device_data, sizeof(int) * nelems, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
        int expected = expected_pe * 1000 + i;
        if (host_data[i] != expected) {
            printf("[PE %d] FAIL: %s[%d] = %d, expected %d\n", nvshmem_my_pe(), name, i,
                   host_data[i], expected);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const int nelems = 64;
    constexpr uint32_t unknown_hint = 1u << 31;
    int status = 0;
    int active = -1;
    nvshmemx_region_handle_t region = 0;
    nvshmemx_region_handle_t nested = 0;
    nvshmemx_region_attrs_t batch_attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    nvshmemx_region_attrs_t no_attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    nvshmemx_region_attrs_t invalid_attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    batch_attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;
    invalid_attrs.hints = unknown_hint;

    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    if (npes < 2) {
        printf("[PE %d] SKIP: region host test requires at least 2 PEs\n", mype);
        finalize_wrapper();
        return EXIT_SUCCESS;
    }

    int next_pe = (mype + 1) % npes;
    int prev_pe = (mype - 1 + npes) % npes;
    std::vector<int> host(nelems);
    for (int i = 0; i < nelems; i++) {
        host[i] = mype * 1000 + i;
    }

    int *source = static_cast<int *>(nvshmem_malloc(sizeof(int) * nelems));
    int *recv = static_cast<int *>(nvshmem_malloc(sizeof(int) * nelems));
    int *get_recv = static_cast<int *>(nvshmem_malloc(sizeof(int) * nelems));
    if (source == NULL || recv == NULL || get_recv == NULL) {
        printf("[PE %d] FAIL: nvshmem_malloc failed\n", mype);
        status = 1;
        goto out;
    }

    CUDA_CHECK(cudaMemcpy(source, host.data(), sizeof(int) * nelems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * nelems));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * nelems));

    status |= check_status("region_stop(unmatched)", nvshmemx_region_stop(1),
                           NVSHMEMX_ERROR_INVALID_VALUE);
    status |= check_status("region_start(NULL)", nvshmemx_region_start(NULL, &batch_attrs),
                           NVSHMEMX_ERROR_INVALID_VALUE);
    status |=
        check_status("region_start(unknown hint)", nvshmemx_region_start(&region, &invalid_attrs),
                     NVSHMEMX_ERROR_INVALID_VALUE);
    status |= check_status("region_is_active(NULL)",
                           nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, NULL),
                           NVSHMEMX_ERROR_INVALID_VALUE);
    status |= check_status("region_is_active(unknown hint)",
                           nvshmemx_region_is_active(unknown_hint, &active),
                           NVSHMEMX_ERROR_INVALID_VALUE);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 0);

    nvshmem_barrier_all();

    status |= check_status("region_start(BATCH_RMA)", nvshmemx_region_start(&region, &batch_attrs),
                           NVSHMEMX_SUCCESS);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 1);
    status |= check_active(NVSHMEMX_REGION_HINT_BATCH_RMA, 1);
    status |= check_status("region_start(nested)", nvshmemx_region_start(&nested, &batch_attrs),
                           NVSHMEMX_ERROR_INVALID_VALUE);

    nvshmem_putmem_nbi(recv, source, sizeof(int) * (nelems / 2), next_pe);
    nvshmem_putmem_nbi(recv + (nelems / 2), source + (nelems / 2), sizeof(int) * (nelems / 2),
                       next_pe);
    nvshmem_getmem_nbi(get_recv, source, sizeof(int) * (nelems / 2), next_pe);
    nvshmem_getmem_nbi(get_recv + (nelems / 2), source + (nelems / 2), sizeof(int) * (nelems / 2),
                       next_pe);

    status |= check_status("region_stop(mismatched)", nvshmemx_region_stop(region + 1),
                           NVSHMEMX_ERROR_INVALID_VALUE);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 1);
    nvshmem_fence();
    status |=
        check_status("region_stop(BATCH_RMA)", nvshmemx_region_stop(region), NVSHMEMX_SUCCESS);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 0);

    nvshmem_quiet();
    nvshmem_barrier_all();

    status |= verify_data("recv after fence", recv, host, prev_pe, nelems);
    status |= verify_data("get_recv after fence", get_recv, host, next_pe, nelems);

    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * nelems));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * nelems));
    nvshmem_barrier_all();

    status |= check_status("region_start(stop-only)", nvshmemx_region_start(&region, &batch_attrs),
                           NVSHMEMX_SUCCESS);
    nvshmem_putmem_nbi(recv, source, sizeof(int) * nelems, next_pe);
    nvshmem_getmem_nbi(get_recv, source, sizeof(int) * nelems, next_pe);
    status |=
        check_status("region_stop(stop-only)", nvshmemx_region_stop(region), NVSHMEMX_SUCCESS);

    nvshmem_quiet();
    nvshmem_barrier_all();

    status |= verify_data("recv after stop", recv, host, prev_pe, nelems);
    status |= verify_data("get_recv after stop", get_recv, host, next_pe, nelems);

    CUDA_CHECK(cudaMemset(recv, 0, sizeof(int) * nelems));
    CUDA_CHECK(cudaMemset(get_recv, 0, sizeof(int) * nelems));
    nvshmem_barrier_all();

    status |= check_status("region_start(NULL attrs)", nvshmemx_region_start(&region, NULL),
                           NVSHMEMX_SUCCESS);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 1);
    status |= check_active(NVSHMEMX_REGION_HINT_BATCH_RMA, 0);
    nvshmem_putmem_nbi(recv, source, sizeof(int) * nelems, next_pe);
    nvshmem_getmem_nbi(get_recv, source, sizeof(int) * nelems, next_pe);
    status |=
        check_status("region_stop(NULL attrs)", nvshmemx_region_stop(region), NVSHMEMX_SUCCESS);

    status |= check_status("region_start(NONE)", nvshmemx_region_start(&region, &no_attrs),
                           NVSHMEMX_SUCCESS);
    status |= check_active(NVSHMEMX_REGION_HINT_NONE, 1);
    status |= check_active(NVSHMEMX_REGION_HINT_BATCH_RMA, 0);
    status |= check_status("region_stop(NONE)", nvshmemx_region_stop(region), NVSHMEMX_SUCCESS);

    nvshmem_quiet();
    nvshmem_barrier_all();

    status |= verify_data("recv without batching hint", recv, host, prev_pe, nelems);
    status |= verify_data("get_recv without batching hint", get_recv, host, next_pe, nelems);

out:
    if (source) {
        nvshmem_free(source);
    }
    if (recv) {
        nvshmem_free(recv);
    }
    if (get_recv) {
        nvshmem_free(get_recv);
    }
    finalize_wrapper();

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
