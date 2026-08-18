/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

static int check_status(const char *name, int got, int expected) {
    if (got != expected) {
        fprintf(stderr, "[PE %d] FAIL: %s returned %d, expected %d\n", nvshmem_my_pe(), name, got,
                expected);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int status = 0;
    int active = -1;
    int rc;
    nvshmemx_region_handle_t region = 0;
    nvshmemx_region_attrs_t attrs = NVSHMEMX_REGION_ATTRS_INITIALIZER;
    attrs.hints = NVSHMEMX_REGION_HINT_BATCH_RMA;

    init_wrapper(&argc, &argv);
    status |= check_status("initial region start", nvshmemx_region_start(&region, &attrs),
                           NVSHMEMX_SUCCESS);
    nvshmem_finalize();

    nvshmem_init();
    rc = nvshmemx_region_is_active(NVSHMEMX_REGION_HINT_NONE, &active);
    status |= check_status("region query after reinit", rc, NVSHMEMX_SUCCESS);
    if (rc == NVSHMEMX_SUCCESS && active != 0) {
        fprintf(stderr, "[PE %d] FAIL: region remained active after reinit\n", nvshmem_my_pe());
        status = 1;
    }
    rc = nvshmemx_region_start(&region, &attrs);
    status |= check_status("region start after reinit", rc, NVSHMEMX_SUCCESS);
    if (rc == NVSHMEMX_SUCCESS) {
        status |= check_status("region stop after reinit", nvshmemx_region_stop(region),
                               NVSHMEMX_SUCCESS);
    }

    finalize_wrapper();
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
