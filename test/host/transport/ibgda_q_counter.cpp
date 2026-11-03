/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx5_ifc.h"
#include "transport_mlx5_common.h"

static_assert(offsetof(struct mlx5_ifc_dctc_bits, counter_set_id) == 0x60,
              "DCT counter ID must start at bit 0x60");
static_assert(sizeof(((struct mlx5_ifc_dctc_bits *)0)->counter_set_id) == 0x8,
              "DCT counter ID must be 8 bits");
static_assert(offsetof(struct mlx5_ifc_dctc_bits, data_in_order) == 0x1d,
              "Preserve the existing data_in_order offset");
static_assert(sizeof(struct mlx5_ifc_dctc_bits) == 0x380, "Preserve the simplified DCTC size");
static_assert(sizeof(struct mlx5_ifc_query_dct_in_bits) / 8 == 0x10,
              "QUERY_DCT input must be 16 bytes");
static_assert(sizeof(struct mlx5_ifc_query_dct_out_bits) / 8 == 0x80,
              "QUERY_DCT output must be 128 bytes");
static_assert((offsetof(struct mlx5_ifc_query_dct_out_bits, dctc) +
               offsetof(struct mlx5_ifc_dctc_bits, counter_set_id)) /
                      8 ==
                  0x1c,
              "Counter ID must be at byte 0x1c of the response");
static_assert(sizeof(struct mlx5_ifc_query_qp_out_bits) / 8 == 0x110,
              "RESET fallback must use the complete QUERY_QP response");
static_assert(offsetof(struct mlx5_ifc_qpc_bits, state) == 0 && MLX5_QPC_STATE_RST == 0,
              "RESET fallback must read the QPC state field");

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

/* Exercise the same retry gate used by ibgda_qp_rst2init, without any hardware calls. */
static int test_retry_policy() {
    CHECK(!nvshmemt_mlx5_q_counter_can_retry(0, 0));
    CHECK(nvshmemt_mlx5_q_counter_can_retry(0, 3));
    CHECK(nvshmemt_mlx5_q_counter_can_retry(EINVAL, 0));
    CHECK(nvshmemt_mlx5_q_counter_can_retry(-EOPNOTSUPP, 0));
#ifdef EREMOTEIO
    CHECK(nvshmemt_mlx5_q_counter_can_retry(EREMOTEIO, 3));
#endif
    CHECK(!nvshmemt_mlx5_q_counter_can_retry(ETIMEDOUT, 3));
    CHECK(!nvshmemt_mlx5_q_counter_can_retry(EIO, 3));
    CHECK(!nvshmemt_mlx5_q_counter_can_retry(INT_MIN, 3));

    const int no_retry_errors[] = {ETIMEDOUT, EIO, ENOMEM, 0};
    for (int error : no_retry_errors) {
        int queries = 0, retries = 0;
        uint8_t disabled = 0;
        bool attempted = nvshmemt_mlx5_q_counter_retry(
            error, 0, 1, disabled,
            [&]() {
                ++queries;
                return true;
            },
            [&]() {
                ++retries;
                return true;
            });
        CHECK(!attempted && !queries && !retries && !disabled);
    }

    int queries = 0, retries = 0;
    uint8_t disabled = 0;
    bool attempted = nvshmemt_mlx5_q_counter_retry(
        EINVAL, 0, 1, disabled,
        [&]() {
            ++queries;
            return false;
        },
        [&]() {
            ++retries;
            return true;
        });
    CHECK(!attempted && queries == 1 && retries == 0 && disabled == 0);

    queries = retries = 0;
    attempted = nvshmemt_mlx5_q_counter_retry(
        EINVAL, 0, 1, disabled,
        [&]() {
            ++queries;
            return true;
        },
        [&]() {
            ++retries;
            return false;
        });
    CHECK(attempted && queries == 1 && retries == 1 && disabled == 0);

    uint8_t ports[2] = {};
    queries = retries = 0;
    attempted = nvshmemt_mlx5_q_counter_retry(
        EINVAL, 0, 1, ports[0],
        [&]() {
            ++queries;
            return true;
        },
        [&]() {
            ++retries;
            return true;
        });
    CHECK(attempted && queries == 1 && retries == 1 && ports[0] == 1 && ports[1] == 0);
    CHECK(!(ports[0] & 2));  // DCI rejection must not disable RC or a different port.
    attempted = nvshmemt_mlx5_q_counter_retry(
        0, 3, 2, ports[0], []() { return true; }, []() { return true; });
    CHECK(attempted && ports[0] == 3 && ports[1] == 0);
    return 0;
}

#ifdef NVSHMEM_TEST_MLX5_QUERY
#include <infiniband/mlx5dv.h>
#include <string.h>
#include "c_api/nvshmemx_status.h"

static int query_calls;
static int query_result;
static uint8_t firmware_status;
static uint8_t response_counter;
static bool valid_command;

static int mock_query(struct ibv_qp *qp, const void *in, size_t inlen, void *out, size_t outlen) {
    ++query_calls;
    valid_command = inlen == 0x10 && outlen == 0x80 &&
                    DEVX_GET(query_dct_in, in, opcode) == MLX5_CMD_OP_QUERY_DCT &&
                    DEVX_GET(query_dct_in, in, dctn) == qp->qp_num;
    if (!valid_command) return EINVAL;
    memset(out, 0, outlen);
    DEVX_SET(query_dct_out, out, status, firmware_status);
    DEVX_SET(query_dct_out, out, syndrome, firmware_status ? 0x12345678 : 0);
    /* Write the documented byte position independently of the newly named IFC field. */
    static_cast<uint8_t *>(out)[0x1c] = response_counter;
    if (query_result) errno = query_result;
    return query_result;
}

static int test_queries() {
    struct ibv_qp dct = {};
    dct.qp_num = 0xabcdef;
    uint8_t id = 0xff;

    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, nullptr) ==
          NVSHMEMX_ERROR_INVALID_VALUE);
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, nullptr, &id) ==
          NVSHMEMX_ERROR_INVALID_VALUE);
    CHECK(id == 0 && query_calls == 0);
    id = 0xff;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(nullptr, &dct, &id) == NVSHMEMX_ERROR_NOT_SUPPORTED);
    CHECK(id == 0 && query_calls == 0);

    const uint8_t ids[] = {1, 0x7f, 0xff};
    for (uint8_t expected : ids) {
        response_counter = expected;
        CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_SUCCESS);
        CHECK(valid_command && id == expected);
    }

    response_counter = 0;
    id = 0xff;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_ERROR_NOT_SUPPORTED);
    CHECK(valid_command && id == 0);

    response_counter = 42;
    firmware_status = 3;
    id = 0xff;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_ERROR_INTERNAL);
    CHECK(valid_command && id == 0);

    firmware_status = 0;
    query_result = EOPNOTSUPP;
    id = 0xff;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_ERROR_INTERNAL);
    CHECK(valid_command && id == 0);

    firmware_status = 3;
    query_result = EREMOTEIO;
    id = 0xff;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_ERROR_INTERNAL);
    CHECK(valid_command && id == 0);

    query_result = 0;
    firmware_status = 0;
    CHECK(nvshmemt_mlx5_query_dct_q_counter(mock_query, &dct, &id) == NVSHMEMX_SUCCESS);
    CHECK(valid_command && id == response_counter && query_calls == 8);

    nvshmemt_mlx5_devx_qp_query_fn query_qp = nullptr;
    void *handle = nullptr;
    CHECK(nvshmemt_mlx5_q_counter_init(nullptr, &query_qp) == NVSHMEMX_ERROR_INVALID_VALUE);
    CHECK(nvshmemt_mlx5_q_counter_init(&handle, nullptr) == NVSHMEMX_ERROR_INVALID_VALUE);
    int status = nvshmemt_mlx5_q_counter_init(&handle, &query_qp);
    CHECK(status == NVSHMEMX_SUCCESS || status == NVSHMEMX_ERROR_NOT_SUPPORTED);
    CHECK(status == NVSHMEMX_SUCCESS ? (handle && query_qp) : (!handle && !query_qp));
    nvshmemt_mlx5_q_counter_fini(&handle, &query_qp);
    CHECK(!handle && !query_qp);
    nvshmemt_mlx5_q_counter_fini(&handle, &query_qp);
    nvshmemt_mlx5_q_counter_fini(nullptr, nullptr);
    return 0;
}
#endif

int main() {
    if (test_retry_policy()) return 1;
#ifdef NVSHMEM_TEST_MLX5_QUERY
    if (test_queries()) return 1;
    printf("IBGDA Q-Counter layout, retry policy and mocked DCT query checks passed\n");
#else
    printf("IBGDA Q-Counter layout and retry policy checks passed (query checks not built)\n");
#endif
    return 0;
}
