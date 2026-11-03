/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <errno.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>

#include "transport_mlx5_common.h"
#include "transport_common.h"
#include "mlx5_ifc.h"
#include "non_abi/c/nvshmemi_error_macros.h"

/* IFC layout arrays describe bits, not the byte layout of command buffers. */
static_assert(offsetof(struct mlx5_ifc_dctc_bits, counter_set_id) == 0x60,
              "DCT counter_set_id must start at bit 0x60");
static_assert(sizeof(((struct mlx5_ifc_dctc_bits *)0)->counter_set_id) == 0x8,
              "DCT counter_set_id must be 8 bits");
static_assert(sizeof(struct mlx5_ifc_dctc_bits) == 0x380, "Preserve the existing DCTC layout");
static_assert(DEVX_ST_SZ_BYTES(query_dct_in) == 0x10, "Unexpected QUERY_DCT input size");
static_assert(DEVX_ST_SZ_BYTES(query_dct_out) == 0x80, "Unexpected QUERY_DCT output size");

int nvshmemt_mlx5_q_counter_init(void **mlx5_handle, nvshmemt_mlx5_devx_qp_query_fn *query_qp) {
    if (!mlx5_handle || !query_qp) return NVSHMEMX_ERROR_INVALID_VALUE;
    *mlx5_handle = nullptr;
    *query_qp = nullptr;

    void *handle = dlopen("libmlx5.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) handle = dlopen("libmlx5.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    LOAD_SYM(handle, "mlx5dv_devx_qp_query", *query_qp);
    if (!*query_qp) {
        dlclose(handle);
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }

    *mlx5_handle = handle;
    return NVSHMEMX_SUCCESS;
}

void nvshmemt_mlx5_q_counter_fini(void **mlx5_handle, nvshmemt_mlx5_devx_qp_query_fn *query_qp) {
    if (query_qp) *query_qp = nullptr;
    if (mlx5_handle && *mlx5_handle) {
        dlclose(*mlx5_handle);
        *mlx5_handle = nullptr;
    }
}

int nvshmemt_mlx5_query_dct_q_counter(nvshmemt_mlx5_devx_qp_query_fn query_qp, struct ibv_qp *dct,
                                      uint8_t *counter_set_id) {
    if (!counter_set_id) return NVSHMEMX_ERROR_INVALID_VALUE;
    *counter_set_id = 0;
    if (!dct) return NVSHMEMX_ERROR_INVALID_VALUE;
    if (!query_qp) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    uint32_t cmd_in[DEVX_ST_SZ_BYTES(query_dct_in) / sizeof(uint32_t)] = {0};
    uint32_t cmd_out[DEVX_ST_SZ_BYTES(query_dct_out) / sizeof(uint32_t)] = {0};
    DEVX_SET(query_dct_in, cmd_in, opcode, MLX5_CMD_OP_QUERY_DCT);
    DEVX_SET(query_dct_in, cmd_in, dctn, dct->qp_num);

    /* A DCT is a Verbs object: QUERY_QP and devx_obj_query are not substitutes here. */
    errno = 0;
    int status = query_qp(dct, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    int query_errno = errno;
    if (status || DEVX_GET(query_dct_out, cmd_out, status)) {
        NVSHMEMI_WARN_PRINT(
            "Q-Counter QUERY_DCT failed: dctn %u ret %d (%s) errno %d (%s) "
            "devx_status 0x%x syndrome 0x%x",
            dct->qp_num, status, nvshmemt_strerror_from_status(status), query_errno,
            nvshmemt_strerror_from_errno(query_errno), DEVX_GET(query_dct_out, cmd_out, status),
            DEVX_GET(query_dct_out, cmd_out, syndrome));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    void *dctc = DEVX_ADDR_OF(query_dct_out, cmd_out, dctc);
    uint8_t id = DEVX_GET(dctc, dctc, counter_set_id);
    if (!id) return NVSHMEMX_ERROR_NOT_SUPPORTED;
    *counter_set_id = id;
    return NVSHMEMX_SUCCESS;
}

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context) {
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));

    if (status == 0) {
        return true;
    }
    return false;
}

int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context) {
    void *cap;
    uint32_t amo_endianness_mode;
    uint32_t amo_endianness_mode_support;
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) | HCA_CAP_OPMOD_GET_CUR);
    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                "mlx5dv_devx_general_cmd for atomic caps failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability);
    amo_endianness_mode_support =
        DEVX_GET(atomic_caps, cap, supported_atomic_req_8B_endianness_mode_1);
    amo_endianness_mode = DEVX_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);
    if (amo_endianness_mode_support && amo_endianness_mode) {
        *endianness_mode = 8;
    } else {
        *endianness_mode = UINT32_MAX;
    }

out:
    return status;
}

int nvshmemt_ib_common_check_nic_ext_atomic_support(struct ibv_context *context) {
    int status = 0;

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };
    void *cap;
    uint16_t atomic_operations;
    uint16_t atomic_size_qp;
    uint16_t atomic_size_dc;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.atomic_caps);
    atomic_operations = DEVX_GET(atomic_caps, cap, atomic_operations);
    atomic_size_qp = DEVX_GET(atomic_caps, cap, atomic_size_qp);
    atomic_size_dc = DEVX_GET(atomic_caps, cap, atomic_size_dc);

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_CAS)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic compared and swap\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_FA)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic fetch and add\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_CAS)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic masked compared and swap\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_FA)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic masked fetch and add\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if ((!(atomic_size_qp & MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B)) ||
        (!(atomic_size_dc & MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B))) {
        NVSHMEMI_WARN_PRINT("device does not support 4B atomics\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if ((!(atomic_size_qp & MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B)) ||
        (!(atomic_size_dc & MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B))) {
        NVSHMEMI_WARN_PRINT("device does not support 8B atomics\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

out:
    return status;
}
