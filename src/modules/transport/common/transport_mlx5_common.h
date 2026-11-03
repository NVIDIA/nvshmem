/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_MLX5_COMMON_H
#define _TRANSPORT_MLX5_COMMON_H

#include <errno.h>   // for EINVAL, EOPNOTSUPP, EREMOTEIO
#include <stddef.h>  // for size_t
#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>

struct ibv_context;
struct ibv_qp;

/* Only synchronous request rejection permits a state-confirmed counter-free retry. */
inline bool nvshmemt_mlx5_q_counter_can_retry(int status, uint32_t firmware_status) {
    if (!status) return firmware_status != 0;
    if (status == EINVAL || status == -EINVAL || status == EOPNOTSUPP || status == -EOPNOTSUPP) {
        return true;
    }
#ifdef EREMOTEIO
    if (status == EREMOTEIO || status == -EREMOTEIO) return true;
#endif
    return false;
}

/* Query and retry callbacks operate on the same unpublished QP under exclusive control.
 * The retry callback reports API/firmware success. This helper returns whether it was attempted. */
template <typename QueryReset, typename Retry>
inline bool nvshmemt_mlx5_q_counter_retry(int status, uint32_t firmware_status, uint8_t qp_type,
                                          uint8_t &disabled_qp_types, QueryReset query_reset,
                                          Retry retry) {
    if (!nvshmemt_mlx5_q_counter_can_retry(status, firmware_status) || !query_reset()) return false;
    if (retry()) disabled_qp_types |= qp_type;
    return true;
}

/* Load this optional Verbs-object query independently of Data Direct support. */
using nvshmemt_mlx5_devx_qp_query_fn = int (*)(struct ibv_qp *, const void *, size_t, void *,
                                               size_t);
int nvshmemt_mlx5_q_counter_init(void **mlx5_handle, nvshmemt_mlx5_devx_qp_query_fn *query_qp);
void nvshmemt_mlx5_q_counter_fini(void **mlx5_handle, nvshmemt_mlx5_devx_qp_query_fn *query_qp);
/* The Verbs DCT must have completed INIT->RTR. The returned ID is borrowed from the driver. */
int nvshmemt_mlx5_query_dct_q_counter(nvshmemt_mlx5_devx_qp_query_fn query_qp, struct ibv_qp *dct,
                                      uint8_t *counter_set_id);

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context);
int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context);
int nvshmemt_ib_common_check_nic_ext_atomic_support(struct ibv_context *context);

/* These values are not defined on all systems.
 * However, they can be traced back to a kernel enum with
 * these values.
 */
#ifndef MLX5DV_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_BF 0x0
#endif

#ifndef MLX5DV_UAR_ALLOC_TYPE_NC
#define MLX5DV_UAR_ALLOC_TYPE_NC 0x1
#endif

enum {
    MLX5_ATOMIC_CAP_OP_SUPPORT_CAS = 0x1,
    MLX5_ATOMIC_CAP_OP_SUPPORT_FA = 0x2,
    MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_CAS = 0x4,
    MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_FA = 0x8,
};

enum {
    MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B = 0x4,
    MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B = 0x8,
};

#endif
