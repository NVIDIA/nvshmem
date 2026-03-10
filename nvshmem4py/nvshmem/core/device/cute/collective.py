# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

from nvshmem.bindings.device.cute import *
from nvshmem.bindings.device.cute import (
    barrier as _nvshmem_barrier,
    barrier_block as _nvshmem_barrier_block,
    barrier_warp as _nvshmem_barrier_warp,
)

import nvshmem.core
import cutlass
from cutlass import cute
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.base_dsl.typing import cast as cute_cast

__all__ = [
    "sync_block",
    "sync_warp",
    "sync",
    "sync_all",
    "sync_all_block",
    "sync_all_warp",
    "barrier",
    "barrier_block",
    "barrier_warp",
    "barrier_all",
    "barrier_all_block",
    "barrier_all_warp",
    "reduce",
    "reduce_block",
    "reduce_warp",
    "reducescatter",
    "reducescatter_block",
    "reducescatter_warp",
    "fcollect",
    "fcollect_block",
    "fcollect_warp",
    "broadcast",
    "broadcast_block",
    "broadcast_warp",
    "alltoall",
    "alltoall_block",
    "alltoall_warp",
]


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@cute.jit
def _size_of(obj):
    return cute.size(obj)


@cute.jit
def _resolve_dtype(dst):
    return dst.dtype


@cute.jit
def _resolve_team(team):
    return cute_cast(team, cutlass.Int32)


@cute.jit
def _resolve_root(root):
    return cute_cast(root, cutlass.Int32)


@cute.jit
def _resolve_nelems_dst(dst):
    return cute_cast(_size_of(dst), cutlass.Uint64)


@cute.jit
def _resolve_nelems_src(src):
    return cute_cast(_size_of(src), cutlass.Uint64)


@cute.jit
def _resolve_nelems_min(dst, src):
    dst_nelems = _size_of(dst)
    src_nelems = _size_of(src)
    return cute_cast(dst_nelems if dst_nelems < src_nelems else src_nelems, cutlass.Uint64)


@cute.jit
def _resolve_nelems_alltoall(src, team):
    src_nelems = cute_cast(_size_of(src), cutlass.Uint64)
    team_size = cute_cast(team_n_pes(team), cutlass.Uint64)
    return cute_cast(src_nelems // team_size, cutlass.Uint64)


# sync variations

@cute.jit
def sync_block(team):
    team = _resolve_team(team)
    return team_sync_block(team)

@cute.jit
def sync_warp(team):
    team = _resolve_team(team)
    return team_sync_warp(team)

@cute.jit
def sync(team):
    team = _resolve_team(team)
    return team_sync(team)


# sync_all variations

@cute.jit
def sync_all_block():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync_block(team)

@cute.jit
def sync_all_warp():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync_warp(team)

@cute.jit
def sync_all():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync(team)


# barrier variations

@cute.jit
def barrier_block(team):
    team = _resolve_team(team)
    return _nvshmem_barrier_block(team)

@cute.jit
def barrier_warp(team):
    team = _resolve_team(team)
    return _nvshmem_barrier_warp(team)

@cute.jit
def barrier(team):
    team = _resolve_team(team)
    return _nvshmem_barrier(team)


# barrier_all variations

@cute.jit
def barrier_all_block():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier_block(team)

@cute.jit
def barrier_all_warp():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier_warp(team)

@cute.jit
def barrier_all():
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier(team)


# reduce variations

@cute.jit
def reduce_block(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce_block(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce_block(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")

@cute.jit
def reduce_warp(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")

@cute.jit
def reduce(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")


# reducescatter variations

@cute.jit
def reducescatter_block(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")

@cute.jit
def reducescatter_warp(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")

@cute.jit
def reducescatter(team, dst, src, op):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")


# fcollect variations

@cute.jit
def fcollect_block(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect_block(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")

@cute.jit
def fcollect_warp(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")

@cute.jit
def fcollect(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")


# broadcast variations

@cute.jit
def broadcast_block(team, dst, src, root=0):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")

@cute.jit
def broadcast_warp(team, dst, src, root=0):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")

@cute.jit
def broadcast(team, dst, src, root=0):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")


# alltoall variations

@cute.jit
def alltoall_block(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall_block(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")

@cute.jit
def alltoall_warp(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")

@cute.jit
def alltoall(team, dst, src):
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")
