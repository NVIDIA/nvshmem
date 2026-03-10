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

import cutlass
from cutlass import cute
from cutlass.cute.typing import dtype as cute_dtype
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.base_dsl.typing import cast as cute_cast

__all__ = ["p", "g", "put", "get", "put_nbi", "get_nbi", "put_block", "get_block", "put_nbi_block", "get_nbi_block", "put_warp", "get_warp", "put_nbi_warp", "get_nbi_warp", "put_signal_block", "put_signal", "put_signal_nbi", "put_signal_warp", "put_signal_nbi_block", "put_signal_nbi_warp"]


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@cute.jit
def _size_of(obj):
    return cute.size(obj)


@cute.jit
def _resolve_nelems(dst, src):
    dst_nelems = _size_of(dst)
    src_nelems = _size_of(src)
    return cute_cast(dst_nelems if dst_nelems < src_nelems else src_nelems, cutlass.Uint64)


@cute.jit
def _resolve_dtype(dst, src=None):
    return dst.dtype


# put variations



@cute.jit
def put_block(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_block(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_nbi_block(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_nbi_block(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def put_warp(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_warp(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_nbi_warp(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def put(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_nbi(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_nbi(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")



# get variations



@cute.jit
def get_block(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get_block(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def get_nbi_block(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get_nbi_block(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def get_warp(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get_warp(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def get_nbi_warp(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get_nbi_warp(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def get(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def get_nbi(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelems = _resolve_nelems(dst, src)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_get_nbi(dst_ptr, src_ptr, nelems, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_get_nbi(dst_ptr, src_ptr, nelems, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")



# put_signal variations



@cute.jit
def put_signal_block(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_signal_nbi_block(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal_nbi_block(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def put_signal_warp(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_signal_nbi_warp(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal_nbi_warp(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")




@cute.jit
def put_signal(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")


@cute.jit
def put_signal_nbi(dst, src, signal_var, signal_val, signal_op, pe):
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    signal_var_ptr = _resolve_ptr(signal_var)
    nelems = _resolve_nelems(dst, src)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    dtype = _resolve_dtype(dst, src)

    if const_expr(dtype == cutlass.Int8):
        return int8_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_put_signal_nbi(dst_ptr, src_ptr, nelems, signal_var_ptr, signal_val, signal_op, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")



# p variations

@cute.jit
def p(dst, src, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_p(dst_ptr, cute_cast(src, cutlass.Int8), pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_p(dst_ptr, cute_cast(src, cutlass.Int16), pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_p(dst_ptr, cute_cast(src, cutlass.Int32), pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_p(dst_ptr, cute_cast(src, cutlass.Int64), pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_p(dst_ptr, cute_cast(src, cutlass.Uint8), pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_p(dst_ptr, cute_cast(src, cutlass.Uint16), pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_p(dst_ptr, cute_cast(src, cutlass.Uint32), pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_p(dst_ptr, cute_cast(src, cutlass.Uint64), pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_p(dst_ptr, cute_cast(src, cutlass.Float32), pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_p(dst_ptr, cute_cast(src, cutlass.Float64), pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_p(dst_ptr, cute_cast(src, cutlass.Float16), pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")

# g variations

@cute.jit
def g(src, pe):
    src_ptr = _resolve_ptr(src)
    pe = cute_cast(pe, cutlass.Int32)
    dtype = _resolve_dtype(src)

    if const_expr(dtype == cutlass.Int8):
        return int8_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Int16):
        return int16_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Int32):
        return int32_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Int64):
        return int64_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Float32):
        return float_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Float64):
        return double_g(src_ptr, pe)

    elif const_expr(dtype == cutlass.Float16):
        return half_g(src_ptr, pe)

    raise RuntimeError(f"Unsupported CuTe dtype for RMA dispatch: {dtype}")

