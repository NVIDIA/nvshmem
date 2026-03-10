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
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.base_dsl.typing import cast as cute_cast

__all__ = [
    "atomic_inc",
    "atomic_fetch_inc",
    "atomic_fetch",
    "atomic_set",
    "atomic_add",
    "atomic_fetch_add",
    "atomic_and",
    "atomic_fetch_and",
    "atomic_or",
    "atomic_fetch_or",
    "atomic_xor",
    "atomic_fetch_xor",
    "atomic_swap",
    "atomic_compare_swap",
]


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@cute.jit
def _resolve_dtype(dst):
    return dst.dtype


@cute.jit
def atomic_fetch(src, pe):
    src_ptr = _resolve_ptr(src)
    dtype = _resolve_dtype(src)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch(src_ptr, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch(src_ptr, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch(src_ptr, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch(src_ptr, pe)


    elif const_expr(dtype == cutlass.Float32):
        return float_atomic_fetch(src_ptr, pe)


    elif const_expr(dtype == cutlass.Float64):
        return double_atomic_fetch(src_ptr, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch: {dtype}")


@cute.jit
def atomic_set(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_set(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_set(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_set(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_set(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Float32):
        return float_atomic_set(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Float64):
        return double_atomic_set(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_set: {dtype}")


@cute.jit
def atomic_compare_swap(dst, cond, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    cond = cute_cast(cond, dtype)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_compare_swap(dst_ptr, cond, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_compare_swap(dst_ptr, cond, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_compare_swap(dst_ptr, cond, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_compare_swap(dst_ptr, cond, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_compare_swap: {dtype}")


@cute.jit
def atomic_swap(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_swap(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_swap(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_swap(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_swap(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Float32):
        return float_atomic_swap(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Float64):
        return double_atomic_swap(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_swap: {dtype}")


@cute.jit
def atomic_fetch_inc(dst, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch_inc(dst_ptr, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch_inc: {dtype}")


@cute.jit
def atomic_inc(dst, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_inc(dst_ptr, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_inc(dst_ptr, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_inc: {dtype}")


@cute.jit
def atomic_fetch_add(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch_add(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch_add: {dtype}")


@cute.jit
def atomic_add(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_add(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_add(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_add: {dtype}")


@cute.jit
def atomic_and(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_and(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_and: {dtype}")


@cute.jit
def atomic_fetch_and(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch_and(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch_and(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch_and: {dtype}")


@cute.jit
def atomic_or(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_or(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_or: {dtype}")


@cute.jit
def atomic_fetch_or(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch_or(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch_or(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch_or: {dtype}")


@cute.jit
def atomic_xor(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_xor(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_xor: {dtype}")


@cute.jit
def atomic_fetch_xor(dst, value, pe):
    dst_ptr = _resolve_ptr(dst)
    dtype = _resolve_dtype(dst)
    value = cute_cast(value, dtype)


    if const_expr(dtype == cutlass.Int32):
        return int32_atomic_fetch_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Int64):
        return long_atomic_fetch_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint32):
        return uint32_atomic_fetch_xor(dst_ptr, value, pe)


    elif const_expr(dtype == cutlass.Uint64):
        return uint64_atomic_fetch_xor(dst_ptr, value, pe)


    raise RuntimeError(f"Unsupported CuTe dtype for atomic_fetch_xor: {dtype}")