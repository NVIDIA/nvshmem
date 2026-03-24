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
    return dst.element_type


@cute.jit
def atomic_fetch(src, pe):
    """
    Atomically fetches (reads) the current value at symmetric ``src`` on PE ``pe``.

    This is a thread-level remote atomic operation. The read is performed atomically
    with respect to other atomic operations on the same location.

    Args:
        - ``src``: CuTe tensor view pointing to a single-element symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched.
        - ``pe`` (``int``): Source PE to fetch from.

    Returns:
        The current value stored at ``src`` on PE ``pe``, with the same dtype as ``src``.

    Note:
        Supported dtypes are determined by the NVSHMEM atomic fetch dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically sets the value at symmetric ``dst`` on PE ``pe`` to ``value``.

    This is a thread-level remote atomic store. The write is performed atomically
    with respect to other atomic operations on the same location.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched.
        - ``value``: The value to store. Cast to the element dtype of ``dst`` before the operation.
        - ``pe`` (``int``): Target PE.

    Note:
        This operation does not return the old value. Use ``atomic_fetch`` before setting
        if you need the previous value.
        Supported dtypes are determined by the NVSHMEM atomic set dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically compares the value at symmetric ``dst`` on PE ``pe`` with ``cond``,
    and if equal, replaces it with ``value``. Returns the old value regardless.

    This is a thread-level remote atomic compare-and-swap (CAS) operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched.
        - ``cond``: Comparison value. Cast to the element dtype of ``dst``.
          The swap only occurs if the current value at ``dst`` equals ``cond``.
        - ``value``: Replacement value. Cast to the element dtype of ``dst``.
          Written to ``dst`` only if the comparison succeeds.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the operation, regardless of
        whether the swap occurred.

    Note:
        Supported dtypes are determined by the NVSHMEM atomic compare-swap dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically replaces the value at symmetric ``dst`` on PE ``pe`` with ``value``,
    and returns the old value.

    This is a thread-level remote atomic swap operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched.
        - ``value``: New value to store. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the swap.

    Note:
        Supported dtypes are determined by the NVSHMEM atomic swap dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically increments the value at symmetric ``dst`` on PE ``pe`` by 1,
    and returns the value prior to the increment.

    This is a thread-level remote atomic fetch-and-increment operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Supported dtypes are integral types
          as determined by the NVSHMEM atomic fetch-inc dispatch table.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the increment.

    Note:
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically increments the value at symmetric ``dst`` on PE ``pe`` by 1.
    Does not return the old value.

    This is a thread-level remote atomic increment operation (non-fetching variant).
    Use ``atomic_fetch_inc`` if you need the value before the increment.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Supported dtypes are integral types
          as determined by the NVSHMEM atomic inc dispatch table.
        - ``pe`` (``int``): Target PE.

    Note:
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically adds ``value`` to the value at symmetric ``dst`` on PE ``pe``,
    and returns the value prior to the addition.

    This is a thread-level remote atomic fetch-and-add operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched (integral and floating-point types supported).
        - ``value``: Value to add. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the addition.

    Note:
        Supported dtypes are determined by the NVSHMEM atomic fetch-add dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically adds ``value`` to the value at symmetric ``dst`` on PE ``pe``.
    Does not return the old value.

    This is a thread-level remote atomic add operation (non-fetching variant).
    Use ``atomic_fetch_add`` if you need the value before the addition.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines
          which underlying NVSHMEM atomic is dispatched (integral and floating-point types supported).
        - ``value``: Value to add. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Note:
        Supported dtypes are determined by the NVSHMEM atomic add dispatch table.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise AND of ``value`` with the value at symmetric ``dst`` on PE ``pe``.
    Does not return the old value.

    This is a thread-level remote atomic bitwise AND operation (non-fetching variant).
    Use ``atomic_fetch_and`` if you need the value before the operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to AND with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise AND of ``value`` with the value at symmetric ``dst`` on PE ``pe``,
    and returns the value prior to the operation.

    This is a thread-level remote atomic fetch-and-AND operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to AND with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the AND operation.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise OR of ``value`` with the value at symmetric ``dst`` on PE ``pe``.
    Does not return the old value.

    This is a thread-level remote atomic bitwise OR operation (non-fetching variant).
    Use ``atomic_fetch_or`` if you need the value before the operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to OR with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise OR of ``value`` with the value at symmetric ``dst`` on PE ``pe``,
    and returns the value prior to the operation.

    This is a thread-level remote atomic fetch-and-OR operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to OR with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the OR operation.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise XOR of ``value`` with the value at symmetric ``dst`` on PE ``pe``.
    Does not return the old value.

    This is a thread-level remote atomic bitwise XOR operation (non-fetching variant).
    Use ``atomic_fetch_xor`` if you need the value before the operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to XOR with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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
    """
    Atomically applies bitwise XOR of ``value`` with the value at symmetric ``dst`` on PE ``pe``,
    and returns the value prior to the operation.

    This is a thread-level remote atomic fetch-and-XOR operation.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
        - ``value``: Mask value to XOR with. Cast to the element dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` on PE ``pe`` prior to the XOR operation.

    Note:
        Only integral dtypes (e.g., ``uint32``, ``uint64``, etc.) are supported.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
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