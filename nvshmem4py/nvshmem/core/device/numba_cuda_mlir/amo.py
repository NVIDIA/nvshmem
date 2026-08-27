# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [ "atomic_inc", "atomic_fetch_inc", "atomic_fetch", "atomic_set", "atomic_add", "atomic_fetch_add", "atomic_and", "atomic_fetch_and", "atomic_or", "atomic_fetch_or", "atomic_xor", "atomic_fetch_xor", "atomic_swap", "atomic_compare_swap" ]

from nvshmem.bindings.device.numba_cuda_mlir import *

import cffi
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, float16, Array
# # bfloat16 is not currently exposed by numba_cuda_mlir.types

ffi = cffi.FFI()

def atomic_fetch(src, pe):
    """
    Fetches the current value at symmetric ``src`` on PE ``pe``.

    Args:
        - ``src`` (``Array``): Symmetric source array on remote PE.
        - ``pe`` (``int``): PE to fetch from.

    Returns:
        Current value stored at ``src`` on PE ``pe``.
    """
    pass
@overload(atomic_fetch, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch(src, pe):
    if src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return float_atomic_fetch(src_ptr, int32(pe))
        return impl
    elif src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return double_atomic_fetch(src_ptr, int32(pe))
        return impl
    elif src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return size_atomic_fetch(src_ptr, int32(pe))
        return impl
    elif src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return uint32_atomic_fetch(src_ptr, int32(pe))
        return impl
    elif src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return int_atomic_fetch(src_ptr, int32(pe))
        return impl
    elif src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            src_ptr = ffi.from_buffer(src)
            return long_atomic_fetch(src_ptr, int32(pe))
        return impl

def atomic_set(dst, value, pe):
    """
    Sets the value at symmetric ``dst`` on PE ``pe`` to ``value``.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to set.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_set, inline="always", typing_registry=typing_registry)
def overload_atomic_set(dst, value, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            float_atomic_set(dstptr, float32(value), int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            double_atomic_set(dstptr, float64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_set(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_set(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_set(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_set(dstptr, int64(value), int32(pe))
        return impl

def atomic_compare_swap(dst, cond, value, pe):
    """
    Atomically compares the current value at ``dst`` with ``cond`` and swaps with ``value`` if equal.

    Args:
        - ``dst`` (``Array``): Symmetric destination location on remote PE.
        - ``cond``: Comparison value.
        - ``value``: Replacement value.
        - ``pe`` (``int``): Target PE.

    Returns:
        The old value previously stored at ``dst``.
    """
    pass
@overload(atomic_compare_swap, inline="always", typing_registry=typing_registry)
def overload_atomic_compare_swap(dst, cond, value, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return float_atomic_compare_swap(dstptr, float32(cond), float32(value), int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return double_atomic_compare_swap(dstptr, float64(cond), float64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_compare_swap(dstptr, uint64(cond), uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_compare_swap(dstptr, uint32(cond), uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_compare_swap(dstptr, int32(cond), int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, cond, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_compare_swap(dstptr, int64(cond), int64(value), int32(pe))
        return impl

def atomic_swap(dst, value, pe):
    """
    Atomically swaps the current value at ``dst`` with ``value`` on PE ``pe``.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Replacement value.
        - ``pe`` (``int``): Target PE.

    Returns:
        The old value previously stored at ``dst``.
    """
    pass
@overload(atomic_swap, inline="always", typing_registry=typing_registry)
def overload_atomic_swap(dst, value, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return float_atomic_swap(dstptr, float32(value), int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return double_atomic_swap(dstptr, float64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_swap(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_swap(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_swap(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_swap(dstptr, int64(value), int32(pe))
        return impl

def atomic_fetch_inc(dst, pe):
    """
    Atomically increments ``dst`` by 1 and returns the old value.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` prior to increment.
    """
    pass
@overload(atomic_fetch_inc, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch_inc(dst, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return float_atomic_fetch_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return double_atomic_fetch_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_fetch_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_fetch_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_fetch_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_fetch_inc(dstptr, int32(pe))
        return impl

def atomic_inc(dst, pe):
    """
    Atomically increments ``dst`` by 1 (no return value).

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_inc, inline="always", typing_registry=typing_registry)
def overload_atomic_inc(dst, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            float_atomic_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            double_atomic_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_inc(dstptr, int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_inc(dstptr, int32(pe))
        return impl

def atomic_fetch_add(dst, value, pe):
    """
    Atomically adds ``value`` to ``dst`` and returns the old value.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to add.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` prior to the addition.
    """
    pass
@overload(atomic_fetch_add, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch_add(dst, value, pe):
    if dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return float_atomic_fetch_add(dstptr, float32(value), int32(pe))
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return double_atomic_fetch_add(dstptr, float64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_fetch_add(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_fetch_add(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_fetch_add(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_fetch_add(dstptr, int64(value), int32(pe))
        return impl

def atomic_add(dst, value, pe):
    """
    Atomically adds ``value`` to ``dst`` (no return value).

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to add.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_add, inline="always", typing_registry=typing_registry)
def overload_atomic_add(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_add(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_add(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_add(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_add(dstptr, int64(value), int32(pe))
        return impl

def atomic_and(dst, value, pe):
    """
    Atomically applies bitwise AND of ``value`` with ``dst`` (no return value).

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to AND with.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_and, inline="always", typing_registry=typing_registry)
def overload_atomic_and(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_and(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_and(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_and(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_and(dstptr, int64(value), int32(pe))
        return impl

def atomic_fetch_and(dst, value, pe):
    """
    Atomically applies bitwise AND and returns the old value.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to AND with.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` prior to the AND.
    """
    pass
@overload(atomic_fetch_and, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch_and(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_fetch_and(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_fetch_and(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_fetch_and(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_fetch_and(dstptr, int64(value), int32(pe))
        return impl

def atomic_or(dst, value, pe):
    """
    Atomically applies bitwise OR of ``value`` with ``dst`` (no return value).

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to OR with.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_or, inline="always", typing_registry=typing_registry)
def overload_atomic_or(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_or(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_or(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_or(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_or(dstptr, int64(value), int32(pe))
        return impl

def atomic_fetch_or(dst, value, pe):
    """
    Atomically applies bitwise OR and returns the old value.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to OR with.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` prior to the OR.
    """
    pass
@overload(atomic_fetch_or, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch_or(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_fetch_or(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_fetch_or(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_fetch_or(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_fetch_or(dstptr, int64(value), int32(pe))
        return impl

def atomic_xor(dst, value, pe):
    """
    Atomically applies bitwise XOR of ``value`` with ``dst`` (no return value).

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to XOR with.
        - ``pe`` (``int``): Target PE.
    """
    pass
@overload(atomic_xor, inline="always", typing_registry=typing_registry)
def overload_atomic_xor(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            size_atomic_xor(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_atomic_xor(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            int_atomic_xor(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            long_atomic_xor(dstptr, int64(value), int32(pe))
        return impl

def atomic_fetch_xor(dst, value, pe):
    """
    Atomically applies bitwise XOR and returns the old value.

    Args:
        - ``dst`` (``Array``): Symmetric destination array on remote PE.
        - ``value``: Value to XOR with.
        - ``pe`` (``int``): Target PE.

    Returns:
        The value stored at ``dst`` prior to the XOR.
    """
    pass
@overload(atomic_fetch_xor, inline="always", typing_registry=typing_registry)
def overload_atomic_fetch_xor(dst, value, pe):
    if dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return size_atomic_fetch_xor(dstptr, uint64(value), int32(pe))
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return uint32_atomic_fetch_xor(dstptr, uint32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return int_atomic_fetch_xor(dstptr, int32(value), int32(pe))
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, value, pe):
            dstptr = ffi.from_buffer(dst)
            return long_atomic_fetch_xor(dstptr, int64(value), int32(pe))
        return impl