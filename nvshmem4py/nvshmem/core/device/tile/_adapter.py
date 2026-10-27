# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Helpers that the generated cuTile NVSHMEM bindings call.

They run while cuTile compiles a kernel, which represents each operand by a
proxy whose class is still ``ct.Array`` or ``ct.Tile``.  A branch on that class
under ``ct.static_eval`` is therefore a compile-time specialization, not a
device-side test.
"""

from __future__ import annotations

import math

import cuda.tile as ct


def as_scalar(value, dtype):
    """Return a Python number, enum member, or scalar Tile as a ``dtype`` scalar."""
    if ct.static_eval(isinstance(value, ct.Tile)):
        return ct.astype(value, dtype)
    return ct.full((), value, dtype=dtype)


def element_count(operand):
    """Return a Tile's or Array's element count as a ``uint64`` scalar.

    NVSHMEM transfers contiguous memory, so an Array operand must be
    contiguous.  Array extents are ``int32``; widen each before multiplying.
    """
    if ct.static_eval(isinstance(operand, ct.Tile)):
        return ct.full((), ct.static_eval(math.prod(operand.shape)), dtype=ct.uint64)
    count = ct.full((), 1, dtype=ct.uint64)
    for axis in ct.static_iter(range(operand.ndim)):
        count = count * ct.astype(operand.shape[axis], ct.uint64)
    return count


def byte_count(operand):
    """Return a Tile's or Array's size in bytes as a ``uint64`` scalar."""
    ct.static_assert(
        operand.dtype.bitwidth % 8 == 0,
        "NVSHMEM byte counts need a byte-addressable dtype",
    )
    return element_count(operand) * ct.static_eval(operand.dtype.bitwidth // 8)


def divide_count(count, divisor):
    """Return ``count // divisor`` for a ``uint64`` count that it divides.

    Divide as ``int64``: cuda.tile lowers ``//`` on an unsigned integer to a
    floor division that Tile IR rejects.  Counts fit in ``int64``.
    """
    return ct.astype(ct.astype(count, ct.int64) // divisor, ct.uint64)


def require_array(operand, name: str):
    """Require a ``ct.Array`` of any dtype; the TileLib declaration types its elements."""
    ct.static_assert(
        isinstance(operand, ct.Array),
        f"{name} must be a ct.Array",
    )
    return operand


def require_rank_one_tile(operand, name: str):
    """Require a rank-one shared-memory ``ct.Tile`` and return its static extent."""
    ct.static_assert(
        isinstance(operand, ct.Tile),
        f"{name} must be a rank-one ct.Tile",
    )
    ct.static_assert(
        operand.ndim == 1,
        f"{name} must be a rank-one ct.Tile",
    )
    return ct.static_eval(operand.shape[0])
