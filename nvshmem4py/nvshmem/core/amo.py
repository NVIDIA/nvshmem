# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level host-initiated atomic memory operations (AMOs).

Targets must refer to symmetric NVSHMEM memory. CuPy and PyTorch targets are
checked against NVSHMEM4Py's allocation tracking; callers supplying a raw
``cuda.core.Buffer`` are responsible for ensuring that requirement themselves.
Each AMO operates on one scalar; ``offset`` may select that scalar within a
larger symmetric allocation.
"""

import operator
from typing import Any, Tuple

from cuda.core import Buffer

import nvshmem.bindings as bindings
from nvshmem.core._internal_tracking import _is_initialized, InternalInitStatus
from nvshmem.core.collective import _check_dtype, external_to_nvshmem_dtypes
from nvshmem.core.nvshmem_types import NvshmemInvalid
from nvshmem.core.utils import _get_device, dtype_nbytes

__all__ = [
    "atomic_fetch",
    "atomic_set",
    "atomic_inc",
    "atomic_fetch_inc",
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


def _target_buffer_and_dtype(target: object, dtype: str = None) -> Tuple[Buffer, int, str]:
    """Resolve an AMO target using the existing high-level dtype conversion."""
    if isinstance(target, Buffer):
        if dtype is None:
            raise NvshmemInvalid("dtype is required when the AMO target is a raw Buffer")
        dtype = str(dtype)
        return target, int(target.size), external_to_nvshmem_dtypes.get(dtype, dtype)

    target_buffer, target_size, target_dtype = _check_dtype(target)
    if dtype is not None:
        requested_dtype = external_to_nvshmem_dtypes.get(str(dtype), str(dtype))
        if requested_dtype != target_dtype:
            raise NvshmemInvalid("The supplied dtype does not match the AMO target dtype")
    return target_buffer, target_size, target_dtype


def _target_pointer(target_buffer: Buffer, target_size: int, target_dtype: str, offset: int) -> int:
    """Return an element-addressed pointer after checking the selected location."""
    if isinstance(offset, bool):
        raise NvshmemInvalid("AMO offset must be a non-negative integer")
    try:
        offset = operator.index(offset)
    except TypeError as exc:
        raise NvshmemInvalid("AMO offset must be a non-negative integer") from exc
    if offset < 0:
        raise NvshmemInvalid("AMO offset must be a non-negative integer")

    try:
        item_size = dtype_nbytes(target_dtype)
    except ValueError as exc:
        raise NvshmemInvalid(f"Host AMOs do not support dtype {target_dtype!r}") from exc
    byte_offset = offset * item_size
    if byte_offset + item_size > target_size:
        raise NvshmemInvalid("AMO target is too small for the requested offset")
    return int(target_buffer.handle) + byte_offset


def _call_atomic(operation: str, target: object, *values: Any, remote_pe: int = -1, offset: int = 0, dtype: Any = None):
    """Dispatch an AMO to the generated native binding for the target dtype."""
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    target_buffer, target_size, target_dtype = _target_buffer_and_dtype(target, dtype)
    target_ptr = _target_pointer(target_buffer, target_size, target_dtype, offset)
    binding_name = f"{target_dtype}_atomic_{operation}"
    try:
        atomic_function = getattr(bindings, binding_name)
    except AttributeError as exc:
        raise NvshmemInvalid(f"atomic_{operation} does not support dtype {target_dtype}") from exc

    _, other_dev = _get_device()
    try:
        return atomic_function(target_ptr, *values, remote_pe)
    finally:
        if other_dev is not None:
            other_dev.set_current()


def atomic_fetch(target: object, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Fetch a value from a symmetric ``target`` on ``remote_pe``.

    ``target`` may be an NVSHMEM-backed CuPy array or PyTorch tensor. For a
    raw :class:`cuda.core.Buffer`, pass the NVSHMEM dtype via ``dtype``. The
    optional ``offset`` selects an element within the target.
    """
    return _call_atomic("fetch", target, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_set(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically store ``value`` at a symmetric ``target`` on ``remote_pe``."""
    _call_atomic("set", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_inc(target: object, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically increment an integral symmetric ``target`` on ``remote_pe``."""
    _call_atomic("inc", target, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_fetch_inc(target: object, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically increment an integral target and return its prior value."""
    return _call_atomic("fetch_inc", target, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_add(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically add ``value`` to a symmetric ``target`` on ``remote_pe``."""
    _call_atomic("add", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_fetch_add(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically add ``value`` and return the target's prior value."""
    return _call_atomic("fetch_add", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_and(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically bitwise-AND ``value`` with an integral symmetric target."""
    _call_atomic("and", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_fetch_and(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically bitwise-AND ``value`` and return the target's prior value."""
    return _call_atomic("fetch_and", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_or(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically bitwise-OR ``value`` with an integral symmetric target."""
    _call_atomic("or", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_fetch_or(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically bitwise-OR ``value`` and return the target's prior value."""
    return _call_atomic("fetch_or", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_xor(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None) -> None:
    """Atomically bitwise-XOR ``value`` with an integral symmetric target."""
    _call_atomic("xor", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_fetch_xor(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically bitwise-XOR ``value`` and return the target's prior value."""
    return _call_atomic("fetch_xor", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_swap(target: object, value: Any, remote_pe: int = -1, *, offset: int = 0, dtype: Any = None):
    """Atomically replace a target with ``value`` and return its prior value."""
    return _call_atomic("swap", target, value, remote_pe=remote_pe, offset=offset, dtype=dtype)


def atomic_compare_swap(target: object,
                        cond: Any,
                        value: Any,
                        remote_pe: int = -1,
                        *,
                        offset: int = 0,
                        dtype: Any = None):
    """Atomically compare an integral target with ``cond`` and conditionally swap it.

    The prior target value is returned whether or not the comparison succeeds.
    """
    return _call_atomic("compare_swap", target, cond, value, remote_pe=remote_pe, offset=offset, dtype=dtype)
