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
    return dst.element_type


# put variations





@cute.jit
def put_block(dst, src, pe):
    """
    Copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: local data movement completes before the call returns.
    """
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
    """
    Non-blockingly copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: local data movement completes before the call returns.
    """
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
    """
    Non-blockingly copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a thread-level operation.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: local data movement completes before the call returns.
    """
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
    """
    Non-blockingly copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a thread-level operation.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``pe`` (``int``): Target PE to copy to.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: data is available in ``dst`` before the call returns.
    """
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
    """
    Non-blockingly copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: data is available in ``dst`` before the call returns.
    """
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
    """
    Non-blockingly copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a thread-level operation.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: data is available in ``dst`` before the call returns.
    """
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
    """
    Non-blockingly copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a thread-level operation.

    Args:
        - ``dst``: CuTe tensor view pointing to the local destination on this PE.
        - ``src``: CuTe tensor view pointing to the symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
        - ``pe`` (``int``): Source PE to copy from.

    Note:
        The number of elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: the transfer may not be complete when the call returns.
        Use a fence or synchronization primitive to ensure completion.
    """
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
    """
    Puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: the data transfer and signal update complete before the call returns.
    """
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
    """
    Non-blockingly puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: neither the data transfer nor the signal update
        are guaranteed to be visible to the remote PE when the call returns.
    """
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
    """
    Puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: the data transfer and signal update complete before the call returns.
    """
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
    """
    Non-blockingly puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: neither the data transfer nor the signal update
        are guaranteed to be visible to the remote PE when the call returns.
    """
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
    """
    Puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a thread-level operation.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a blocking operation: the data transfer and signal update complete before the call returns.
    """
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
    """
    Non-blockingly puts data from ``src`` to symmetric ``dst`` on PE ``pe``, then signals ``signal_var``.
    This is a thread-level operation.

    The signal operation atomically updates ``signal_var`` on the remote PE after the data transfer,
    allowing the remote PE to detect transfer completion via a signal variable.

    Args:
        - ``dst``: CuTe tensor view pointing to the symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``src``: CuTe tensor view pointing to the local source data on this PE.
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal variable (dtype ``uint64``)
          on PE ``pe``. Must be a 1-element NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable. Cast to ``uint64``.
        - ``signal_op``: Signal operation type. Supported values are ``NVSHMEM_SIGNAL_SET``
          (set the signal to ``signal_val``) and ``NVSHMEM_SIGNAL_ADD``
          (atomically add ``signal_val`` to the signal variable).
        - ``pe`` (``int``): Target PE.

    Note:
        The number of data elements transferred is ``min(size(dst), size(src))``.
        ``dst`` and ``src`` must have the same element dtype.
        This is a non-blocking operation: neither the data transfer nor the signal update
        are guaranteed to be visible to the remote PE when the call returns.
    """
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
    """
    Writes a single scalar value ``src`` to the symmetric location ``dst`` on PE ``pe``.
    This is a thread-level point operation (scalar put).

    Unlike ``put``, which transfers an array of elements, ``p`` transfers exactly one scalar
    value. ``dst`` must point to a single-element symmetric location.

    Args:
        - ``dst``: CuTe tensor view pointing to a single-element symmetric destination on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor with exactly one element.
        - ``src``: Scalar value to write. The value is cast to the dtype of ``dst``.
        - ``pe`` (``int``): Target PE.

    Note:
        This is a blocking, thread-level operation.
        ``src`` is cast to the element dtype of ``dst`` before the transfer.
    """
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
    """
    Reads and returns a single scalar value from the symmetric location ``src`` on PE ``pe``.
    This is a thread-level get operation (scalar get).

    Unlike ``get``, which transfers an array of elements, ``g`` retrieves exactly one scalar
    value. ``src`` must point to a single-element symmetric location.

    Args:
        - ``src``: CuTe tensor view pointing to a single-element symmetric source on PE ``pe``.
          Must be a symmetric (NVSHMEM-allocated) tensor with exactly one element.
        - ``pe`` (``int``): Source PE to read from.

    Returns:
        The scalar value stored at ``src`` on PE ``pe``, with the same dtype as ``src``.

    Note:
        This is a blocking, thread-level operation. The returned value is immediately available.
    """
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

