# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

from nvshmem.bindings.device.numba import *

import cffi
from numba.core.extending import overload
from numba.types import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, float16, Array
from numba.cuda.types import bfloat16

__all__ = ["p", "g", "put", "get", "put_nbi", "get_nbi", "put_block", "get_block", "put_nbi_block", "get_nbi_block", "put_warp", "get_warp", "put_nbi_warp", "get_nbi_warp", "put_signal_block", "put_signal", "put_signal_nbi", "put_signal_warp", "put_signal_nbi_block", "put_signal_nbi_warp", "put_signal_nbi_warp"]

# TODO: create a global ffi object for other high level bindings to use
ffi = cffi.FFI()

def put(src: Array, dst: Array, pe: int32) -> None:
    """
    Copies data from local ``src`` to symmetric ``dst`` on PE ``pe``. 
    This is a thread-level operation.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def get(dst: Array, src: Array, pe: int32) -> None:
    """
    Copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a thread-level operation.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def put_nbi(src: Array, dst: Array, pe: int32) -> None:
    """
    Non-blockingly copies data from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a thread-level operation.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def get_nbi(dst: Array, src: Array, pe: int32) -> None:
    """
    Non-blockingly copies data from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a thread-level operation.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def put_block(src: Array, dst: Array, pe: int32) -> None:
    """
    Copies from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def get_block(dst: Array, src: Array, pe: int32) -> None:
    """
    Copies from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def put_nbi_block(src: Array, dst: Array, pe: int32) -> None:
    """
    Non-blockingly copies from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    """
    pass

def get_nbi_block(dst: Array, src: Array, pe: int32) -> None:
    """
    Non-blockingly copies from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    """
    pass

def put_warp(src: Array, dst: Array, pe: int32) -> None:
    """
    Copies from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def get_warp(dst: Array, src: Array, pe: int32) -> None:
    """
    Copies from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def put_nbi_warp(src: Array, dst: Array, pe: int32) -> None:
    """
    Non-blockingly copies from local ``src`` to symmetric ``dst`` on PE ``pe``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``src`` (``Array``): Local source array on this PE to copy from.
        - ``dst`` (``Array``): Symmetric destination array on PE ``pe`` to copy to.
        - ``pe`` (``int``): PE to copy to.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

def get_nbi_warp(dst: Array, src: Array, pe: int32) -> None:
    """
    Non-blockingly copies from symmetric ``src`` on PE ``pe`` to local ``dst``.
    This is a warp-level operation. All threads in the warp must call this function with the same arguments.

    Args:
        - ``dst`` (``Array``): Local destination array on this PE to copy to.
        - ``src`` (``Array``): Symmetric source array from PE ``pe`` to copy from.
        - ``pe`` (``int``): PE to copy from.
    
    Note:
        When ``src`` and ``dst`` are of different sizes, only data of the smaller size is copied.
        ``src`` and ``dst`` must have the same data type.
    """
    pass

# put variations
@overload(put_block)
def put_block_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put_block(dstptr, srcptr, nelems, pe)
        return impl

@overload(put_nbi_block)
def put_nbi_block_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put_nbi_block(dstptr, srcptr, nelems, pe)
        return impl


@overload(put_warp)
def put_warp_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put_warp(dstptr, srcptr, nelems, pe)
        return impl

@overload(put_nbi_warp)
def put_nbi_warp_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl


@overload(put)
def put_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put(dstptr, srcptr, nelems, pe)
        return impl

@overload(put_nbi)
def put_nbi_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_put_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_put_nbi(dstptr, srcptr, nelems, pe)
        return impl



# get variations
@overload(get_block)
def get_block_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get_block(dstptr, srcptr, nelems, pe)
        return impl

@overload(get_nbi_block)
def get_nbi_block_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get_nbi_block(dstptr, srcptr, nelems, pe)
        return impl


@overload(get_warp)
def get_warp_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get_warp(dstptr, srcptr, nelems, pe)
        return impl

@overload(get_nbi_warp)
def get_nbi_warp_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get_nbi_warp(dstptr, srcptr, nelems, pe)
        return impl


@overload(get)
def get_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get(dstptr, srcptr, nelems, pe)
        return impl

@overload(get_nbi)
def get_nbi_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int8_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int16_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int32_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            int64_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint8_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint16_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint32_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            uint64_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            float_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            double_get_nbi(dstptr, srcptr, nelems, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            nelems = min(dst.size, src.size)
            half_get_nbi(dstptr, srcptr, nelems, pe)
        return impl



# put_signal variations
def put_signal_block():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a  manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal_block)
def put_signal_block_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl

def put_signal_nbi_block():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a _nbi manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal_nbi_block)
def put_signal_nbi_block_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal_nbi_block(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl


def put_signal_warp():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a  manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal_warp)
def put_signal_warp_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl

def put_signal_nbi_warp():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a _nbi manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal_nbi_warp)
def put_signal_nbi_warp_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal_nbi_warp(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl


def put_signal():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a  manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal)
def put_signal_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl

def put_signal_nbi():
    """
    Put data with a signal operation from ``src`` to ``dst`` on PE ``pe`` in a _nbi manner at a  scope. Device initiated.

    Signal variables must be an ``Array`` of dtype ``int64`` (8 bytes) allocated by or registered with NVSHMEM4Py.
    Supported signal operations are ``SignalOp.SIGNAL_SET`` and ``SignalOp.SIGNAL_ADD``.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Source symmetric array on this PE.
        - ``signal_var`` (``Array``): Symmetric signal variable (dtype ``int64``).
        - ``signal_val`` (``int``): Signal value.
        - ``signal_op`` (``SignalOp``): Signal operation type.
        - ``pe`` (``int``): Target PE to put to.
    """
    pass
@overload(put_signal_nbi)
def put_signal_nbi_ol(dst, src, signal_var, signal_val, signal_op, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int8_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int16_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int32_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            int64_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint8_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint16_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint32_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            uint64_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            float_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            double_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout) and src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(dst, src, signal_var, signal_val, signal_op, pe):
            dstptr = ffi.from_buffer(dst)
            srcptr = ffi.from_buffer(src)
            signal_varptr = ffi.from_buffer(signal_var)
            nelems = min(dst.size, src.size)
            half_put_signal_nbi(dstptr, srcptr, nelems, signal_varptr, signal_val, signal_op, pe)
        return impl



# p variations
def p():
    """
    Put immediate data from src to dst on PE pe. Device initiated.

    ``src`` must be a scalar value, passed as a symmetric Array of size 1.

    Args:
        - ``dst`` (``Array``): Destination symmetric array on remote PE.
        - ``src`` (``Array``): Scalar source value (size-1 array) on this PE.
        - ``pe`` (``int``): PE to put to.
    """
    pass
@overload(p)
def p_ol(dst, src, pe):
    if dst == Array(dtype=int8, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            int8_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=int16, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            int16_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=int32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            int32_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=int64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            int64_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=uint8, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            uint8_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=uint16, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            uint16_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=uint32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            uint32_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=uint64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            uint64_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=float32, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            float_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=float64, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            double_p(dstptr, src, pe)
        return impl
    elif dst == Array(dtype=float16, ndim=dst.ndim, layout=dst.layout):
        def impl(dst, src, pe):
            dstptr = ffi.from_buffer(dst)
            half_p(dstptr, src, pe)
        return impl

def g():
    """
    Get immediate data from src on PE pe to local dst. Device initiated.

    ``src`` must be a scalar value, passed as a symmetric Array of size 1.

    Args:
        - ``src`` (``Array``): Source symmetric array (size-1) on remote PE.
        - ``pe`` (``int``): PE to get from.
    Returns:
        Scalar value retrieved from ``src`` on PE ``pe``.
    """
    pass
@overload(g)
def g_ol(src, pe):
    if src == Array(dtype=int8, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return int8_g(srcptr, pe)
        return impl
    elif src == Array(dtype=int16, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return int16_g(srcptr, pe)
        return impl
    elif src == Array(dtype=int32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return int32_g(srcptr, pe)
        return impl
    elif src == Array(dtype=int64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return int64_g(srcptr, pe)
        return impl
    elif src == Array(dtype=uint8, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return uint8_g(srcptr, pe)
        return impl
    elif src == Array(dtype=uint16, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return uint16_g(srcptr, pe)
        return impl
    elif src == Array(dtype=uint32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return uint32_g(srcptr, pe)
        return impl
    elif src == Array(dtype=uint64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return uint64_g(srcptr, pe)
        return impl
    elif src == Array(dtype=float32, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return float_g(srcptr, pe)
        return impl
    elif src == Array(dtype=float64, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return double_g(srcptr, pe)
        return impl
    elif src == Array(dtype=float16, ndim=src.ndim, layout=src.layout):
        def impl(src, pe):
            srcptr = ffi.from_buffer(src)
            return half_g(srcptr, pe)
        return impl