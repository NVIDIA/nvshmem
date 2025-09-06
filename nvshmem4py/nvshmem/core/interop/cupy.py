# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

"""
The following are interoperability helpers for NVSHMEM4Py memory used in CuPy
"""
import nvshmem.core
from nvshmem.core.utils import get_size
from nvshmem.core.nvshmem_types import *
from nvshmem.core._internal_tracking import _mr_references

import logging
from typing import Tuple, Union

from cuda.core.experimental._memory import Buffer
from cuda.core.experimental import Device

__all__ = ["bytearray", "array", "free_array", "array_get_buffer", "get_peer_array"]

logger = logging.getLogger("nvshmem")

try:
    import cupy
    from cupy import ndarray
    _cupy_enabled = True
except:
    _cupy_enabled = False
    ndarray = None

import numpy as np

def _is_array(array: Union[ndarray, object]) -> bool:
    """
    Helper function to check if an object is a CuPy array
    This is used in collectives to avoid putting the complicated 
    import logic for CuPy in any other file but this.
    """
    if not _cupy_enabled:
        return False
    return isinstance(array, ndarray)

def array_get_buffer(array: ndarray) -> Tuple[Buffer, int, str]:
    """
    Get a nvshmem Buffer object from a Cupy NDArray object which was allocated with ``nvshmem.core.array()`` or ``nvshmem.core.bytearray()`` 

    Returns a Tuple of the array and its size in bytes
    """
    mr = _mr_references.get(array.device.id)
    if mr is None:
        # This avoids a raw KeyError which would be confusing to users
        raise NvshmemInvalid("Tried to retrieve MemoryResource for GPU with no NVSHMEM Allocations")
    buf = mr._mem_references.get(int(array.data.ptr), {}).get("buffer")
    if buf is None:
        raise NvshmemInvalid("Tried to retrieve buffer from Array not tracked by nvshmem")
    return buf, get_size(array.shape, array.dtype), str(array.dtype)


def array(shape: Tuple[int], dtype: str="float32") -> ndarray:
    """
    Create a CuPy array view on NVSHMEM-allocated memory with the given shape and dtype.

    This function allocates memory using NVSHMEM, wraps it with a DLPack-compatible CuPy array,
    and returns a reshaped and retyped view of that memory.

    Args:
       -  shape (tuple or list of int): Shape of the desired array.
       -dtype (``str``, ``np.dtype``, or ``cupy.dtype``, optional): Data type of the array. Defaults to ``"float32"``.

    Any future calls to ``.view()`` on this object should set copy=False, to avoid copying the object off of the sheap

    Returns:
        ``cupy.ndarray``: A CuPy array view on NVSHMEM-allocated memory.

    Raises:
        ``ModuleNotFoundError``: If CuPy is not available or enabled.
    """
    if not _cupy_enabled:
        logger.error("Can not create CuPy array: CuPy not installed.")
        raise ModuleNotFoundError

    buf = nvshmem.core.buffer(get_size(shape, dtype))
    # Important! Disable copy to force allocation to stay on sheap
    cupy_array = cupy.from_dlpack(buf, copy=False)
    view = cupy_array.view(dtype).reshape(shape)
    return view

def bytearray(shape: Tuple[int], dtype: str="float32", device_id: int=None) -> ndarray:
    """
    Create a raw CuPy byte array from NVSHMEM-allocated memory.

    This function allocates raw memory using NVSHMEM and wraps it with a CuPy array
    without reshaping or reinterpreting the dtype view.

    This function uses the shape and dtype to choose how much memory to allocate, but does not cast or reshape
    Therefore, the type of the array will always be cupy.uint8.

    Any future calls to ``.view()`` on this object should set ``copy=False``, to avoid copying the object off of the sheap

    Args:
        - shape (tuple or list of int): Shape of the desired array.
        - dtype (``str``, ``np.dtype``, or ``cupy.dtype``, optional): Data type of the array. Defaults to ``"float32"``.

    Returns:
        ``cupy.ndarray``: A CuPy array backed by NVSHMEM-allocated memory.

    Raises:
        ``ModuleNotFoundError``: If CuPy is not available or enabled.
    """
    if not _cupy_enabled:
        return
    buf = nvshmem.core.buffer(get_size(shape, dtype))
    cupy_array = cupy.from_dlpack(buf, copy=False)
    return cupy_array

def get_peer_array(array: ndarray, peer_pe: int=None) -> ndarray:
    """
    Return a Buffer based on the peer_buffer (wrapper of nvshmem_ptr) API
    """
    if not _cupy_enabled:
        return
    buf, size, dtype = array_get_buffer(array)
    peer_buf = nvshmem.core.get_peer_buffer(buf, peer_pe)
    return cupy.from_dlpack(peer_buf, copy=False).view(array.dtype).reshape(cupy.shape(array))

def free_array(array: ndarray) -> None:
    """
    Free an NVSHMEM-backed CuPy Array

    Args:
        array (``cupy.ndarray``): A CuPy array backed by NVSHMEM memory.

    Returns:
        None

    Raises:
        ``ModuleNotFoundError``: If CuPy is not available or enabled.
    """
    if not _cupy_enabled:
        return
    # Convert array to Buffer
    buf, arr_size, dtype = array_get_buffer(array)
    nvshmem.core.free(buf)
