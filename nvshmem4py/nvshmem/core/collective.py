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
The following are NVSHMEM4Py APIs that expose host-initiated collective communication
"""

from cuda.core.experimental._stream import Stream
from cuda.core.experimental._memory import Buffer
from cuda.core.experimental import Device

from nvshmem.core.utils import _get_device, dtype_nbytes
from nvshmem.core.interop.torch import _is_tensor, tensor_get_buffer
from nvshmem.core.interop.cupy import _is_array, array_get_buffer
from nvshmem.core._internal_tracking import _mr_references, _is_initialized, InternalInitStatus
from nvshmem.core.direct import Teams, team_n_pes, n_pes
from nvshmem.core.nvshmem_types import *
import nvshmem.bindings as bindings

from typing import Union, Tuple
import logging

logger = logging.getLogger("nvshmem")

__all__ = ["reduce", "reducescatter", "alltoall", "fcollect", "broadcast", "barrier", "sync", "collective_on_buffer"]

"""
On-Stream Collectives

All take a team, a Stream. Some take an operation.
"""

# Valid reduction/reducescatter ops. Enhance if needed.
valid_ops = ["min", "max", "sum", None]

# Collectives
valid_collectives = ["reduce", "reducescatter", "alltoall", "fcollect", "broadcast"]

# Mapping from Cupy/Torch dtypes to NVSHMEM dtype names
external_to_nvshmem_dtypes = {
    # --------------------
    # Torch dtypes
    # --------------------
    "torch.float16": "half",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float",
    "torch.float64": "double",
    
    "torch.uint8": "uint8",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.bool": "uint8",  # mapped to uint8 for collective support

    # --------------------
    # CuPy dtypes
    # --------------------
    "float16": "half",
    "bfloat16": "bfloat16",
    "float32": "float",
    "float64": "double",
    
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "uint8",  # mapped to uint8 for collective support
}

def _call_collective(coll: str, team: Teams, dst_array:object, src_array: object, op: str=None, root: int=0, stream:Stream=None):
    """
    Executes a collective operation on structured array types (Torch or CuPy).

    This function performs validation and dispatches the appropriate NVSHMEM
    collective routine using the binding layer. It is intended for host-initiated
    collectives operating on array types.

    Args:
        - team: The NVSHMEM team handle.
        - src_array (object): The source array (Torch Tensor or CuPy ndarray).
        - dst_array (object): The destination array (Torch Tensor or CuPy ndarray).
        - coll (str): Collective operation name (e.g., "reduce", "broadcast").
        - op (str, optional): Reduction operator (e.g., "sum", "max"). Defaults to None.
        - stream (Stream, optional): CUDA stream to associate with the operation. Defaults to None.
        - root: Root PE. Currently only used for broadcast. Defaults to 0

    Raises:
        NvshmemInvalid: If the input arrays are incompatible or an invalid operator is used.
    """

    # Excepts upon invalid dtype
    src_buf, src_buf_size, src_nvshmem_dtype = _check_dtype(src_array)
    dst_buf, dst_buf_size, dst_nvshmem_dtype = _check_dtype(dst_array)

    if src_nvshmem_dtype != dst_nvshmem_dtype:
        raise NvshmemInvalid("Non-matching data types for src and dest arrays")

    if op not in valid_ops:
        raise NvshmemInvalid(f"Invalid operator passed to reduce_on_stream. Valid options: {valid_ops.join(', ', )}")

    collective_on_buffer(coll, team, dst_buf, src_buf, dtype=src_nvshmem_dtype, op=op, root=root, stream=stream)

def collective_on_buffer(coll: str, team: Teams, dest: Buffer, src: Buffer, dtype: str=None, op: str=None, root: int=0, stream:Stream=None, enable_timing=False) -> float:
    """
    This function allows host-initiated collectives over raw memory buffers.
    It is used by higher-level wrappers or when working with DLPack-converted memory directly.

    Args:
        - team: The NVSHMEM team handle.
        - src (Buffer): Source buffer.
        - dest (Buffer): Destination buffer.
            Source and dest buffers MUST be allocated by nvshmem4py
        - coll (str): Collective type (e.g., "reduce", "fcollect").
        - dtype (str, optional): NVSHMEM dtype string for function resolution.
        - op (str, optional): Reduction operator if required.
        - stream (Stream, optional): CUDA stream to execute on.
        - root (int): Root PE for collective. Only used for Broadcast today.
        - enable_timing: If True, return the time (in ms) it took the collective to execute

    Raises:
        ``NvshmemInvalid``: If the Buffer or collective type is invalid.

    Returns:
        float: time in ms that the collective took to execute if enable_timing is set, else 0

    NOTE:
        This API is considered experimental and should be used only by expert users
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    if stream is None:
        # TODO: Implement default stream and support non-on-stream collectives
        logger.warning("Non-on-stream collectives are not yet implemented. Stream may not be None")
        raise NotImplemented

    user_nvshmem_dev, other_dev = _get_device()
    if not isinstance(src, Buffer) or not isinstance(dest, Buffer):
        raise NvshmemInvalid("Called collective on an invalid Buffer")

    # Assert that these buffers are tracked by nvshmem
    if not _mr_references.get(user_nvshmem_dev.device_id, {})._mem_references.get(src.handle) \
       or not _mr_references.get(user_nvshmem_dev.device_id, {})._mem_references.get(dest.handle):
        raise NvshmemInvalid("Tried to perform a collective on a buffer not allocated by NVSHMEM4Py")

    if not coll in valid_collectives:
        raise NvshmemInvalid("Passed invalid collective")

    # Set up timing
    if enable_timing:
        start_event = user_nvshmem_dev.create_event({"enable_timing": True})
        stop_event = user_nvshmem_dev.create_event({"enable_timing": True})
    else:
        time_ms = 0

    # Op and dtype validity are checked in _call_collective()
    # For callers of this function directly, we don't know anymore what the types are
    # - it's on the user to get it right.

    src_size = src.size
    dest_size = dest.size
    size = dest_size

    # We assume the user has passed the correct dest_size.
    # If the user doesn't, they will encouter some kind of Exception later on
    if coll in ("reduce", "reducescatter", "fcollect",):
        size_elem = max(1, size // dtype_nbytes(dtype))
    elif coll in ("alltoall", "broadcast"):
        size_elem = max(1, size // (n_pes() * dtype_nbytes(dtype)))

    func_name = ""
    if dtype:
        func_name += f"{dtype}_"
    if op:
        func_name += f"{op}_"
    func_name += f"{coll}_on_stream"

    # We have a string of the coll function name
    coll_func = getattr(bindings, func_name)
    function_args = [team, dest._mnff.ptr, src._mnff.ptr, size_elem, int(stream.handle)]
    if coll == "broadcast":
        # Rewrite the whole thing instead of append to make the ordering requirements more obvious
        function_args = [team, dest._mnff.ptr, src._mnff.ptr, size_elem, root, int(stream.handle)]
    if enable_timing:
        stream.record(start_event)
    result = coll_func(*function_args)

    if enable_timing:
        stream.record(stop_event)
        stop_event.sync()
        time_ms = (stop_event - start_event)  # in ms

    # If result is not 0 or None, it failed
    if result:
        raise NvshmemError(f"Collective {coll} on team {team} failed to execute")

    if other_dev is not None:
        other_dev.set_current()

    return time_ms

def _check_dtype(array: object) -> Tuple[Buffer, int, str]:
    """
    Validates an array for collective operations and extracts metadata.

    This function ensures the array is a supported type (Torch or CuPy) and maps
    its dtype to an NVSHMEM-compatible type.

    Args:
        array (object): The array object to inspect.

    Returns:
        Tuple[Buffer, int, str]: A tuple containing the device buffer, the size in bytes,
                                 and the NVSHMEM dtype string.

    Raises:
        ``NvshmemInvalid``: If the array is not a supported type or has an unsupported dtype.
    """
    # The correct device must be current here because we need to know the device ID to look up the correct MR
    user_nvshmem_dev, other_dev = _get_device()
    is_array = _is_array(array)
    is_tensor = _is_tensor(array)
    if not (is_array or is_tensor):
        raise NvshmemInvalid("Passed a non-array object into a collective function")

    if is_array:
        buf, buf_size, external_dtype = array_get_buffer(array)
        
    elif is_tensor:
        buf, buf_size, external_dtype = tensor_get_buffer(array)
    
    nvshmem_dtype = external_to_nvshmem_dtypes.get(external_dtype)
    if not nvshmem_dtype:
        raise NvshmemInvalid("Passed an invalid datatype into a collective function")
    if other_dev is not None:
        other_dev.set_current()
    return buf, buf_size, nvshmem_dtype

def barrier(team: Teams, stream: Stream=None) -> None:
    """
    Executes a team-wide barrier on a specified CUDA stream.

    Args:
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream for synchronization.
    """
    user_nvshmem_dev, other_dev = _get_device()
    # Because Barrier doesn't have a datatype, it's a special case and doesn't need to use call_collective function
    bindings.barrier_on_stream(team, int(stream.handle))
    if other_dev is not None:
        other_dev.set_current()

def sync(team: Teams, stream: Stream=None) -> None:
    """
    Executes a team-wide sync on a specified CUDA stream.

    Args:
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream for synchronization.
    """
    user_nvshmem_dev, other_dev = _get_device()
    # Because Barrier doesn't have a datatype, it's a special case and doesn't need to use call_collective function
    bindings.team_sync_on_stream(team, int(stream.handle))
    if other_dev is not None:
        other_dev.set_current()

def reduce(team: Teams, dst_array: object, src_array: object, op: str, stream: Stream=None):
    """
    Performs a reduction from src_array to dst_array on a CUDA stream.

    Args:
        - src_array: Source array (Torch or CuPy).
        - dst_array: Destination array.
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream to perform the operation on.
        - op (str): Reduction operator (e.g., "sum").
    """
    _call_collective("reduce", team, dst_array, src_array, op=op, stream=stream)

def reducescatter(team: Teams, dst_array: object, src_array: object, op: str, stream: Stream=None):
    """
    Performs a reduce-scatter operation on a CUDA stream.

    Args:
        - src_array: Source array (Torch or CuPy).
        - dst_array: Destination array.
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream to perform the operation on.
        - op (str): Reduction operator (e.g., "sum").
    """
    _call_collective("reducescatter", team, dst_array, src_array, op=op, stream=stream)

def alltoall(team: Teams, dst_array: object, src_array: object, stream: Stream=None):
    """
    Performs an all-to-all communication on a CUDA stream.

    Args:
        - src_array: Source array (Torch or CuPy).
        - dst_array: Destination array.
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream to perform the operation on.
    """
    _call_collective("alltoall", team, dst_array, src_array, op=None, stream=stream)

def fcollect(team: Teams, dst_array: object, src_array: object, stream: Stream=None):
    """
    Performs a full-collective operation on a CUDA stream.

    Args:
        - src_array: Source array (Torch or CuPy).
        - dst_array: Destination array.
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream to perform the operation on.
    """
    _call_collective("fcollect", team, dst_array, src_array, op=None, stream=stream)

def broadcast(team: Teams, dst_array: object, src_array: object, root: int=0, stream: Stream=None):
    """
    Broadcasts data from src_array to dst_array across the team on a CUDA stream.

    Args:
        - src_array: Source array (Torch or CuPy).
        - dst_array: Destination array.
        - team: NVSHMEM team handle.
        - stream (Stream): CUDA stream to perform the operation on.
        - root: Root PE
    """
    _call_collective("broadcast", team, dst_array, src_array, root=root, op=None, stream=stream)
