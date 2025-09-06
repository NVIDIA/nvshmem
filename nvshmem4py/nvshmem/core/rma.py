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
These functions are NVSHMEM4Py APIs that expose host-initiated remote memory accesses (RMA)
"""

from cuda.core.experimental._stream import Stream
from cuda.core.experimental._memory import Buffer

from nvshmem.core.interop.cupy import _is_array, array_get_buffer
from nvshmem.core.interop.torch import _is_tensor, tensor_get_buffer
from nvshmem.core._internal_tracking import _is_initialized, InternalInitStatus
from nvshmem.core.nvshmem_types import *
from nvshmem.core.utils import _get_device
import nvshmem.bindings as bindings
from nvshmem.core.direct import ComparisonType, SignalOp

import logging
from enum import IntEnum
from typing import Tuple

__all__ = ["put_signal", "signal_wait", "put", "get", "quiet"]

logger = logging.getLogger("nvshmem")


def _get_buffers(dst, src) -> Tuple[Buffer, Buffer]:
    """
    Converts high-level inputs into cuda.core Buffer objects.

    Supports inputs as Cupy arrays, Torch tensors, or Buffer instances.
    Both ``dst`` and ``src`` must resolve to valid Buffers for the operation to proceed.

    Args:
        - dst (object): Destination memory object.
        - src (object): Source memory object.

    Returns:
        ``Tuple[Buffer, Buffer]``: A tuple of (dst_buffer, src_buffer).

    Raises:
        ``NvshmemInvalid``: If either input cannot be converted into a valid Buffer.
    """
    src_buf = None
    dst_buf = None

    if _is_array(src):
        src_buf, size, type = array_get_buffer(src)
    if _is_array(dst):
        dst_buf, size, type = array_get_buffer(dst)

    if _is_tensor(src):
        src_buf, size, type = tensor_get_buffer(src)
    if _is_tensor(dst):
        dst_buf, size, type = tensor_get_buffer(dst)

    if isinstance(dst, Buffer):
        dst_buf = dst
    if isinstance(src, Buffer):
        src_buf = src

    # if either are None at this point, it's invalid
    if src_buf is None or dst_buf is None:
        raise NvshmemInvalid("Invalid memory objects passed to NVSHMEM operation")

    return dst_buf, src_buf

def _call_putget(dst: object, src: object, op:str = "put", 
                 signal: bool=False, signal_var: Buffer = 0, signal_val: int = 0, signal_op: SignalOp = 0,
                 remote_pe: int = 0, stream: Stream=None) -> None:
    """
    Internal helper to invoke host-initiated NVSHMEM put/get with optional signaling.

    This function wraps low-level C bindings and computes the safe minimum transfer size.
    If `signal` is enabled, the provided `signal_var` must be a symmetric NVSHMEM buffer
    of at least 8 bytes.

    Args:
        - dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
        - src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
        - op (str): Either "put" or "get".
        - signal (bool): If True, invoke signal-enabled variant of the op.
        - signal_var (Buffer): Symmetric memory buffer used as the signal address.
                             This buffer must be >=8 bytes in size (the underlying API expects a uint64)
        - signal_val (int): Value to use in the signal operation.
        - signal_op (SignalOp): Signal operation type.
        - remote_pe (int): Target PE for the remote memory access.
        - stream (Stream): CUDA stream to issue the operation on.

    Raises:
        ``NotImplementedError``: If stream is None (non-stream variants unsupported).
        ``ValueError``: If op is not "put" or "get", or if inputs are invalid.
        ``NvshmemError``: If any operations do not complete successfully
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    user_nvshmem_dev, other_dev = _get_device()
    dst_buf, src_buf = _get_buffers(dst, src)
    if stream is None:
        logger.error("Non on-stream put/get operations are not yet implemented")
        raise NotImplemented

    if op not in ("put", "get"):
        raise NvshmemInvalid("Tried to call put/get function with an operation not put nor get")

    if not isinstance(src_buf, Buffer) or not isinstance(dst_buf, Buffer):
        raise NvshmemInvalid("Called collective on an invalid Buffer")

    f_name = f"{op}mem{'_signal' if signal else ''}_on_stream"
    func = getattr(bindings, f_name)
    safe_size = min(src_buf.size, dst_buf.size)
    if signal:
        if not isinstance(signal_var, Buffer) or signal_var.size < 8:
            raise NvshmemInvalid("Signal must be a Buffer >= 8 bytes allocated by NVSHMEM4Py")
        f_args = [
                  dst_buf.handle, src_buf.handle, safe_size, 
                  signal_var.handle, signal_val, signal_op,
                  remote_pe, int(stream.handle)
                 ]
    else:
        f_args = [dst_buf.handle, src_buf.handle, safe_size, remote_pe, int(stream.handle)]

    func(*f_args)

    if other_dev is not None:
        other_dev.set_current()

def put_signal(dst: object, src: object,
               signal_var: Buffer, signal_val: int, signal_op: SignalOp,
               remote_pe: int=-1, stream=None) -> None:
    """
    Performs a put with signal on a CUDA stream.

    Args:
        - dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
        - src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
        - signal_var (Buffer): Symmetric memory buffer used as signal variable.
        - signal_val (int): Value to use in the signal operation.
        - signal_op (SignalOp): Signal operation type.
        - remote_pe (int): Target PE for the put.
        - stream (Stream): CUDA stream to issue the put on.

    Raises:
        - ``NotImplementedError``: If stream is None.
        - ``ValueError``: If the signal buffer is invalid or too small.
        - ``NvshmemError``: If any operations do not complete successfully
    """
    _call_putget(dst, src, op="put", 
                 signal=True, signal_var=signal_var, signal_val=signal_val, signal_op=signal_op,
                 remote_pe=remote_pe, stream=stream)


def signal_wait(signal_var: Buffer, signal_val: int, signal_op: ComparisonType, stream: Stream=None) -> None:
    """
    Waits until a symmetric signal variable satisfies a given condition.

    Args:
        - signal_var (Buffer): Symmetric memory buffer used as the signal source.
        - signal_val (int): Value to compare against.
        - signal_op (SignalOp): Wait condition
        - stream (Stream): CUDA stream to issue the wait on.

    Raises:
        - ``NotImplementedError``: If stream is None.
        - ``ValueError``: If the signal buffer is invalid.
        - ``NvshmemError``: If any operations do not complete successfully
    """
    if stream is None:
        logger.error("Non on-stream put/get operations are not yet implemented")
        raise NotImplemented
    user_nvshmem_dev, other_dev = _get_device()

    bindings.signal_wait_until_on_stream(signal_var.handle, signal_op, signal_val, int(stream.handle))

    if other_dev is not None:
        other_dev.set_current()


def quiet(stream: Stream=None) -> None:
    """
    Ensures completion of all previously issued NVSHMEM operations on the given stream.

    This is equivalent to a device-side ``shmem_quiet`` for host-initiated NVSHMEM operations.

    Note that this function will return when local (the PE this is called from) operations are completed.
    Remote operations may not yet be. Other synchronizations are required

    Args:
        - stream (``Stream``): CUDA stream to synchronize.

    Raises:
        - ``NotImplementedError``: If stream is None.
        - ``NvshmemError``: If any operations do not complete successfully
    """
    if stream is None:
        logger.error("Non on-stream put/get operations are not yet implemented")
        raise NotImplemented
    user_nvshmem_dev, other_dev = _get_device()
    # Because quiet doesn't have a datatype, it's a special case and doesn't need to use _call_putget function
    bindings.quiet_on_stream(int(stream.handle))
    if other_dev is not None:
        other_dev.set_current()

def put(dst: object, src: object, remote_pe: int=-1, stream: Stream=None):
    """
    Performs a host-initiated NVSHMEM put operation on a CUDA stream.

    Args:
        - dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
        - src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
        - remote_pe (int): Target PE for the put.
        - stream (Stream): CUDA stream to issue the put on.

    Raises:
        - ``NotImplementedError``: If stream is ``None``.
        - ``NvshmemInvalid``: If inputs are not valid Buffer-compatible types.
        - ``NvshmemError``: If any operations do not complete successfully
    """
    _call_putget(dst, src, op="put", signal=False,
                 remote_pe=remote_pe, stream=stream)

def get(dst: object, src: object, remote_pe: int=-1, stream: Stream=None):
    """
    Performs a host-initiated NVSHMEM get operation on a CUDA stream.

    Args:
        - dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
        - src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
        - remote_pe (int): Target PE for the get.
        - stream (Stream): CUDA stream to issue the get on.

    Raises:
        - ``NotImplementedError``: If stream is None.
        - ``ValueError``: If inputs are not valid Buffer-compatible types.
        - ``NvshmemError``: If any operations do not complete successfully
    """
    _call_putget(dst, src, op="get", signal=False,
                 remote_pe=remote_pe, stream=stream)

