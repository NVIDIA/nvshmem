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
The following functions relate to management of NVSHMEM symmetric memory in Python
"""
import logging

from nvshmem.core.nvshmem_types import *
import nvshmem.bindings as bindings
from nvshmem.core._internal_tracking import _mr_references, _cached_device, _is_initialized, InternalInitStatus
from nvshmem.core.utils import _get_device

from cuda.core.experimental import Device, system
from cuda.core.experimental._memory import Buffer

__all__ = ['buffer', 'free', 'get_peer_buffer', 'get_multicast_buffer', 'register_external_buffer', 'unregister_external_buffer']

logger = logging.getLogger("nvshmem")

def _free_all_buffers() -> None: 
    """
    Frees all allocated NVSHMEM buffers currently being tracked.

    This is an internal helper used to clean up all memory allocations managed by
    ``_mr_references``. Each allocation is freed only once its reference count has reached zero.

    Logs each deallocation for debugging or auditing purposes.
    """
    found_leak = False
    for key in sorted(_mr_references.keys()):
        mr = _mr_references[key]
        for ptr in sorted(mr._mem_references.keys()):
            if mr._mem_references[ptr]["type"] != BufferTypes.NORMAL or mr._mem_references[ptr]["freed"]:
                continue
            found_leak = True
            logger.info(f"Found object open at pointer {ptr} and ref count {mr._mem_references[ptr]['ref_count']}. Freeing it.")
            # We already printed the warning message so we can safely suppress the message
            mr._mem_references[ptr]["freed"] = True
            mr._mem_references[ptr]["released"] = True
            mr._mem_references[ptr]["except_on_del"] = False
            mr._mem_references[ptr]["ref_count"] = 0
            mr.deallocate(ptr, 0)
    if found_leak:
        logger.error("Some NVSHMEM Symmetric memory was not freed explicitly (you may have forgotten to clean up before finalizing, or an unrelated exception crashed the program before it was freed)")

def buffer(size, release=False, except_on_del=True) -> Buffer:
    """
    Allocates an NVSHMEM-backed CUDA buffer.

    Args:
        size (int): The size in bytes of the buffer to allocate.
        release (bool, optional): Do not track this buffer internally to NVSHMEM
                If True, it is the user's responsibility to hold references to the buffer until free() is called
                otherwise, deadlocks may occur.
        except_on_del (bool, optional): If True, raise an exception if the buffer is not tracked or has already been freed.
                If False, print an error message.

    Returns:
        ``cuda.core.Buffer``: A DLPack-compatible CUDA buffer with NVSHMEM backing.

    Raises:
        ``NvshmemError``: If the buffer could not be allocated properly.

    Note that this is a collective. All participating PEs must call ``buffer()`` in concert.

    This operation runs on the cached Device. If the cached Device is not the current device, it will set the cached device to current and set it back at the end of the operation.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    user_nvshmem_dev, other_dev = _get_device()

    dev_id = user_nvshmem_dev.device_id
    resource = _mr_references.get(dev_id)
    if resource is None:
        logger.debug(f"Creating NvshmemResource for device {dev_id}")
        resource = NvshmemResource(user_nvshmem_dev)
        _mr_references[dev_id] = resource

    buf = resource.allocate(size, release=release, except_on_del=except_on_del)

    if other_dev is not None:
        other_dev.set_current()
    return buf

def free(buffer: Buffer) -> None:
    """
    Frees an NVSHMEM buffer that was previously allocated.

    Args:
        buffer (``cuda.core.Buffer``): The buffer to free.

    Raises:
        - ``NvshmemInvalid``: If the buffer is not a valid NVSHMEM-managed buffer.
        - ``NvshmemError``: If the buffer is not tracked or has already been freed.

    Note that this is a collective. All participating PEs must call ``free()`` in concert.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")
    # _get_device() excepts if no device is current
    user_nvshmem_dev, other_dev = _get_device()

    if not (isinstance(buffer, Buffer)) or not hasattr(buffer, "handle"):
        raise NvshmemInvalid("Tried to free a buffer not from NVSHmem")

    buffer.memory_resource.set_freed(buffer)
    try:
        buffer.memory_resource.set_freed(buffer)
    except NvshmemError:
        logger.error(f"Freed a buffer {buffer} that was previously freed or not tracked")

    buffer.close()
    if other_dev is not None:
        other_dev.set_current()

def get_peer_buffer(buffer: Buffer, pe: int):
    """
    Returns a peer buffer associated with an NVSHMEM-allocated object.

    This is the Python object equivalent of nvshmem_ptr, which:
        - Given a pointer to an object on the NVSHMEM Symmetric Heap
        - Returns a pointer to a local object to which loads and stores can be performed

    The Python equivalent returns a cuda.core.Buffer which starts at the address of the Buffer passed in, with the same size as the Buffer passed in.

    For more information on nvshmem_ptr, see https://docs.nvidia.com/nvshmem/archives/nvshmem-101/api/docs/gen/api/setup.html#nvshmem-ptr

    The get_peer_buffer function offers an efficient means to accomplish communication, for example when a sequence of reads and writes to a data object on a remote PE does not match the access pattern provided in other APIs.

    Args:
        - buffer (``cuda.core.Buffer``): A buffer allocated with NVSHMEM.
        - pe (``int``): The peer's PE

    Returns:
        - ``cuda.core.Buffer``: The buffer object representing the remote peer's buffer.
            User need not call ``nvshmem.core.free()`` on this Buffer. It will be a no-op

    Raises:
        - ``NvshmemInvalid``: If the input buffer is not a valid NVSHMEM buffer.
        - ``NvshmemError``: If the buffer is not tracked internally or no peer information is found.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    # _get_device() excepts if no device is current
    user_nvshmem_dev, other_dev = _get_device()

    if not isinstance(buffer, Buffer) or not hasattr(buffer, "_mnff"):
        raise NvshmemInvalid("Tried to use a buffer not from NVSHmem")
    
    mr = buffer.memory_resource
    peer_buffer = mr.get_peer_buffer(buffer, pe)
    if other_dev is not None:
        other_dev.set_current()
    return peer_buffer

def register_external_buffer(buffer: Buffer) -> Buffer:
    """
    Register an external buffer with NVSHMEM.
    
    This function is a collective. All participating PEs must call ``register_external_buffer()`` in concert.
    This function will return a new buffer object that wraps a symmetric pointer to the external buffer.
    Users must pass this buffer to NVSHMEM operations, rather than the original buffer.

    Args:
        - buffer (``cuda.core.Buffer``): A buffer to register with NVSHMEM.

    Returns: 
        - ``cuda.core.Buffer``: A buffer object wrapping the registered external buffer.

    The user must call ``unregister_external_buffer()` on the buffer returned by this 
    function to unregister the buffer from NVSHMEM when they are done with it.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")
    
    # _get_device() excepts if no device is current
    user_nvshmem_dev, other_dev = _get_device()

    # Unlike other functions, we do not expect the buffer to be tracked by NVSHMEM
    mr = _mr_references.get(user_nvshmem_dev.device_id)
    if mr is None:
        raise NvshmemInvalid("Tried to register an external buffer on a device that is not initialized with NVSHMEM")
    registered_buffer = mr.register_external_buffer(buffer)
    if other_dev is not None:
        other_dev.set_current()
    if registered_buffer is not None:
        return registered_buffer

def unregister_external_buffer(buffer: Buffer) -> None:
    """
    Unregister an external buffer with NVSHMEM.

    This function is a collective. All participating PEs must call ``unregister_external_buffer()`` in concert.
    
    Args:
        - buffer (``cuda.core.Buffer``): The buffer returned by ``register_external_buffer()`` to unregister.
    
    Raises:
        - ``NvshmemInvalid``: If NVSHMEM is not initialized or buffer is invalid.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")
    
    # _get_device() excepts if no device is current
    user_nvshmem_dev, other_dev = _get_device()

    if not isinstance(buffer, Buffer) or not hasattr(buffer, "_mnff"):
        raise NvshmemInvalid("Tried to use a buffer not from NVSHmem")
    
    mr = _mr_references.get(user_nvshmem_dev.device_id)
    if mr is None:
        raise NvshmemInvalid("Tried to unregister an external buffer on a device that is not initialized with NVSHMEM")
    mr.unregister_external_buffer(buffer)
    if other_dev is not None:
        other_dev.set_current()

def get_multicast_buffer(team: Teams, buffer: Buffer) -> Buffer:
    """
    Returns a peer buffer associated with an NVSHMEM-allocated object which uses Multicast Memory.

    This is the Python object equivalent of ``nvshmemx_mc_ptr``, which:
        - Given a pointer to an object on the NVSHMEM Symmetric Heap
        - Returns a pointer to a local object to which multicast loads and stores can be performed

    The Python equivalent returns a cuda.core.Buffer which starts at the address of the Buffer passed in, with the same size as the Buffer passed in.

    For more information on ``nvshmemx_mc_ptr``, see https://docs.nvidia.com/nvshmem/api/using.html#communication-model, or
      https://docs.nvidia.com/nvshmem/api/gen/api/setup.html?highlight=mc_ptr#c.nvshmemx_mc_ptr

    The ``get_multicast_buffer`` function offers a quick means to leverage NVSwitch based multicast and reduction offload features also referred to as NVLink SHARP.

    IMPORTANT: The buffer allocated by ``get_multicast_buffer`` may NOT be accessed from the host (CPU). It can only be accessed by GPU kernels.

    NOTE: At present, copy_from() and copy_to() are not supported for Multicast buffers.

    Args:
        - buffer (``cuda.core.Buffer``): A multicast buffer allocated with NVSHMEM.
        - Team (``Teams`` enum): The team which this multicast memory applies to

    Returns:
        - ``cuda.core.Buffer``: The buffer object representing the remote peer's buffer.
            User need not call ``nvshmem.core.free()`` on this Buffer. It will be a no-op

    Raises:
        - ``NvshmemInvalid``: If the input buffer is not a valid NVSHMEM buffer.
                              or if the platform does not support Multicast memory
        - ``NvshmemError``: If the buffer is not tracked internally or no team information is found.
    """
    if _is_initialized["status"] != InternalInitStatus.INITIALIZED:
        raise NvshmemInvalid("NVSHMEM Library is not initialized")

    # _get_device() excepts if no device is current
    user_nvshmem_dev, other_dev = _get_device()

    if not isinstance(buffer, Buffer) or not hasattr(buffer, "_mnff"):
        raise NvshmemInvalid("Tried to use a buffer not from NVSHmem")
    
    mr = buffer.memory_resource
    peer_buffer = mr.get_mc_buffer(team, buffer)
    if other_dev is not None:
        other_dev.set_current()
    return peer_buffer
 
 