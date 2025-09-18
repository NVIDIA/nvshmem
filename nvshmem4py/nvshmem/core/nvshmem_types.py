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
These are the Python datatypes for NVSHMEM
"""
import uuid
import logging
from typing import Union
from enum import Enum, IntEnum

from cuda.core.experimental._memory import MemoryResource, Buffer
from cuda.core.experimental import Device, system, ObjectCode
from cuda.core.experimental._stream import Stream
import cuda.bindings.driver

from nvshmem.bindings import malloc, free, ptr, mc_ptr, Team_id, buffer_register_symmetric, buffer_unregister_symmetric
from nvshmem.core._internal_tracking import _except_on_del

logger = logging.getLogger("nvshmem")

__all__ = ["Version", "NvshmemInvalid", "NvshmemError", "NvshmemResource", "NvshmemStream", "NvshmemStreamsType", "MutableEnum", "Teams", "BufferTypes", "NvshmemKernelObject"]

try:
    import torch
    from torch.cuda import Stream as torch_stream
    _torch_enabled = True
except:
    torch_stream = None

"""
Mutable Enum to support team creation
"""
class MutableEnum:
    def __init__(self, **kwargs):
        self._members = {}
        for key, value in kwargs.items():
            self._members[key] = value

    @classmethod
    def from_enum(cls, enum_cls: Enum):
        # Extract members from the standard enum class and create a MutableEnum instance
        members = {member.name: member.value for member in enum_cls}
        return cls(**members)

    def add(self, key, value):
        if key in self._members:
            raise ValueError(f"Member '{key}' already exists.")
        self._members[key] = value

    def remove(self, key):
        if key not in self._members:
            raise ValueError(f"Member '{key}' does not exist.")
        del self._members[key]
    
    def remove_by_value(self, value):
        """
        Remove a member by value.
        This is useful when we want to remove a team by its handle.
        Calling this function on a value notin the Enum is idempotent.

        Args:
            - value (int): The value of the member to remove.

        Returns:
            None

        Raises:
        """
        for key, val in self._members.items():
            if val == value:
                del self._members[key]
                return

    def __getattr__(self, item):
        if item in self._members:
            return self._members[item]
        raise AttributeError(f"No such member: {item}")

    def __getitem__(self, item):
        if item in self._members:
            return self._members[item]
        raise KeyError(f"No such member: {item}")

    def __contains__(self, item):
        return item in self._members

    def __repr__(self):
        members = ", ".join(f"{k}={v}" for k, v in self._members.items())
        return f"<MutableEnum {members}>"

    def keys(self):
        return self._members.keys()

    def values(self):
        return self._members.values()

    def items(self):
        return self._members.items()
     
    def __iter__(self):
        return iter(self._members)

"""
IntEnum which matches 1:1 with ``nvshmem_team_id_t``

This is a special Mutable enum so we can add elements to it later after a team was created.
"""
Teams = MutableEnum.from_enum(Team_id)
    
class NvshmemKernelObject:

    def __new__(self, *args, **kwargs):
        raise RuntimeError(
            "NvshmemKernelObject objects cannot be instantiated directly. "
            "Please use NvshmemKernelObject APIs (from_cubin, from_handle)"
        )
    
    @classmethod
    def _init(cls, handle: int):
        self = super().__new__(cls)
        self.handle = handle
        return self
    
    @staticmethod
    def from_handle(handle: int=None):
        """
        Create a NvshmemKernelObject from a handle
        """
        return NvshmemKernelObject._init(handle=int(handle))

    @staticmethod
    def from_obj(obj: ObjectCode):
        """
        Create a NvshmemKernelObject from a ObjectCode
        """
        return NvshmemKernelObject._init(handle=int(obj.handle))
    
    @staticmethod
    def from_obj_bytes(obj_bytes: bytes):
        """
        Create a NvshmemKernelObject from a bytes-string of the assembled CUDA object.
        """
        obj =  ObjectCode.from_cubin(obj_bytes)
        return NvshmemKernelObject._init(handle=int(obj.handle))

    @staticmethod
    def from_triton(triton_kernel: "JITFunction"):
        """
        Create a NvshmemKernelObject from a Triton kernel compiled with JITFunction
        """
        return NvshmemKernelObject._init(handle=int(triton_kernel.handle))
    
    # TODO: convert to class when cuda.bindings.Kernel is imported
    @staticmethod
    def from_cuda_bindings(cuda_kernel: cuda.bindings.driver.CUmodule):
        """
        Create a NvshmemKernelObject from a CUDA kernel
        """
        return NvshmemKernelObject._init(handle=int(cuda_kernel.handle))

"""
Version class
"""
class Version:
    def __init__(self, openshmem_spec_version="", nvshmem4py_version="", libnvshmem_version=""):
        self.openshmem_spec_version = openshmem_spec_version
        self.nvshmem4py_version = nvshmem4py_version
        self.libnvshmem_version = libnvshmem_version

    def __repr__(self):
        return f"""
NVSHMEM4Py Library:
    NVSHMEM Library version: {self.libnvshmem_version}
    OpenShmem Spec version: {self.openshmem_spec_version}
    NVSHMEM4Py version: {self.nvshmem4py_version}
"""

"""
Wrapper class for cuda.core StreamInteroperability protocol
"""
class NvshmemStream:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

NvshmemStreamsType = Union[torch_stream, NvshmemStream, Stream]

"""
Exceptions
"""

class NvshmemInvalid(Exception):
    def __init__(self,  msg):
        self.msg = msg

    def __repr__(self):
        return f"<NvshmemInvalid: {self.msg}>"

class NvshmemError(Exception):
    def __init__(self,  msg):
        self.msg = msg

    def __repr__(self):
        return f"<NvshmemError: {self.msg}>"

class NvshmemWarning(UserWarning):
    def __init__(self,  msg):
        self.msg = msg

    def __repr__(self):
        return f"<NvshmemWarning: {self.msg}>"

"""
Buffer types for NvshmemResource
"""
class BufferTypes(IntEnum):
    NORMAL = 0 # Normal buffer
    PEER = 1 # Peer buffer (nvshmem_ptr)
    MULTIMEM = 2 # Multicast buffer (nvshmemx_mc_ptr)
    EXTERNAL = 3 # External buffer (nvshmemx_buffer_register)


"""
Memory Resource
"""
class NvshmemResource(MemoryResource):
    """
    A memory resource that uses NVSHMEM to allocate device memory.

    This class implements the MemoryResource interface and allocates memory using
    ``nvshmem_malloc``. It supports device-accessible memory but does not allow host access.

    Attributes:
        device (Device): The CUDA device associated with this resource.
    """
    def __init__(self, device):
        """
        Initialize the NVSHMEM memory resource for a specific device.

        Args:
            device (Device): The CUDA device object on which memory will be allocated.
        """
        self.device = device
        """
        Map of symmetric heap pointers to nvshmem.core objects
        Keys: ptr, values: 
            {
            # A reference count
            "ref_count": <int>,
            # An object of type NvshmemResource
            "device": <int>,
            # An object of type cuda.core.experimental._memory.Buffer()
            "buffer": <Buffer>,
            # Different buffers (peer, mc) have different needs
            "type": Enum,
            # Used to raise an exception when the GC reaches here without free() getting called
            "freed": Bool,
            # Used to track whether the user expects us to prevent them from experiencing deadlocks
            "released": Bool
            }
        """
        self._mem_references = {}


    def allocate(self, size: int, stream: Stream=None, release=False) -> Buffer:
        """
        Allocate memory on the device using NVSHMEM.

        Args:
            - size (int): Number of bytes to allocate.
            - stream (optional): CUDA stream for allocation context (not used here).
            - release (bool, optional): do not track the buffer if True

        Returns:
            ``Buffer``: A buffer object wrapping the allocated device memory.

        Raises:
            ``NvshmemError``: If the allocation fails.
        """

        ptr = malloc(size)
        if not ptr or ptr == 0:
            raise NvshmemError(f"Failed to allocate memory of bytes {size}")
        r_buf = Buffer.from_handle(ptr=ptr, size=size, mr=self)
        logger.debug(f"Created Buffer on resource {self} at address {ptr} with size {size} on stream {stream}")

        if not release:
            buf_ref = r_buf
        else:
            # If we're not holding references, create our own shadow-buffer
            buf_ref = Buffer.from_handle(ptr=r_buf.handle, size=r_buf.size, mr=self)
        self._mem_references[ptr] = {"ref_count": 1, "resource": self, "buffer": buf_ref, "type": BufferTypes.NORMAL, "freed": False, "released": release}

        return r_buf

    def deallocate(self, ptr: int, size: int, stream: NvshmemStreamsType=None) -> None:
        """
        Placeholder method for deallocation.

        This function will be called when a Buffer has ``.close()`` called.
        ``.close()`` must be called explicitly

        Args:
            - ptr (int): Pointer to the memory block (ignored).
            - size (int): Size of the memory block in bytes (ignored).
            - stream (optional): CUDA stream (ignored).

        Returns:
            None
        """
        # Extract info
        logger.debug(f"Free called on buffer with address {ptr}")
        if self._mem_references.get(ptr) is None:
            logger.debug("Freed a buffer that is not tracked")
            return
        released  = self._mem_references[ptr].get("released", False)

        # If someone got here without calling free(), we have to except
        if self._mem_references[ptr]["type"] == BufferTypes.NORMAL and not self._mem_references[ptr]["freed"]:
            if _except_on_del["value"]:
                raise NvshmemError(f'Buffer {self._mem_references[ptr]["buffer"]} freed implicitly.')
            else:
                logger.error(f'Buffer {self._mem_references[ptr]["buffer"]} freed implicitly.')
                return

        # remove the reference
        # We keep the references around to legalize freed-ness.
        # If nvshmem_malloc returns a new ptr, we will end up creating a new entry with refcount 1
        # and freed False
        if self._mem_references[ptr]["ref_count"] > 0:
            self._mem_references[ptr]["ref_count"] -= 1
        elif self._mem_references[ptr]["ref_count"] == 0:
            # The counter is already 0, so we must have already freed the pointer. Just return.
            logger.debug(f"Ref count on {ptr} is already 0. Already freed.")
            return
        logger.debug(f"New ref count on {self._mem_references[ptr]['type'].name} buf {ptr} {self._mem_references[ptr]['ref_count'] }")
        # If this was the last reference to that pointer, free the pointer
        # The MR itself has a ref_count, but we want to free only when the last call is made.
        # Leave this as if ( == 1) and if it's 0, we will delete the reference
        if self._mem_references[ptr]["ref_count"] == 0:
            # If the buffer is a peer buffer, don't do anything 
            # except delete it from the tracker
            # NVShmem handles these internally.
            if self._mem_references[ptr]["type"] == BufferTypes.NORMAL:
                free(ptr)
                # If the buffer has a child (peer) buffer, free it now
                child_ptr = self._mem_references[ptr].get("child", None)
                if child_ptr is not None:
                    # Child is a pointer which may still be tracked.
                    child_entry = self._mem_references[child_ptr]
                    child_buffer = child_entry.get("buffer", None) 
                    if child_buffer is not None:
                        self._mem_references[child_ptr]["freed"] = True
                        del self._mem_references[child_ptr]["buffer"]
                        del self._mem_references[ptr]["child"]
                logger.debug(f"Freed buffer at address {ptr}")
            else:
                logger.debug("free() requested on a peer buffer. Not calling free()")
                self._mem_references[ptr]["freed"] = True
                buf_entry = self._mem_references[ptr].get("buffer", None)
                if buf_entry is not None:
                    del self._mem_references[ptr]["buffer"]
            

    def get_peer_buffer(self, buffer: Buffer, pe: int) -> Buffer:

        # This should be the pointer on the calling PE
        # None or raising an exception is the failing case
        parent_ptr = buffer.handle
        result = ptr(parent_ptr, pe)

        if result is None:
            raise NvshmemError("Failed to retrieve peer buffer")

        entry = self._mem_references.get(result, None)
        if entry is not None and not entry["freed"]:
            # Someone already called get_peer_buffer on the Buffer
            # Increase ref count and return existing buffer
            self._mem_references[result]["ref_count"] += 1
            logger.debug(f"Found already tracked peer buffer with address {result}. Returning it. Ref count {self._mem_references[result]['ref_count']}")
            return self._mem_references[result]["buffer"]

        logger.debug(f"Did not find peer buffer with address {result}. Creating a new one.")

        # This Buffer doesn't need to go through any .allocate() calls, since we know the pointer is valid
        r_buf = Buffer.from_handle(ptr=result, size=buffer.size, mr=self)

        self._mem_references[result] = {"ref_count": 1, "resource": self, "buffer": r_buf, "type": BufferTypes.PEER, "freed": False, "parent": parent_ptr, "released": False}
        self._mem_references[parent_ptr]["child"] = result
        return r_buf


    def get_mc_buffer(self, team: Teams, buffer: Buffer) -> Buffer:

        # This should be the pointer on the calling PE
        # None or raising an exception is the failing case
        parent_ptr = buffer.handle
        result = mc_ptr(team, parent_ptr)
        if result is None:
            raise NvshmemError("Failed to retrieve multicast buffer")

        entry = self._mem_references.get(result, None)
        if entry is not None and not entry["freed"]:
            # Someone already called get_peer_buffer on the Buffer
            # Increase ref count and return existing buffer
            self._mem_references[result]["ref_count"] += 1
            logger.debug(f"Found already tracked MC buffer with address {result}. Returning it. Ref count {self._mem_references[result]['ref_count']}")
            return self._mem_references[result]["buffer"]

        logger.debug(f"Did not find MC buffer with address {result}. Creating a new one.")

        # This Buffer doesn't need to go through any .allocate() calls, since we know the pointer is valid
        r_buf = Buffer.from_handle(ptr=result, size=buffer.size, mr=self)
        self._mem_references[result] = {"ref_count": 1, "resource": self, "buffer": r_buf, "type": BufferTypes.MULTIMEM , "freed": False, "parent": parent_ptr, "released": False}
        self._mem_references[parent_ptr]["child"] = result
        return r_buf

    def set_freed(self, buffer: Buffer) -> None:
        ptr = buffer.handle
        if self._mem_references.get(ptr) is None:
            raise NvshmemError("Freed a buffer that is not tracked")
        if self._mem_references[ptr]['type'] != BufferTypes.NORMAL:
            return
        self._mem_references[ptr]["freed"] = True

    def register_external_buffer(self, buffer: Buffer, call_register: bool = True) -> Buffer:
        """
        Register an external buffer with NVSHMEM.

        Args:
            - buffer (Buffer): The buffer to register.
            - call_register (bool, optional): Whether to call the register function on the buffer.
                                              This should be False if the buffer is already registered
                                              or was allocated via nvshmem_malloc outside of NVSHMEM4Py
        Returns:
            Buffer: A buffer object wrapping the registered external buffer.
            Users must pass this buffer to NVSHMEM operations, rather than the original buffer.
        """
        ptr = int(buffer.handle)
        if self._mem_references.get(ptr) is not None:
            if self._mem_references[ptr]['type'] != BufferTypes.EXTERNAL:
                raise NvshmemError("Tried to register an external buffer that is already tracked as a different type of buffer")
            self._mem_references[ptr]["ref_count"] += 1
            logger.warning(f"Found already tracked external buffer with address {ptr}. Returning it. Ref count {self._mem_references[ptr]['ref_count']}")
            return self._mem_references[ptr].get("buffer", None)
        if call_register:
            registered_ptr = buffer_register_symmetric(int(ptr), int(buffer.size), 0)
            if registered_ptr is None:
                raise NvshmemError("Failed to register external buffer")
            registered_ptr = int(registered_ptr)
            logger.debug(f"Registered external buffer with address {ptr}.")
            new_buf = Buffer.from_handle(ptr=registered_ptr, size=buffer.size, mr=self)
            logger.debug(f"Created new buffer with address {new_buf.handle}.")
        else:
            registered_ptr = int(ptr)
            new_buf = buffer
        self._mem_references[int(registered_ptr)] = {"ref_count": 1, "resource": self, "buffer": new_buf, "type": BufferTypes.EXTERNAL, "freed": False, "released": not call_register}
        logger.debug(f"Registered external buffer with address {registered_ptr}. Ref count {self._mem_references[registered_ptr]['ref_count']}")
        return new_buf

    def unregister_external_buffer(self, buffer: Buffer) -> None:
        ptr = int(buffer.handle)
        if self._mem_references.get(ptr) is None:
            logger.warning(f"Tried to unregister an external buffer that is not tracked with address {ptr}. Make sure you pass the buffer that NVSHMEM4Py returned to you")
            raise NvshmemError("Tried to unregister an external buffer that is not tracked")
        if self._mem_references[ptr]['type'] != BufferTypes.EXTERNAL:
            raise NvshmemError("Tried to unregister an external buffer that is not external")
        del self._mem_references[ptr]["buffer"]
        self._mem_references[ptr]["ref_count"] = 0
        self._mem_references[ptr]["freed"] = True
        if not self._mem_references[ptr]["released"]:
            buffer_unregister_symmetric(ptr, buffer.size)

        self._mem_references[int(ptr)]["freed"] = True
        logger.debug(f"Unregistered external buffer with address {ptr}. Ref count {self._mem_references[ptr]['ref_count']}")

    @property
    def is_device_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the device.

        Returns:
            bool: Always True for NVSHMEM memory.
        """
        return True

    @property
    def is_host_accessible(self) -> bool:
        """
        Indicates whether the allocated memory is accessible from the host.

        Returns:
            bool: Always False for NVSHMEM memory.
        """
        return False

    @property
    def device_id(self) -> int:
        """
        Get the device ID associated with this memory resource.

        Returns:
            int: CUDA device ID.
        """
        return self.device.device_id

    def __repr__(self) -> str:
        """
        Return a string representation of the NvshmemResource.

        Returns:
            str: A string describing the object
        """
        return f"<NvshmemResource device={self.device}>"


