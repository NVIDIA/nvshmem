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

from cuda.core.experimental._memory import MemoryResource, Buffer
from cuda.core.experimental import Device, system
from cuda.core.experimental._stream import Stream

from nvshmem.bindings import malloc, free, ptr

__all__ = ["Version", "NvshmemInvalid", "NvshmemError", "NvshmemResource"]

logger = logging.getLogger("nvshmem")

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
            # Used to make free a no-op for peer buffers 
            "is_peer_buffer": Bool,
            # Used to raise an exception when the GC reaches here without free() getting called
            "freed": Bool
            }
        """
        self._mem_references = {}


    def allocate(self, size: int, stream: Stream=None) -> Buffer:
        """
        Allocate memory on the device using NVSHMEM.

        Args:
            - size (int): Number of bytes to allocate.
            - stream (optional): CUDA stream for allocation context (not used here).

        Returns:
            ``Buffer``: A buffer object wrapping the allocated device memory.

        Raises:
            ``NvshmemError``: If the allocation fails.
        """

        ptr = malloc(size)
        if not ptr or ptr == 0:
            raise NvshmemError(f"Failed to allocate memory of bytes {size}")
        r_buf = Buffer(ptr=ptr, size=size, mr=self)
        logger.debug(f"Created Buffer on resource {self} at address {ptr} with size {size} on stream {stream}")
        self._mem_references[ptr] = {"ref_count": 1, "resource": self, "buffer": r_buf, "is_peer_buffer": False, "freed": False}
        return r_buf

    def deallocate(self, ptr: int, size: int, stream: Stream=None) -> None:
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

        # If someone got here without calling free(), we have to except
        if not self._mem_references[ptr]["is_peer_buffer"] and not self._mem_references[ptr]["freed"]:
            raise NvshmemError(f'Buffer {self._mem_references[ptr]["buffer"]} freed implicitly.')

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
        logger.debug(f"New ref count on {'peer' if  self._mem_references[ptr]['is_peer_buffer'] else ''} buf {ptr} {self._mem_references[ptr]['ref_count'] }")
        # If this was the last reference to that pointer, free the pointer
        # The MR itself has a ref_count, but we want to free only when the last call is made.
        # Leave this as if ( == 1) and if it's 0, we will delete the reference
        if self._mem_references[ptr]["ref_count"] == 0:
            # If the buffer is a peer buffer, don't do anything 
            # except delete it from the tracker
            # NVShmem handles these internally.
            if not self._mem_references[ptr]["is_peer_buffer"]:
                free(ptr)
                logger.debug(f"Freed buffer at address {ptr}")
            else:
                logger.debug("free() requested on a peer buffer. Not calling free()")
            

    def get_peer_buffer(self, buffer: Buffer, pe: int) -> Buffer:

        # This should be the pointer on the calling PE
        # None or raising an exception is the failing case
        result = ptr(buffer._mnff.ptr, pe)
        if result is None:
            raise NvshmemError("Failed to retrieve peer buffer")

        if self._mem_references.get(result, None) is not None:
            # Someone already called get_peer_buffer on the Buffer
            # Increase ref count and return existing buffer
            self._mem_references[result]["ref_count"] += 1
            logger.debug(f"Found already tracked peer buffer with address {result}. Returning it. Ref count {self._mem_references[result]['ref_count']}")
            return self._mem_references[result]["buffer"]

        logger.debug(f"Did not find peer buffer with address {result}. Creating a new one.")

        # This Buffer doesn't need to go through any .allocate() calls, since we know the pointer is valid
        r_buf = Buffer(ptr=result, size=buffer.size, mr=self)
        
        self._mem_references[result] = {"ref_count": 1, "resource": self, "buffer": r_buf, "is_peer_buffer": True, "freed": False}
        return r_buf

    def set_freed(self, buffer: Buffer) -> None:
        ptr = buffer._mnff.ptr
        if self._mem_references.get(ptr) is None:
            raise NvshmemError("Freed a buffer that is not tracked")
        if self._mem_references[ptr]['is_peer_buffer']:
            return
        self._mem_references[ptr]["freed"] = True


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


