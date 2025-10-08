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
The following are interoperability helpers for NVSHMEM4Py memory used in Torch
"""
import nvshmem.core
from nvshmem.core.memory import buffer
from nvshmem.core.utils import get_size
from nvshmem.core._internal_tracking import _mr_references
from nvshmem.core.nvshmem_types import *
from nvshmem import bindings

from cuda.core.experimental._memory import Buffer
from cuda.core.experimental import Device
from cuda.core.experimental._stream import Stream

from typing import Tuple, Union

__all__ = ["bytetensor", "tensor", "free_tensor", "tensor_get_buffer", "get_peer_tensor", "get_multicast_tensor", "register_external_tensor", "unregister_external_tensor"]

try:
    import torch
    from torch import float32
    from torch import uint8
    from torch import Tensor
    from torch import dtype
    _torch_enabled = True
except:
    float32 = None
    torch = None
    Tensor = None
    dtype = None
    uint8 = None
    _torch_enabled = False

def _is_tensor(tensor: Union[Tensor, object]) -> bool:
    """
    Helper function to check if an object is a Torch tensor
    This is used in collectives to avoid putting the complicated 
    import logic for Torch in any other file but this.
    """
    if not _torch_enabled:
        return False
    return isinstance(tensor, Tensor)

def tensor_get_buffer(tensor: Tensor) -> Tuple[Buffer, int, str]:
    """
    Get a nvshmem Buffer object from a Torch tensor object which was allocated with
    ``nvshmem.core.tensor()`` or ``nvshmem.core.bytetensor()``
    Returns the buffer and the array's size
    """
    mr = _mr_references.get(int(tensor.get_device()))
    if mr is None:
        # This avoids a raw KeyError which would be confusing to users
        raise NvshmemInvalid("Tried to retrieve MemoryResource for GPU with no NVSHMEM Allocations")
    buf = mr._mem_references.get(int(tensor.data_ptr()), {}).get("buffer")
    if buf is None:
        raise NvshmemInvalid("Tried to retrieve buffer from Tensor not tracked by nvshmem")
    return buf, (torch.numel(tensor) * tensor.element_size()), str(tensor.dtype)

def tensor(shape: Tuple[int] , dtype: dtype=float32, release=False, morder="C", except_on_del=True) -> Tensor:
    """
    Create a PyTorch tensor view on NVSHMEM-allocated memory with the given shape and dtype.

    This function allocates memory using NVSHMEM, wraps it with a DLPack tensor, and then
    converts it into a PyTorch tensor with the desired dtype and shape.

    Args:
        - shape (tuple or list of int): Shape of the desired tensor.
        - dtype (torch.dtype, optional): Data type of the tensor. Defaults to ``torch.float32``.
        - release (bool, optional): Do not track this buffer internally to NVSHMEM
                If True, it is the user's responsibility to hold references to the buffer until free() is called
                otherwise, deadlocks may occur.
        - morder (``str``, optional): The memory format to use. ``"C"`` for C-style, and ``"F"`` for "Fortran-style

    Returns:
        torch.Tensor: A PyTorch tensor view on the NVSHMEM-allocated buffer.

    Raises:
        RuntimeError: If NVSHMEM or PyTorch is not properly initialized or enabled.
    """
    if not _torch_enabled:
        return

    if morder not in ("C", "F"):
        raise NvshmemInvalid("Tensor with invalid memory format requested")

    if dtype is None:
        dtype = torch.get_default_dtype() 
    buf = buffer(get_size(shape, dtype), release=release, except_on_del=except_on_del)
    tensor = torch.utils.dlpack.from_dlpack(buf)
    view = tensor.view(dtype).view(shape)
    if morder == "F":
        # Compute Fortran-style (column-major) strides
        strides = [1]
        for dim in shape[:-1]:
            strides.append(strides[-1] * dim)
        view = view.as_strided(size=shape, stride=strides)
    return view

def bytetensor(shape: Tuple[int] , dtype: dtype=float32, release=False, morder="C", except_on_del=True) -> Tensor:
    """
    Create a PyTorch tensor from NVSHMEM-allocated memory with the given shape and dtype.

    This function allocates raw memory using NVSHMEM and wraps it in a PyTorch tensor
    without reshaping or changing the dtype view. Useful for low-level manipulation.

    Args:
       - shape (tuple or list of int): Shape of the desired tensor.
       - dtype (``str``, ``np.dtype``, or ``torch.dtype``, optional): Data type of the tensor. Defaults to ``"float32"``.
       - release (bool, optional): Do not track this buffer internally to NVSHMEM
                If True, it is the user's responsibility to hold references to the buffer until free() is called
                otherwise, deadlocks may occur.
        - morder (``str``, optional): The memory format to use. ``"C"`` for C-style, and ``"F"`` for "Fortran-style

    Returns:
        torch.Tensor: A raw PyTorch tensor referencing the NVSHMEM-allocated memory.

    Raises:
        RuntimeError: If NVSHMEM or PyTorch is not properly initialized or enabled.
    """
    if not _torch_enabled:
        return
    if dtype is None:
        dtype = torch.get_default_dtype()
    return tensor(shape, dtype=uint8, release=release, morder=morder, except_on_del=except_on_del)

def get_peer_tensor(tensor: Tensor, peer_pe: int=None) -> Tensor:
    """
    Return a Buffer based on the ``peer_buffer`` (wrapper of nvshmem_ptr) API
    """
    if not _torch_enabled:
        return
    buf, size, dtype  = tensor_get_buffer(tensor)
    peer_buf = nvshmem.core.get_peer_buffer(buf, peer_pe)
    return torch.utils.dlpack.from_dlpack(peer_buf).view(tensor.dtype).view(tensor.shape)

def get_multicast_tensor(team: Teams, tensor: Tensor) -> Tensor:
    """
    Returns a PyTorch Tensor view on multicast-accessible memory corresponding to the input tensor.

    This function takes a PyTorch tensor that wraps NVSHMEM-allocated memory and returns a new tensor
    that uses a Multicast Memory alias for that buffer, obtained via ``nvshmemx_mc_ptr``. The resulting tensor
    is suitable for use in GPU kernels that leverage multicast features such as NVSwitch-based SHARP collectives.
    
    The Tensor passed into it must be allocated by NVSHMEM4Py.

    IMPORTANT:
        - The returned tensor's memory cannot be accessed from the host (CPU). It is only valid for use
          in GPU kernels. Host-side access or copying is undefined behavior and may result in errors.

    NOTE: This function does not copy data. It provides a device-side view of the same underlying memory,
    but aliased for multicast access. Any modifications made through the returned tensor will affect the
    same memory region.

    Args:
        tensor (``torch.Tensor``): A PyTorch tensor backed by NVSHMEM-allocated memory.
        team (``Teams``): The NVSHMEM team for which multicast access is requested.

    Returns:
        ``torch.Tensor``: A PyTorch tensor view on the multicast alias of the NVSHMEM buffer.

    Raises:
        ``NvshmemInvalid``: If the input tensor is not backed by NVSHMEM memory or multicast is not supported.
        ``NvshmemError``: If the tensor's backing buffer is not properly tracked or initialized.
    """
    if not _torch_enabled:
        return
    buf, size, dtype  = tensor_get_buffer(tensor)
    mc_buf = nvshmem.core.get_multicast_buffer(team, buf)
    return torch.utils.dlpack.from_dlpack(mc_buf).view(tensor.dtype).view(tensor.shape)

def register_external_tensor(tensor: Tensor) -> Tensor:
    """
    Register an external tensor with NVSHMEM.
    """
    if not _torch_enabled:
        return
    buf = Buffer.from_handle(int(tensor.data_ptr()), get_size(tensor.shape, tensor.dtype))
    registered_buf = nvshmem.core.register_external_buffer(buf)
    return torch.utils.dlpack.from_dlpack(registered_buf).view(tensor.dtype).view(tensor.shape)

def unregister_external_tensor(tensor: Tensor) -> None:
    """
    Unregister an external tensor with NVSHMEM.
    """
    if not _torch_enabled:
        return
    buf, size, dtype = tensor_get_buffer(tensor)
    nvshmem.core.unregister_external_buffer(buf)

def free_tensor(tensor: Tensor) -> None:
    """
    Free an NVSHMEM-backed Torch Tensor

    Args:
        tensor (``torch.Tensor``): A PyTorch tensor backed by NVSHMEM memory.

    Returns:
        ``None``

    Raises:
        ``RuntimeError``: If NVSHMEM or PyTorch is not properly initialized or enabled.
    """
    if not _torch_enabled:
        return
    # Convert array to Buffer
    buf, sz, dtype = tensor_get_buffer(tensor)
    nvshmem.core.free(buf)
