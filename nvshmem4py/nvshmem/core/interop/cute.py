# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information
"""
The following are interoperability helpers for NVSHMEM4Py memory used with CuTe DSL

NOTE! CuTe DSL Tensor API aliases names with Torch tensors. Because of this, we do not import the cute module to the top level of the nvshmem.core module. 

Users of the CuTe DSL Tensor API should import the cute module directly via ``import nvshmem.core.interop.cute as cute``.
"""
import math
import ctypes
from enum import IntEnum
import nvshmem.core
from nvshmem.core.memory import buffer
from nvshmem.core.utils import get_size
from nvshmem.core._internal_tracking import _mr_references
from nvshmem.core.nvshmem_types import *
from nvshmem import bindings

from cuda.core import Buffer
from cuda.core import Device
from cuda.core import Stream

from typing import Tuple, Union

__all__ = [
    "bytetensor", "tensor", "free_tensor", "tensor_get_buffer", "get_peer_tensor", "get_multicast_tensor",
    "register_external_tensor", "unregister_external_tensor", "cleanup_cute", "cute_compile_helper"
]

try:
    from cutlass import cute
    from cutlass.cute.typing import Pointer, Boolean, Int32, Int64, Constexpr, Float32, Int8
    from cutlass.cute import Tensor
    from cutlass.cute.typing import dtype
    from cutlass.cute.runtime import from_dlpack
    _cute_enabled = True
except Exception as e:
    print(f"Error importing cutlass.cute: {e}")
    Tensor = None
    Pointer = None
    Boolean = None
    Int32 = None
    Int64 = None
    Constexpr = None
    Float32 = None
    Int8 = None
    _cute_enabled = False

_CUTE_TENSOR_BUFFERS = {}
_CUTE_MLIR_CONTEXT = None
_CUTE_MLIR_LOCATION = None
_CUTE_MLIR_MODULE = None

try:
    from cutlass._mlir import ir
    _cute_mlir_available = True
except Exception:
    ir = None
    _cute_mlir_available = False

_CUTE_DTYPE_NBYTES = {}
if _cute_enabled:
    _CUTE_DTYPE_NBYTES = {
        cute.Float16: 2,
        cute.BFloat16: 2,
        cute.Float32: 4,
        cute.Float64: 8,
        cute.Int8: 1,
        cute.Int16: 2,
        cute.Int32: 4,
        cute.Int64: 8,
        cute.Uint8: 1,
        cute.Uint16: 2,
        cute.Uint32: 4,
        cute.Uint64: 8,
        cute.Boolean: 1,
    }

# DLPack helpers for contiguous layout
try:
    import cuda.core._dlpack as _cuda_dlpack
    _dlpack_available = True
except Exception:
    _cuda_dlpack = None
    _dlpack_available = False


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_DLPACK_TENSOR_HOLDERS = {}


def _safe_get_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


# Map CuTe dtype to DLPack code/bits
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
class _DLPackTypeCode(IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BFLOAT = 4
    BOOL = 6


_CUTE_DLPACK_DTYPE = {
    cute.Float16: (_DLPackTypeCode.FLOAT, 16),
    cute.BFloat16: (_DLPackTypeCode.BFLOAT, 16),
    cute.Float32: (_DLPackTypeCode.FLOAT, 32),
    cute.Float64: (_DLPackTypeCode.FLOAT, 64),
    cute.Int8: (_DLPackTypeCode.INT, 8),
    cute.Int16: (_DLPackTypeCode.INT, 16),
    cute.Int32: (_DLPackTypeCode.INT, 32),
    cute.Int64: (_DLPackTypeCode.INT, 64),
    cute.Uint8: (_DLPackTypeCode.UINT, 8),
    cute.Uint16: (_DLPackTypeCode.UINT, 16),
    cute.Uint32: (_DLPackTypeCode.UINT, 32),
    cute.Uint64: (_DLPackTypeCode.UINT, 64),
    cute.Boolean: (_DLPackTypeCode.BOOL, 1),
}


@_DLManagedTensorDeleter
def _dlpack_deleter(ptr):
    if not ptr:
        return
    key = ctypes.addressof(ptr.contents)
    _DLPACK_TENSOR_HOLDERS.pop(key, None)


class _DLPackTensorWrapper:

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        return self._capsule

    def __dlpack_device__(self):
        # Not used by from_dlpack(), but provided for completeness
        return self._dlpack_device


def _make_dlpack_capsule(buf, shape, dtype, contiguous=True):
    if not _dlpack_available:
        raise NvshmemInvalid("cuda.core._dlpack is unavailable")
    if dtype not in _CUTE_DLPACK_DTYPE:
        raise NvshmemInvalid(f"Unsupported CuTe dtype for DLPack: {dtype}")
    # device info
    dev_type, dev_id = buf.__dlpack_device__()
    device = _DLDevice(int(dev_type), int(dev_id))
    code, bits = _CUTE_DLPACK_DTYPE[dtype]
    dl_dtype = _DLDataType(code=code, bits=bits, lanes=1)

    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    strides_ptr = None
    if not contiguous:
        raise NvshmemInvalid("Only contiguous layouts supported in DLPack path")

    dl_tensor = _DLTensor(
        data=ctypes.c_void_p(int(buf.handle)),
        device=device,
        ndim=len(shape),
        dtype=dl_dtype,
        shape=ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)),
        strides=strides_ptr,
        byte_offset=0,
    )

    managed = _DLManagedTensor(
        dl_tensor=dl_tensor,
        manager_ctx=None,
        deleter=_dlpack_deleter,
    )
    key = ctypes.addressof(managed)
    _DLPACK_TENSOR_HOLDERS[key] = (managed, shape_arr)

    py_capsule = ctypes.pythonapi.PyCapsule_New
    py_capsule.restype = ctypes.py_object
    py_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = py_capsule(ctypes.addressof(managed), b"dltensor", None)
    wrapper = _DLPackTensorWrapper(capsule)
    wrapper._dlpack_device = (int(dev_type), int(dev_id))
    return wrapper


def _ensure_cute_mlir():
    global _CUTE_MLIR_CONTEXT, _CUTE_MLIR_LOCATION, _CUTE_MLIR_MODULE
    if not _cute_enabled:
        return
    if not _cute_mlir_available:
        raise NvshmemInvalid("CuTe MLIR context is not available")
    if _CUTE_MLIR_CONTEXT is None:
        _CUTE_MLIR_CONTEXT = ir.Context()
        _CUTE_MLIR_CONTEXT.__enter__()
        _CUTE_MLIR_LOCATION = ir.Location.unknown()
        _CUTE_MLIR_LOCATION.__enter__()
        _CUTE_MLIR_MODULE = ir.Module.create()


def _normalize_shape(shape):
    if isinstance(shape, int):
        return (shape, )
    return tuple(shape)


def _compute_strides(shape, morder):
    if len(shape) == 0:
        return ()
    if morder == "F":
        strides = [1]
        for dim in shape[:-1]:
            strides.append(strides[-1] * dim)
        return tuple(strides)
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = running
        running *= shape[i]
    return tuple(strides)


def _size_in_bytes(shape, dtype):
    if dtype in _CUTE_DTYPE_NBYTES:
        return int(math.prod(shape) * _CUTE_DTYPE_NBYTES[dtype])
    try:
        return get_size(shape, dtype)
    except Exception as exc:
        raise NvshmemInvalid(f"Unsupported dtype for CuTe tensor: {dtype}") from exc


def _register_tensor_buffer(tensor, buf, dtype=None, shape=None, strides=None):
    _CUTE_TENSOR_BUFFERS[id(tensor)] = {
        "buffer": buf,
        "dtype": dtype,
        "element_type": dtype,  # Alias for consistency with tensor_get_buffer
        "shape": shape,
        "strides": strides,
    }


def _lookup_tensor_buffer(tensor):
    entry = _CUTE_TENSOR_BUFFERS.get(id(tensor))
    if entry is not None:
        return entry
    if isinstance(tensor, Tensor):
        return {
            "buffer": Buffer.from_handle(int(tensor.iterator), _size_in_bytes(tensor.shape, tensor.element_type)),
            "dtype": tensor.element_type,
            "shape": tensor.shape,
            "strides": tensor.stride,
        }
    return None


def _make_tensor_from_buffer(buf, shape, strides, dtype):
    _ensure_cute_mlir()
    with ir.InsertionPoint(_CUTE_MLIR_MODULE.body):
        ptr = cute.make_ptr(dtype, int(buf.handle), cute.AddressSpace.gmem)
        layout = cute.make_layout(tuple(shape), stride=tuple(strides))
        tensor = cute.make_tensor(ptr, layout)
    _register_tensor_buffer(tensor, buf, dtype=dtype, shape=shape, strides=strides)
    return tensor


def _is_tensor(tensor: Union[Tensor, object]) -> bool:
    """
    Helper function to check if an object is a CuTe DSL tensor
    This is used in collectives to avoid putting the complicated 
    import logic for CuTe DSL in any other file but this.
    """
    if not _cute_enabled:
        return False
    return isinstance(tensor, Tensor)


def tensor_get_buffer(tensor: Tensor) -> Tuple[Buffer, int, str]:
    """
    Get a nvshmem Buffer object from a CuTe DSL tensor object which was allocated with
    ``nvshmem.core.interop.cute.tensor()`` or ``nvshmem.core.interop.cute.bytetensor()``.
    Returns the buffer and the array's size.
    """
    entry = _lookup_tensor_buffer(tensor)
    if entry is None:
        raise NvshmemInvalid("Tried to retrieve buffer from Tensor not tracked by nvshmem")
    assert isinstance(tensor, Tensor), "Tried to register an external tensor that is not a CuTe DSL tensor"
    buf = entry.get("buffer")
    dtype = _safe_get_attr(tensor, "element_type")
    shape = entry.get("shape") or _safe_get_attr(tensor, "shape")
    size = _size_in_bytes(tuple(shape), dtype) if dtype is not None and shape is not None else buf.size
    return buf, size, dtype


def tensor(shape: Tuple[int], dtype: dtype = Float32, release=False, morder="C", except_on_del=True) -> Tensor:
    """
    Create a CuTe tensor view on NVSHMEM-allocated memory with the given shape and dtype.

    This function allocates memory using NVSHMEM and constructs a CuTe tensor with the
    desired dtype, shape, and memory order. For contiguous layouts, it uses DLPack.
    """
    if not _cute_enabled:
        return

    if morder not in ("C", "F"):
        raise NvshmemInvalid("Tensor with invalid memory format requested")

    if dtype is None:
        dtype = Float32

    shape = _normalize_shape(shape)
    strides = _compute_strides(shape, morder)
    size = _size_in_bytes(shape, dtype)
    buf = buffer(size, release=release, except_on_del=except_on_del)

    if morder == "C" and _dlpack_available:
        capsule = _make_dlpack_capsule(buf, shape, dtype, contiguous=True)
        tensor = from_dlpack(capsule)
        _register_tensor_buffer(tensor, buf, dtype=dtype, shape=shape, strides=strides)
        return tensor

    return _make_tensor_from_buffer(buf, shape, strides, dtype)


def bytetensor(shape: Tuple[int], dtype: dtype = Float32, release=False, morder="C", except_on_del=True) -> Tensor:
    """
    Create a CuTe tensor from NVSHMEM-allocated memory with the given shape and dtype.
    """
    if not _cute_enabled:
        return
    return tensor(shape, dtype=Int8, release=release, morder=morder, except_on_del=except_on_del)


def get_peer_tensor(tensor: Tensor, peer_pe: int = None) -> Tensor:
    """
    Return a Buffer based on the ``peer_buffer`` (wrapper of nvshmem_ptr) API
    """
    if not _cute_enabled:
        return
    buf, _, _ = tensor_get_buffer(tensor)
    peer_buf = nvshmem.core.get_peer_buffer(buf, peer_pe)
    return _make_tensor_from_buffer(peer_buf, tensor.shape, tensor.stride, tensor.element_type)


def get_multicast_tensor(team: Teams, tensor: Tensor) -> Tensor:
    """
    Returns a CuTe Tensor view on multicast-accessible memory corresponding to the input tensor.
    """
    if not _cute_enabled:
        return
    buf, _, _ = tensor_get_buffer(tensor)
    mc_buf = nvshmem.core.get_multicast_buffer(team, buf)
    return _make_tensor_from_buffer(mc_buf, tensor.shape, tensor.stride, tensor.element_type)


def register_external_tensor(tensor: Tensor) -> Tensor:
    """
    Register an external tensor with NVSHMEM.
    """
    if not _cute_enabled:
        return
    buf, _, _ = tensor_get_buffer(tensor)
    registered_buf = nvshmem.core.register_external_buffer(buf)
    return _make_tensor_from_buffer(registered_buf, tensor.shape, tensor.stride, tensor.element_type)


def unregister_external_tensor(tensor: Tensor) -> None:
    """
    Unregister an external tensor with NVSHMEM.
    """
    if not _cute_enabled:
        return
    buf, _, _ = tensor_get_buffer(tensor)
    nvshmem.core.unregister_external_buffer(buf)


def cleanup_cute():
    """
    Release CuTe MLIR context and DLPack holders to avoid teardown crashes.
    """
    global _CUTE_MLIR_CONTEXT, _CUTE_MLIR_LOCATION, _CUTE_MLIR_MODULE
    _DLPACK_TENSOR_HOLDERS.clear()
    try:
        if _CUTE_MLIR_LOCATION is not None:
            _CUTE_MLIR_LOCATION.__exit__(None, None, None)
    except Exception:
        pass
    try:
        if _CUTE_MLIR_CONTEXT is not None:
            _CUTE_MLIR_CONTEXT.__exit__(None, None, None)
    except Exception:
        pass
    _CUTE_MLIR_LOCATION = None
    _CUTE_MLIR_CONTEXT = None
    _CUTE_MLIR_MODULE = None


def free_tensor(tensor: Tensor) -> None:
    """
    Free an NVSHMEM-backed CuTe DSL Tensor

    Args:
        tensor (``cute.Tensor``): A CuTe tensor backed by NVSHMEM memory.

    Returns:
        ``None``

    Raises:
        ``NvshmemInvalid``: If the input tensor is not backed by NVSHMEM memory.
    """
    if not _cute_enabled:
        return
    # Convert array to Buffer
    buf, sz, dtype = tensor_get_buffer(tensor)
    nvshmem.core.free(buf)


def cute_compile_helper(kernel_fn, *args, **kwargs):
    """
    Helper function to compile a CuTe DSL kernel function.

    Finds the libnvshmem_device.bc library and compiles the kernel function with it.

    Runs nvshmem.core.library_init with the compiled kernel.

    Args:
        kernel_fn: A CuTe kernel function decorated with @cute.jit that contains a launcher.
                   The launcher should call a @cute.kernel with .launch().
        *args: Example arguments for compilation (tensors, etc.)
        **kwargs: Additional arguments passed to cute.compile()

    Returns:
        A tuple containing:
        - The compiled kernel function. (a callable object)
        - The nvshmem kernel object. (a NvshmemKernelObject) - the user should run ``nvshmem.core.library_finalize`` 
                  on this object after the kernel is executed.

    NOTE: This function assumes that the device being used as the NVSHMEM PE is already set current.
    """
    nvshmem_device_bc = nvshmem.core.find_device_bitcode_library()
    # Important: If _CUTE_MLIR_MODULE exists (from tensor creation via _make_tensor_from_buffer),
    # its context is active. cute.compile() checks "if ir.Context.current is None" and if not,
    # tries to access "ir.InsertionPoint.current" which raises an error if no insertion point is active.
    #
    # Solution: Temporarily exit our context and location if they're active, so cute.compile()
    # can create its own context. Then re-enter them after compilation.
    context_exited = False
    location_exited = False

    if _CUTE_MLIR_CONTEXT is not None:
        # Check if our context is currently active
        try:
            current_ctx = ir.Context.current
            if current_ctx is not None:
                # Our context might be active - exit it temporarily
                # We need to exit location first, then context
                if _CUTE_MLIR_LOCATION is not None:
                    _CUTE_MLIR_LOCATION.__exit__(None, None, None)
                    location_exited = True
                _CUTE_MLIR_CONTEXT.__exit__(None, None, None)
                context_exited = True
        except Exception:
            # If we can't check, assume context is not active
            pass

    try:
        # Build compile_kwargs with options and any user-provided kwargs
        compile_kwargs = {"options": f" --link-libraries={nvshmem_device_bc}"}
        if kwargs:
            compile_kwargs.update(kwargs)

        # Call cute.compile() - it will create its own context if needed
        compilerd_func = cute.compile(kernel_fn, *args, **compile_kwargs)
    finally:
        # Re-enter our context and location if we exited them
        if context_exited and _CUTE_MLIR_CONTEXT is not None:
            _CUTE_MLIR_CONTEXT.__enter__()
        if location_exited and _CUTE_MLIR_LOCATION is not None:
            _CUTE_MLIR_LOCATION.__enter__()
    # NOTE! assumes that device is already set current.
    dev = Device()
    compilerd_func = compilerd_func.to(dev.device_id)
    cuda_library = compilerd_func.jit_module.cuda_library
    nvshmem_kernel = nvshmem.core.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.core.library_init(nvshmem_kernel)
    return compilerd_func, nvshmem_kernel
