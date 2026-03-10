# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights ...
from nvshmem.bindings.device import numba as bindings
from nvshmem.core import Teams

import numpy as np
from numba.core.extending import overload
from numba.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, float16, Array,
)
from numba.cuda.types import bfloat16
from numba.core import types, cgutils
from numba.cuda.extending import intrinsic
from numba.np import arrayobj
from numba.cuda.np.numpy_support import carray
from numba.cuda import declare_device
from numba.types import CPointer, void
from numba.cuda.cudaimpl import lower


# Imports for binding shims
from numba.cuda import CUSource
from numba.cuda.typing.templates import ConcreteTemplate
import numba
from numba.types import uint16
from numba.cuda import declare_device
from numba.cuda.cudaimpl import lower_constant
from numba.types import float32
from numba.cuda.cudadecl import register_global
from numba import types
from numba.types import int16
from numba.types import float64
from numba.types import CPointer
from numba.core.typing import signature
from numba.cuda.cudadecl import register_attr
from numba.core.extending import lower_cast
from numba.types import uint64
from numba.cuda.types import bfloat16
from numba.cuda.vector_types import vector_types
from numba.cuda.cudaimpl import lower
import io
from numba.cuda.cudadecl import register
from numba.cuda.cudaimpl import lower_attr
from numba.types import int64
from numba.types import uint32
from numba.types import float16
from numba.types import int32
from numba.types import uint8
from numba.types import bool_
from numba.types import void
from numba.types import int8

__all__ = ["get_multicast_array", "get_peer_array"]

@intrinsic
def _array_data_noneptr(typingctx, arrty):
    """
    Return the data pointer of a Numba Array as an opaque none* (CPointer(none)).
    numba-CUDA 0.21+: arr.data is T**; one load yields T*. We then bitcast to none*.
    """
    if not isinstance(arrty, types.Array):
        return None
    sig = types.CPointer(types.uint8)(arrty)

    def codegen(cgctx, builder, sig, args):
        arr_val = args[0]
        ary = arrayobj.make_array(arrty)(cgctx, builder, arr_val)
        # Bitcast T* -> i8* (opaque)
        optr = builder.bitcast(ary.data, cgutils.voidptr_t)
        return optr

    return sig, codegen

@intrinsic
def _noneptr_as_typedptr(typingctx, optr, arrty):
    """
    Bitcast an opaque none* (void*) back to T* using arr.dtype.
    """
    if not (isinstance(optr, types.CPointer) and optr.dtype is types.none):
        return None
    if not isinstance(arrty, types.Array):
        return None

    ret = types.CPointer(arrty.dtype)
    sig = ret(optr, arrty)

    def codegen(cgctx, builder, sig, args):
        optr_val, _ = args
        ret_llty = cgctx.get_value_type(sig.return_type)  # T* type
        return builder.bitcast(optr_val, ret_llty)        # T*

    return sig, codegen

# -----------------------------------------------------------------------------
# Multicast Array
# -----------------------------------------------------------------------------
def get_multicast_array(team: Teams, array: Array):
    """
    Returns an array view on multicast-accessible memory corresponding to the input array.
    The Array passed into it must be allocated by NVSHMEM4Py.

    This is the Python array equivalent of `nvshmemx_mc_ptr` which returns a pointer into
    a peer's symmetric heap

    Args:
        array: Array - A symmetric array allocated by NVSHMEM
        team: Teams - A NVSHMEM Team to create the Multicast object across

    Returns:
        - A Numba ArrayView which represents the Multicast object

    NOTE: This function is only supported with the Numba Runtime. To enable it,
          set the environment variable ``NUMBA_CUDA_ENABLE_NRT=1``
    """
    pass


@overload(get_multicast_array)
def get_multicast_array_ol(team: Teams, arr: Array) -> Array:
    
    if arr == Array(dtype=int8, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=int8)
        return impl
    
    elif arr == Array(dtype=int16, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=int16)
        return impl
    
    elif arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=int32)
        return impl
    
    elif arr == Array(dtype=int64, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=int64)
        return impl
    
    elif arr == Array(dtype=uint8, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=uint8)
        return impl
    
    elif arr == Array(dtype=uint16, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=uint16)
        return impl
    
    elif arr == Array(dtype=uint32, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=uint32)
        return impl
    
    elif arr == Array(dtype=uint64, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=uint64)
        return impl
    
    elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=float32)
        return impl
    
    elif arr == Array(dtype=float64, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=float64)
        return impl
    
    elif arr == Array(dtype=float16, ndim=arr.ndim, layout=arr.layout):
        def impl(team: Teams, arr: Array) -> Array:
            base_optr = _array_data_noneptr(arr)         # i8*
            team_int32 = np.int32(team) 
            mc_optr   = mc_ptr(team_int32, base_optr)
            mc_tptr   = _noneptr_as_typedptr(mc_optr, arr)  # T*
            return carray(mc_tptr, arr.shape, dtype=float16)
        return impl
    
    return None


# -----------------------------------------------------------------------------
# Peer Array
# -----------------------------------------------------------------------------
def get_peer_array(arr: Array, pe: int):
    """
    Returns an array view of a peer buffer associated with an NVSHMEM-allocated object.

    This is the Python array equivalent of `nvshmem_ptr` which returns a pointer into a peer's symmetric heap

    Args:
        array: Array - A symmetric array allocated by NVSHMEM
        pe: int - The remote PE to retrieve an ArrayView into

    Returns:
        - A Numba ArrayView which represents the peer object

    NOTE: This function is only supported with the Numba Runtime. To enable it,
          set the environment variable ``NUMBA_CUDA_ENABLE_NRT=1``
    """
    pass


@overload(get_peer_array)
def get_peer_array_ol(arr: Array, pe: int) -> Array:
    
    if arr == Array(dtype=int8, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=int8)
        return impl
    
    elif arr == Array(dtype=int16, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=int16)
        return impl
    
    elif arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=int32)
        return impl
    
    elif arr == Array(dtype=int64, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=int64)
        return impl
    
    elif arr == Array(dtype=uint8, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=uint8)
        return impl
    
    elif arr == Array(dtype=uint16, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=uint16)
        return impl
    
    elif arr == Array(dtype=uint32, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=uint32)
        return impl
    
    elif arr == Array(dtype=uint64, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=uint64)
        return impl
    
    elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=float32)
        return impl
    
    elif arr == Array(dtype=float64, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=float64)
        return impl
    
    elif arr == Array(dtype=float16, ndim=arr.ndim, layout=arr.layout):
        def impl(arr: Array, pe: int) -> Array:
            base_optr = _array_data_noneptr(arr)          # i8*
            pe32      = np.int32(pe)                      # match (none*, int32)
            peer_optr = ptr(base_optr, pe32)     # (uint8*, int32) -> none*
            peer_tptr = _noneptr_as_typedptr(peer_optr, arr)  # T*
            return carray(peer_tptr, arr.shape, dtype=float16)
        return impl
    
    return None


###
# The below are shim bindings for nvshmem_ptr and nvshmemx_mc_ptr. 
# We need them because Numbast and LLVM doesn't support passing void*s fully. 
# We exploit our knowledge of the fact that these APIs always have void* as a one-degree pointer (e.g. a void* not a void**)
# So it is safe to use them this way. However, this is not a general solution so we can't use this approach inside Numbast
# This means we have to ship the bindings as part of nvshmem.core as is below
# TODO: remove the following section if there's a general solution in Numbast for passing void*s
###


def ptr():
    pass

def _lower__Z11nvshmem_ptr_nbst(shim_stream, shim_obj):


    shim_raw_str = """
    extern "C" __device__ int
    _Z11nvshmem_ptr_nbst(void * &retval , void ** ptr, int* pe) {
        retval = nvshmem_ptr(*ptr, *pe);
        return 0;
    }
        """


    _Z11nvshmem_ptr_nbst = declare_device(
        '_Z11nvshmem_ptr_nbst',
        CPointer(void)(
            CPointer(CPointer(uint8)), CPointer(int32)
        )
    )
    

    def _Z11nvshmem_ptr_nbst_caller(arg_0, arg_1):
        return _Z11nvshmem_ptr_nbst(arg_0, arg_1)
    

    @lower(ptr, CPointer(uint8), int32)
    def impl(context, builder, sig, args):
        context.active_code_library.add_linking_file(shim_obj)
        shim_stream.write_with_key("_Z11nvshmem_ptr_nbst", shim_raw_str)

        val_types = [context.get_value_type(arg) for arg in sig.args]
        ptrs = [builder.alloca(val_type) for val_type in val_types]
        for ptr, ty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

        return context.compile_internal(
            builder,
            _Z11nvshmem_ptr_nbst_caller,
            signature(CPointer(void), CPointer(CPointer(uint8)), CPointer(int32)),
            ptrs,
        )
    

_lower__Z11nvshmem_ptr_nbst(bindings._numbast.shim_stream, bindings._numbast.shim_obj)


@register
class _typing_ptr(ConcreteTemplate):
    key = globals()["ptr"]
    cases = [
        signature(CPointer(void), CPointer(uint8), int32),
    ]


register_global(ptr, types.Function(_typing_ptr))

def mc_ptr():
    pass



def _lower__Z11nvshmemx_mc_ptr_nbst(shim_stream, shim_obj):


    shim_raw_str = """
    extern "C" __device__ int
    _Z11nvshmemx_mc_ptr_nbst(void * &retval , int* team, void ** ptr) {
        retval = nvshmemx_mc_ptr(*team, *ptr);
        return 0;
    }
        """


    _Z11nvshmemx_mc_ptr_nbst = declare_device(
        '_Z11nvshmemx_mc_ptr_nbst',
        CPointer(void)(
            CPointer(int32), CPointer(CPointer(uint8))
        )
    )
    

    def _Z11nvshmemx_mc_ptr_nbst_caller(arg_0, arg_1):
        return _Z11nvshmemx_mc_ptr_nbst(arg_0, arg_1)
    

    @lower(mc_ptr, int32, CPointer(uint8))
    def impl(context, builder, sig, args):
        context.active_code_library.add_linking_file(shim_obj)
        shim_stream.write_with_key("_Z11nvshmemx_mc_ptr_nbst", shim_raw_str)

        val_types = [context.get_value_type(arg) for arg in sig.args]
        ptrs = [builder.alloca(val_type) for val_type in val_types]
        for ptr, ty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

        return context.compile_internal(
            builder,
            _Z11nvshmemx_mc_ptr_nbst_caller,
            signature(CPointer(void), CPointer(int32), CPointer(CPointer(uint8))),
            ptrs,
        )
    

_lower__Z11nvshmemx_mc_ptr_nbst(bindings._numbast.shim_stream, bindings._numbast.shim_obj)


@register
class _typing_mc_ptr(ConcreteTemplate):
    key = globals()["mc_ptr"]
    cases = [
        signature(CPointer(void), int32, CPointer(uint8)),
    ]


register_global(mc_ptr, types.Function(_typing_mc_ptr))