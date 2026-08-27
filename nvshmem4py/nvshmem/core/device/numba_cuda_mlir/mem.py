# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cffi

import nvshmem.bindings.device.numba_cuda_mlir as bindings
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.extending import (
    lowering_registry,
    overload,
    refresh_registries,
    typing_registry,
)
from numba_cuda_mlir.numba_cuda.typing import signature
from numba_cuda_mlir.numba_cuda.typing.templates import ConcreteTemplate
from numba_cuda_mlir.types import (
    Array,
    CPointer,

    int8,

    int16,

    int32,

    int64,

    uint8,

    uint16,

    uint32,

    uint64,

    float32,

    float64,

    float16,

)

__all__ = ["get_multicast_array", "get_peer_array"]

ffi = cffi.FFI()
lower = lowering_registry.lower
register_global = typing_registry.register_global


def _ptr():
    pass


_ptr_shim = r"""
extern "C" __device__ int
_Z11nvshmem_ptr_nbst(void * &retval, void **ptr, int *pe) {
    retval = nvshmem_ptr(*ptr, *pe);
    return 0;
}
"""

bindings._numbast.shim_writer.write_to_shim(_ptr_shim, "_Z11nvshmem_ptr_nbst")


@lower(_ptr, CPointer(int8), int32)
@lower(_ptr, CPointer(int16), int32)
@lower(_ptr, CPointer(int32), int32)
@lower(_ptr, CPointer(int64), int32)
@lower(_ptr, CPointer(uint8), int32)
@lower(_ptr, CPointer(uint16), int32)
@lower(_ptr, CPointer(uint32), int32)
@lower(_ptr, CPointer(uint64), int32)
@lower(_ptr, CPointer(float32), int32)
@lower(_ptr, CPointer(float64), int32)
@lower(_ptr, CPointer(float16), int32)
def _lower_ptr(builder, target, args, kws):
    callconv = bindings._numbast.FunctionCallConv(
        itanium_mangled_name="_Z11nvshmem_ptr",
        shim_writer=bindings._numbast.shim_writer,
        shim_code=_ptr_shim,
        arg_is_ref=[False, False],
        intent_plan=None,
        out_return_types=None,
        cxx_return_type=None,
    )
    bindings._numbast._numbast_link_shim(builder, bindings._numbast.shim_obj)
    return callconv(builder, target, args, kws)


class _TypingPtr(ConcreteTemplate):
    key = _ptr
    cases = [

        signature(types.voidptr, CPointer(int8), int32),

        signature(types.voidptr, CPointer(int16), int32),

        signature(types.voidptr, CPointer(int32), int32),

        signature(types.voidptr, CPointer(int64), int32),

        signature(types.voidptr, CPointer(uint8), int32),

        signature(types.voidptr, CPointer(uint16), int32),

        signature(types.voidptr, CPointer(uint32), int32),

        signature(types.voidptr, CPointer(uint64), int32),

        signature(types.voidptr, CPointer(float32), int32),

        signature(types.voidptr, CPointer(float64), int32),

        signature(types.voidptr, CPointer(float16), int32),

    ]


register_global(_ptr, types.Function(_TypingPtr))


def _mc_ptr():
    pass


_mc_ptr_shim = r"""
extern "C" __device__ int
_Z15nvshmemx_mc_ptr_nbst(void * &retval, int *team, void **ptr) {
    retval = nvshmemx_mc_ptr(*team, *ptr);
    return 0;
}
"""

bindings._numbast.shim_writer.write_to_shim(_mc_ptr_shim, "_Z15nvshmemx_mc_ptr_nbst")


@lower(_mc_ptr, int32, CPointer(int8))
@lower(_mc_ptr, int32, CPointer(int16))
@lower(_mc_ptr, int32, CPointer(int32))
@lower(_mc_ptr, int32, CPointer(int64))
@lower(_mc_ptr, int32, CPointer(uint8))
@lower(_mc_ptr, int32, CPointer(uint16))
@lower(_mc_ptr, int32, CPointer(uint32))
@lower(_mc_ptr, int32, CPointer(uint64))
@lower(_mc_ptr, int32, CPointer(float32))
@lower(_mc_ptr, int32, CPointer(float64))
@lower(_mc_ptr, int32, CPointer(float16))
def _lower_mc_ptr(builder, target, args, kws):
    callconv = bindings._numbast.FunctionCallConv(
        itanium_mangled_name="_Z15nvshmemx_mc_ptr",
        shim_writer=bindings._numbast.shim_writer,
        shim_code=_mc_ptr_shim,
        arg_is_ref=[False, False],
        intent_plan=None,
        out_return_types=None,
        cxx_return_type=None,
    )
    bindings._numbast._numbast_link_shim(builder, bindings._numbast.shim_obj)
    return callconv(builder, target, args, kws)


class _TypingMcPtr(ConcreteTemplate):
    key = _mc_ptr
    cases = [

        signature(types.voidptr, int32, CPointer(int8)),

        signature(types.voidptr, int32, CPointer(int16)),

        signature(types.voidptr, int32, CPointer(int32)),

        signature(types.voidptr, int32, CPointer(int64)),

        signature(types.voidptr, int32, CPointer(uint8)),

        signature(types.voidptr, int32, CPointer(uint16)),

        signature(types.voidptr, int32, CPointer(uint32)),

        signature(types.voidptr, int32, CPointer(uint64)),

        signature(types.voidptr, int32, CPointer(float32)),

        signature(types.voidptr, int32, CPointer(float64)),

        signature(types.voidptr, int32, CPointer(float16)),

    ]


register_global(_mc_ptr, types.Function(_TypingMcPtr))
refresh_registries()


def get_multicast_array(team, array):
    """Return a device array view of an NVSHMEM team's multicast memory.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``array`` (``Array``): Symmetric source array.

    Returns:
        ``Array``: A device array view of the team's multicast memory.
    """
    pass


@overload(get_multicast_array, inline="always", typing_registry=typing_registry)
def get_multicast_array_ol(team, arr):
    if arr == Array(dtype=int8, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=int8)

        return impl
    elif arr == Array(dtype=int16, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=int16)

        return impl
    elif arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=int32)

        return impl
    elif arr == Array(dtype=int64, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=int64)

        return impl
    elif arr == Array(dtype=uint8, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=uint8)

        return impl
    elif arr == Array(dtype=uint16, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=uint16)

        return impl
    elif arr == Array(dtype=uint32, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=uint32)

        return impl
    elif arr == Array(dtype=uint64, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=uint64)

        return impl
    elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=float32)

        return impl
    elif arr == Array(dtype=float64, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=float64)

        return impl
    elif arr == Array(dtype=float16, ndim=arr.ndim, layout=arr.layout):

        def impl(team, arr):
            base_ptr = ffi.from_buffer(arr)
            multicast_ptr = _mc_ptr(int32(team), base_ptr)
            return cuda.carray(multicast_ptr, arr.shape, dtype=float16)

        return impl
    


def get_peer_array(arr, pe):
    """Return a device array view of symmetric memory on a peer PE.

    Args:
        - ``arr`` (``Array``): Local symmetric array.
        - ``pe`` (``int``): Global rank of the target PE.

    Returns:
        ``Array``: A device array view of the target PE's symmetric memory.
    """
    pass


@overload(get_peer_array, inline="always", typing_registry=typing_registry)
def get_peer_array_ol(arr, pe):
    if arr == Array(dtype=int8, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=int8)

        return impl
    elif arr == Array(dtype=int16, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=int16)

        return impl
    elif arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=int32)

        return impl
    elif arr == Array(dtype=int64, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=int64)

        return impl
    elif arr == Array(dtype=uint8, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=uint8)

        return impl
    elif arr == Array(dtype=uint16, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=uint16)

        return impl
    elif arr == Array(dtype=uint32, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=uint32)

        return impl
    elif arr == Array(dtype=uint64, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=uint64)

        return impl
    elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=float32)

        return impl
    elif arr == Array(dtype=float64, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=float64)

        return impl
    elif arr == Array(dtype=float16, ndim=arr.ndim, layout=arr.layout):

        def impl(arr, pe):
            base_ptr = ffi.from_buffer(arr)
            peer_ptr = _ptr(base_ptr, int32(pe))
            return cuda.carray(peer_ptr, arr.shape, dtype=float16)

        return impl
    