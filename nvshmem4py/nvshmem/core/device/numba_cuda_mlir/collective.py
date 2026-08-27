# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nvshmem.core
import nvshmem.bindings.device.numba_cuda_mlir as bindings

import cffi
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir import types
from numba_cuda_mlir.types import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, float16, Array
# # bfloat16 is not currently exposed by numba_cuda_mlir.types


__all__ = [ "sync_block", "sync_warp", "sync", "sync_all", "sync_all_block", "sync_all_warp",
            "barrier", "barrier_block", "barrier_warp", "barrier_all", "barrier_all_block", "barrier_all_warp",
            "reduce", "reduce_block", "reduce_warp",
            "reducescatter", "reducescatter_block", "reducescatter_warp",
            "fcollect", "fcollect_block", "fcollect_warp",
            "broadcast", "broadcast_block", "broadcast_warp",
            "alltoall", "alltoall_block", "alltoall_warp"]

# TODO: create a global ffi object for other high level bindings to use
ffi = cffi.FFI()

def sync(team: nvshmem.core.Teams):
    """
    Executes a sync across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def sync_block(team: nvshmem.core.Teams):
    """
    Executes a block-wide sync across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def sync_warp(team: nvshmem.core.Teams):
    """
    Executes a warp-wide sync across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def sync_all():
    """
    Executes a thread-wide sync across all PEs in the runtime.
    """
    pass

def sync_all_block():
    """
    Executes a block-wide sync across all PEs in the runtime.
    """
    pass

def sync_all_warp():
    """
    Executes a warp-wide sync across all PEs in the runtime.
    """
    pass

def barrier(team: nvshmem.core.Teams):
    """
    Executes a thread-wide barrier across the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def barrier_block(team: nvshmem.core.Teams):
    """
    Executes a block-wide barrier across all threads in the block.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def barrier_warp(team: nvshmem.core.Teams):
    """
    Executes a warp-wide barrier across all threads in the warp.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
    """
    pass

def barrier_all():
    """
    Executes a barrier across all PEs in the runtime.
    """
    pass


def barrier_all_block():
    """
    Executes a block-wide barrier across all PEs in the runtime.
    """
    pass

def barrier_all_warp():
    """
    Executes a warp-wide barrier across all PEs in the runtime.
    """
    pass

def reduce(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a reduction from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduction routine.
        - ``src_array`` (``Array``): Symmetric array that contains at least one element for each separate reduction routine.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    In case that ``src_array`` and ``dst_array`` are of different sizes, the size of the smaller array is reduced.
    """
    pass

def reduce_block(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a block-wide reduction from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduction routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    In case that ``src_array`` and ``dst_array`` are of different sizes, the size of the smaller array is reduced.
    """
    pass

def reduce_warp(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a warp-wide reduction from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduction routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    In case that ``src_array`` and ``dst_array`` are of different sizes, the size of the smaller array is reduced.
    """
    pass

def reducescatter(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a thread-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduce-scatter routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    Array size is taken from the ``dst_array``. The ``dst_array`` must be >= ``(src_array.size // team_n_pes(team))``.
    """
    pass

def reducescatter_block(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a block-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduce-scatter routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    Array size is taken from the ``dst_array``. The ``src_array`` must be >= ``(dst_array.size * team_n_pes(team))``.
    """
    pass

def reducescatter_warp(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, op: str):
    """
    Performs a warp-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the reduce-scatter routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.
        - ``op`` (``str``): String representing the reduction operator.

    Supported reduction operators:
    See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions
    for supported reduction operators.

    Note:
    Array size is taken from the ``dst_array``. The ``src_array`` must be >= ``(dst_array.size * team_n_pes(team))``.
    """
    pass

def fcollect(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a thread-scoped fcollect operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the fcollect routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.

    Note:
    Array size is taken from the ``src_array``. The ``dst_array`` must be >= ``(src_array.size * team_n_pes(team))``.
    """
    pass

def fcollect_block(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a block-scoped fcollect operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the fcollect routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.

    Note:
    Array size is taken from the ``src_array``. The ``dst_array`` must be >= ``(src_array.size * team_n_pes(team))``.
    """
    pass

def fcollect_warp(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a warp-scoped fcollect operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Symmetric array to store the result of the fcollect routine.
        - ``src_array`` (``Array``): Symmetric array that contains the source data.

    Note:
    Array size is taken from the ``src_array``. The ``dst_array`` must be >= ``(src_array.size * team_n_pes(team))``.
    """
    pass

def broadcast(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, root: int=0):
    """
    Performs a thread-scoped broadcast operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
        - ``root`` (``int``): Root PE for the broadcast.
    """
    pass

def broadcast_block(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, root: int=0):
    """
    Performs a block-scoped broadcast operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
        - ``root`` (``int``): Root PE for the broadcast.
    """
    pass

def broadcast_warp(team: nvshmem.core.Teams, dst_array: Array, src_array: Array, root: int=0):
    """
    Performs a warp-scoped broadcast operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
        - ``root`` (``int``): Root PE for the broadcast.
    """
    pass

def alltoall(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a thread-scoped alltoall operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
    """
    pass

def alltoall_block(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a block-scoped alltoall operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
    """
    pass

def alltoall_warp(team: nvshmem.core.Teams, dst_array: Array, src_array: Array):
    """
    Performs a warp-scoped alltoall operation from src_array to dst_array across all PEs in the team.

    Args:
        - ``team`` (``Teams``): NVSHMEM team handle.
        - ``dst_array`` (``Array``): Destination symmetric array.
        - ``src_array`` (``Array``): Source symmetric array.
    """
    pass

# sync variations
@overload(sync_block, inline="always", typing_registry=typing_registry)
def sync_block_ol(team):
    def impl(team):
        bindings.team_sync_block(int32(team))
    return impl
@overload(sync_warp, inline="always", typing_registry=typing_registry)
def sync_warp_ol(team):
    def impl(team):
        bindings.team_sync_warp(int32(team))
    return impl
@overload(sync, inline="always", typing_registry=typing_registry)
def sync_ol(team):
    def impl(team):
        bindings.team_sync(int32(team))
    return impl

# sync_all variations
@overload(sync_all_block, inline="always", typing_registry=typing_registry)
def sync_all_block_ol():
    def impl():
        bindings.sync_all_block()
    return impl
@overload(sync_all_warp, inline="always", typing_registry=typing_registry)
def sync_all_warp_ol():
    def impl():
        bindings.sync_all_warp()
    return impl
@overload(sync_all, inline="always", typing_registry=typing_registry)
def sync_all_ol():
    def impl():
        bindings.sync_all()
    return impl

# barrier variations
@overload(barrier_block, inline="always", typing_registry=typing_registry)
def barrier_block_ol(team):
    def impl(team):
        bindings.barrier_block(int32(team))
    return impl
@overload(barrier_warp, inline="always", typing_registry=typing_registry)
def barrier_warp_ol(team):
    def impl(team):
        bindings.barrier_warp(int32(team))
    return impl
@overload(barrier, inline="always", typing_registry=typing_registry)
def barrier_ol(team):
    def impl(team):
        bindings.barrier(int32(team))
    return impl

# barrier_all variations
@overload(barrier_all_block, inline="always", typing_registry=typing_registry)
def barrier_all_block_ol():
    def impl():
        bindings.barrier_all_block()
    return impl
@overload(barrier_all_warp, inline="always", typing_registry=typing_registry)
def barrier_all_warp_ol():
    def impl():
        bindings.barrier_all_warp()
    return impl
@overload(barrier_all, inline="always", typing_registry=typing_registry)
def barrier_all_ol():
    def impl():
        bindings.barrier_all()
    return impl

# reduce variations
@overload(reduce_block, inline="always", typing_registry=typing_registry)
def reduce_block_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reduce_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
@overload(reduce_warp, inline="always", typing_registry=typing_registry)
def reduce_warp_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reduce_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
@overload(reduce, inline="always", typing_registry=typing_registry)
def reduce_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reduce(int32(team), dst_ptr, src_ptr, nelem)
                return impl

# reducescatter variations
@overload(reducescatter_block, inline="always", typing_registry=typing_registry)
def reducescatter_block_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reducescatter_block(int32(team), dst_ptr, src_ptr, nelem)
                return impl
@overload(reducescatter_warp, inline="always", typing_registry=typing_registry)
def reducescatter_warp_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reducescatter_warp(int32(team), dst_ptr, src_ptr, nelem)
                return impl
@overload(reducescatter, inline="always", typing_registry=typing_registry)
def reducescatter_ol(team, dst_array, src_array, op):
    if isinstance(op, types.Literal) and isinstance(op.literal_value, str):
        if isinstance(dst_array.dtype, types.Integer) and op.literal_value not in { "and", "or", "xor", "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for integer data type")
        if isinstance(dst_array.dtype, types.Float) and op.literal_value not in { "min", "max", "sum", "prod" }:
            raise ValueError(f"Invalid reduction operator: {op.literal_value} for float data type")
        if op.literal_value == "min":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_min_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "max":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_max_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "sum":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_sum_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "prod":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.float_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.double_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.half_prod_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        if op.literal_value == "and":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_and_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "or":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_or_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
        elif op.literal_value == "xor":
            if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int8_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int16_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int32_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.int64_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint8_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint16_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint32_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl
            elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
                def impl(team, dst_array, src_array, op):
                    dst_ptr = ffi.from_buffer(dst_array)
                    src_ptr = ffi.from_buffer(src_array)
                    nelem = uint64(dst_array.size)
                    bindings.uint64_xor_reducescatter(int32(team), dst_ptr, src_ptr, nelem)
                return impl

# fcollect variations
@overload(fcollect_block, inline="always", typing_registry=typing_registry)
def fcollect_block_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int8_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int16_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int32_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int64_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint8_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint16_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint32_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint64_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.float_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.double_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.half_fcollect_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
@overload(fcollect_warp, inline="always", typing_registry=typing_registry)
def fcollect_warp_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int8_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int16_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int32_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int64_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint8_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint16_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint32_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint64_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.float_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.double_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.half_fcollect_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
@overload(fcollect, inline="always", typing_registry=typing_registry)
def fcollect_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int8_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int16_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int32_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.int64_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint8_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint16_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint32_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.uint64_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.float_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.double_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(src_array.size)
            bindings.half_fcollect(int32(team), dst_ptr, src_ptr, nelem)
        return impl

# broadcast variations
@overload(broadcast_block, inline="always", typing_registry=typing_registry)
def broadcast_block_ol(team, dst_array, src_array, root=0):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int8_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int16_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int32_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int64_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint8_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint16_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint32_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint64_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.float_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.double_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.half_broadcast_block(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
@overload(broadcast_warp, inline="always", typing_registry=typing_registry)
def broadcast_warp_ol(team, dst_array, src_array, root=0):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int8_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int16_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int32_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int64_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint8_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint16_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint32_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint64_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.float_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.double_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.half_broadcast_warp(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
@overload(broadcast, inline="always", typing_registry=typing_registry)
def broadcast_ol(team, dst_array, src_array, root=0):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int8_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int16_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int32_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.int64_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint8_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint16_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint32_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.uint64_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.float_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.double_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array, root=0):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            nelem = uint64(dst_array.size)
            src_nelem = uint64(src_array.size)
            if src_nelem < nelem:
                nelem = src_nelem
            bindings.half_broadcast(int32(team), dst_ptr, src_ptr, nelem, int32(root))
        return impl

# alltoall variations
@overload(alltoall_block, inline="always", typing_registry=typing_registry)
def alltoall_block_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int8_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int16_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int32_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int64_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint8_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint16_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint32_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint64_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.float_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.double_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.half_alltoall_block(int32(team), dst_ptr, src_ptr, nelem)
        return impl
@overload(alltoall_warp, inline="always", typing_registry=typing_registry)
def alltoall_warp_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int8_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int16_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int32_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int64_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint8_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint16_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint32_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint64_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.float_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.double_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.half_alltoall_warp(int32(team), dst_ptr, src_ptr, nelem)
        return impl
@overload(alltoall, inline="always", typing_registry=typing_registry)
def alltoall_ol(team, dst_array, src_array):
    if dst_array == Array(dtype=int8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int8_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int16_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int32_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=int64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=int64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.int64_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint8, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint8, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint8_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint16_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint32_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=uint64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=uint64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.uint64_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float32, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float32, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.float_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float64, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float64, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.double_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl
    elif dst_array == Array(dtype=float16, ndim=dst_array.ndim, layout=dst_array.layout) and src_array == Array(dtype=float16, ndim=src_array.ndim, layout=src_array.layout):
        def impl(team, dst_array, src_array):
            dst_ptr = ffi.from_buffer(dst_array)
            src_ptr = ffi.from_buffer(src_array)
            # alltoall is a special case because it takes the per-device count.
            nelem = uint64(src_array.size) // uint64(bindings.team_n_pes(int32(team)))
            bindings.half_alltoall(int32(team), dst_ptr, src_ptr, nelem)
        return impl