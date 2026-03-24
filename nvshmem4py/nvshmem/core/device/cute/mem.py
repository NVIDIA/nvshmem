# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import cutlass
from cutlass import cute
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.base_dsl.typing import cast as cute_cast

__all__ = ["get_peer_tensor", "get_multicast_tensor"]
"""
We cannot use the bindings in collective.py for these functions because Cutlass Python does not have an equivalent of a Void*

Theoretically, we could cast the returned void* , but it requires so much complicated type inference that it'd be too confusing to debug.
Since there are just two typeless functions, it's easier to just have explicit bindings for them
"""

nvshmem_ptr = cute.ffi(name="nvshmem_ptr", params_types=[cutlass.Int64, cutlass.Int32], return_type=cutlass.Int64)
nvshmemx_mc_ptr = cute.ffi(name="nvshmemx_mc_ptr",
                           params_types=[cutlass.Int32, cutlass.Int64],
                           return_type=cutlass.Int64)


@cute.jit
def _resolve_team(team):
    return cute_cast(team, cutlass.Int32)


@cute.jit
def _make_tensor_from_ptr(ptr, tensor):
    dtype = tensor.element_type
    pointer = cute.make_ptr(dtype, ptr, cute.AddressSpace.gmem)
    return cute.make_tensor(pointer, tensor.layout)


@cute.jit
def get_peer_tensor(tensor: cute.Tensor, pe: cutlass.Int32):
    """
    Returns a CuTe tensor view that aliases the symmetric tensor ``tensor``
    on a remote PE ``pe``.

    Wraps ``nvshmem_ptr`` to translate the base pointer of ``tensor`` to the
    address of the corresponding symmetric allocation on PE ``pe``, then
    reconstructs a tensor with the same layout over the remote memory region.
    The returned tensor can be used as a destination or source in NVSHMEM RMA
    operations without additional pointer arithmetic.

    Args:
        - ``tensor`` (``cute.Tensor``): A CuTe tensor view backed by a
          symmetric (NVSHMEM-allocated) buffer on the calling PE.  The layout
          of the returned tensor matches the layout of this argument.
        - ``pe`` (``cutlass.Int32``): Target PE whose symmetric copy of
          ``tensor`` is requested.

    Returns:
        A ``cute.Tensor`` with the same dtype and layout as ``tensor`` but
        whose base pointer refers to the symmetric allocation on PE ``pe``.

    Note:
        The symmetric object must have been allocated with
        ``nvshmem_malloc`` (or equivalent) so that a corresponding
        allocation exists at the same symmetric offset on every PE.
        If PE ``pe`` is the calling PE this function returns a tensor
        equivalent to ``tensor`` itself.
    """
    base_ptr = tensor.iterator
    peer_ptr = nvshmem_ptr(cutlass.Int64(base_ptr.toint()), pe)
    return _make_tensor_from_ptr(peer_ptr, tensor)


@cute.jit
def get_multicast_tensor(team: cutlass.Int32, tensor: cute.Tensor):
    """
    Returns a CuTe tensor view that aliases the symmetric tensor ``tensor``
    via the multicast address for ``team``.

    Wraps ``nvshmemx_mc_ptr`` to obtain the multicast virtual address
    corresponding to the symmetric allocation backing ``tensor`` within
    ``team``, then reconstructs a tensor with the same layout over that
    multicast memory region.  Writes to the returned tensor are delivered to
    all PEs in ``team`` simultaneously, enabling efficient one-to-many
    communication patterns.

    Args:
        - ``team`` (``cutlass.Int32``): NVSHMEM team handle identifying the
          set of PEs that share the multicast mapping.  The calling PE must
          be a member of ``team``.
        - ``tensor`` (``cute.Tensor``): A CuTe tensor view backed by a
          symmetric (NVSHMEM-allocated) buffer on the calling PE.  The layout
          of the returned tensor matches the layout of this argument.

    Returns:
        A ``cute.Tensor`` with the same dtype and layout as ``tensor`` but
        whose base pointer is the multicast virtual address for the symmetric
        allocation within ``team``.

    Note:
        Multicast support requires hardware and driver support (NVLink
        multicast or equivalent).  The symmetric object must have been
        allocated with multicast support enabled.  All PEs in ``team`` must
        participate in the multicast setup before any PE uses the returned
        tensor.
    """
    team = _resolve_team(team)
    base_ptr = tensor.iterator.toint()
    multicast_ptr = nvshmemx_mc_ptr(team, cutlass.Int64(base_ptr))
    return _make_tensor_from_ptr(multicast_ptr, tensor)
