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
We cannot use the generated bindings for these functions because Cutlass Python does not have an equivalent of a Void*

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
    base_ptr = tensor.iterator
    peer_ptr = nvshmem_ptr(cutlass.Int64(base_ptr.toint()), pe)
    return _make_tensor_from_ptr(peer_ptr, tensor)


@cute.jit
def get_multicast_tensor(team: cutlass.Int32, tensor: cute.Tensor):
    team = _resolve_team(team)
    base_ptr = tensor.iterator.toint()
    multicast_ptr = nvshmemx_mc_ptr(team, cutlass.Int64(base_ptr))
    return _make_tensor_from_ptr(multicast_ptr, tensor)
