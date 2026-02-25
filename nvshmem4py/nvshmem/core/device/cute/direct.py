# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import nvshmem.bindings.device.cute as bindings
from nvshmem.core import Teams

__all__ = ["my_pe", "team_my_pe", "team_n_pes", "n_pes", "barrier_all", "sync_all", "signal_op", "signal_wait"]

from cutlass import cute
import cutlass
from cutlass.base_dsl.typing import cast as cute_cast


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@cute.jit
def my_pe():
    return bindings.my_pe()


@cute.jit
def team_my_pe(team):
    return bindings.team_my_pe(team)


@cute.jit
def team_n_pes(team):
    return bindings.team_n_pes(team)


@cute.jit
def n_pes():
    return bindings.n_pes()


@cute.jit
def barrier_all():
    # Use TEAM_WORLD for all PEs; match collective.py's pattern
    team = Teams.TEAM_WORLD
    return bindings.team_barrier(team)


@cute.jit
def sync_all():
    team = Teams.TEAM_WORLD
    return bindings.team_sync(team)


@cute.jit
def signal_op(signal_var, signal_val, signal_op, pe):
    signal_var_ptr = _resolve_ptr(signal_var)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    return bindings.signal_op(signal_var_ptr, signal_val, signal_op, pe)


@cute.jit
def signal_wait(signal_var, signal_op, signal_val):
    signal_var_ptr = _resolve_ptr(signal_var)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    return bindings.signal_wait_until(signal_var_ptr, signal_op, signal_val)
