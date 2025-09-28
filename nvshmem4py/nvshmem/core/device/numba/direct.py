# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

import nvshmem.bindings.device.numba as bindings

import cffi
from numba.core.extending import overload
from numba.types import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, float16, Array
# from numba.cuda.types import bfloat16

__all__ = ["my_pe", "team_my_pe", "team_n_pes", "n_pes", "barrier_all", "sync_all", "signal_op", "signal_wait"]

# TODO: create a global ffi object for other high level bindings to use
ffi = cffi.FFI()

def my_pe(): pass
@overload(my_pe)
def my_pe_ol():
    def impl():
        return bindings.my_pe()
    return impl

def team_my_pe(): pass
@overload(team_my_pe)
def team_my_pe_ol(team):
    def impl(team):
        return bindings.team_my_pe(team) 
    return impl

def team_n_pes(): pass
@overload(team_n_pes)
def team_n_pes_ol(team):
    def impl(team):
        return bindings.team_n_pes(team)
    return impl

def n_pes(): pass
@overload(n_pes)
def n_pes_ol():
    def impl():
        return bindings.n_pes()
    return impl

def barrier_all(): pass
@overload(barrier_all)
def barrier_all_ol():
    def impl():
        return bindings.barrier_all()
    return impl

def sync_all(): pass
@overload(sync_all)
def sync_all_ol():
    def impl():
        return bindings.sync_all()
    return impl

def signal_op(): pass
@overload(signal_op)
def signal_op_ol(signal_var, signal_val, signal_op, pe):
    def impl(signal_var, signal_val, signal_op, pe):
        signal_varptr = ffi.from_buffer(signal_var)
        bindings.signal_op(signal_varptr, signal_val, signal_op, pe)
    return impl

def signal_wait(): pass
@overload(signal_wait)
def signal_wait_ol(signal_var, signal_op, signal_val):
    def impl(signal_var, signal_op, signal_val):
        signal_varptr = ffi.from_buffer(signal_var)
        return bindings.signal_wait_until(signal_varptr, signal_op, signal_val)
    return impl 
