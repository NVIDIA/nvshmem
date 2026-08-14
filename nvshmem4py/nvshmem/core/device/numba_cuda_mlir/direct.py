# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cffi

import nvshmem.bindings.device.numba_cuda_mlir as bindings
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import int32, uint64

__all__ = [
    "my_pe",
    "team_my_pe",
    "team_n_pes",
    "n_pes",
    "barrier_all",
    "sync_all",
    "signal_op",
    "signal_wait",
]

ffi = cffi.FFI()


def my_pe():
    pass


@overload(my_pe, inline="always", typing_registry=typing_registry)
def my_pe_ol():

    def impl():
        return bindings.my_pe()

    return impl


def team_my_pe(team):
    pass


@overload(team_my_pe, inline="always", typing_registry=typing_registry)
def team_my_pe_ol(team):

    def impl(team):
        return bindings.team_my_pe(int32(team))

    return impl


def team_n_pes(team):
    pass


@overload(team_n_pes, inline="always", typing_registry=typing_registry)
def team_n_pes_ol(team):

    def impl(team):
        return bindings.team_n_pes(int32(team))

    return impl


def n_pes():
    pass


@overload(n_pes, inline="always", typing_registry=typing_registry)
def n_pes_ol():

    def impl():
        return bindings.n_pes()

    return impl


def barrier_all():
    pass


@overload(barrier_all, inline="always", typing_registry=typing_registry)
def barrier_all_ol():

    def impl():
        return bindings.barrier_all()

    return impl


def sync_all():
    pass


@overload(sync_all, inline="always", typing_registry=typing_registry)
def sync_all_ol():

    def impl():
        return bindings.sync_all()

    return impl


def signal_op(signal_var, signal_val, signal_op, pe):
    pass


@overload(signal_op, inline="always", typing_registry=typing_registry)
def signal_op_ol(signal_var, signal_val, signal_op, pe):

    def impl(signal_var, signal_val, signal_op, pe):
        signal_varptr = ffi.from_buffer(signal_var)
        bindings.signal_op(signal_varptr, uint64(signal_val), int32(signal_op.value), int32(pe))

    return impl


def signal_wait(signal_var, signal_op, signal_val):
    pass


@overload(signal_wait, inline="always", typing_registry=typing_registry)
def signal_wait_ol(signal_var, signal_op, signal_val):

    def impl(signal_var, signal_op, signal_val):
        signal_varptr = ffi.from_buffer(signal_var)
        return bindings.signal_wait_until(signal_varptr, int32(signal_op.value), uint64(signal_val))

    return impl
