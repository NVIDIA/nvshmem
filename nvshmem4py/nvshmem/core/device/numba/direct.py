# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nvshmem.bindings.device.numba as bindings
import nvshmem.bindings as host_bindings
from nvshmem.core import Teams

import cffi
from numba.core import cgutils, types
from numba.core.extending import overload
from numba.types import int8, int16, int64, uint8, uint16, uint32, uint64, float32, float64, float16, Array
from numba import cuda
from numba.np.numpy_support import carray
from numba.np import arrayobj
from numba.cuda.extending import intrinsic

__all__ = [
    "SmemAmount", "ask_smem", "give_smem", "release_smem", "my_pe", "team_my_pe", "team_n_pes", "n_pes", "barrier_all",
    "sync_all", "signal_op", "signal_wait"
]

# TODO: create a global ffi object for other high level bindings to use
ffi = cffi.FFI()

# IntEnum matching ``nvshmemx_smem_amount_t``.
SmemAmount = host_bindings.Smem_amount


@intrinsic
def _as_numbast_smem_amount(typingctx, amount):
    if amount != int64:
        return None

    result = types.IntEnumMember(bindings.nvshmemx_smem_amount_t, int64)
    signature = result(amount)

    def codegen(context, builder, signature, args):
        return args[0]

    return signature, codegen


@intrinsic
def _array_data_voidptr(typingctx, arrty):
    """Return an array data pointer as Numbast's opaque ``void*`` type."""
    if not isinstance(arrty, Array):
        return None

    sig = types.CPointer(types.none)(arrty)

    def codegen(cgctx, builder, sig, args):
        ary = arrayobj.make_array(arrty)(cgctx, builder, args[0])
        return builder.bitcast(ary.data, cgutils.voidptr_t)

    return sig, codegen


def ask_smem(flag):
    """Return the TMA dynamic shared-memory requirement for ``flag`` in a CUDA kernel."""
    pass


@overload(ask_smem)
def ask_smem_ol(flag):

    def impl(flag):
        return bindings.ask_smem(_as_numbast_smem_amount(int64(flag)))

    return impl


def give_smem(smem):
    """Register a dynamic shared-memory array for TMA use in the calling CTA.

    ``smem`` must be a C-contiguous Numba CUDA shared-memory array covering
    the full allocation (not a view). Its byte size is inferred from
    ``smem.nbytes`` so callers do not pass a separate size. NVSHMEM requires a
    16-byte-aligned base, at least ``ask_smem(SmemAmount.SMEM_MINIMUM)`` bytes,
    and an equally sized allocation from all threads in every CTA; call
    :func:`release_smem` before the kernel returns.
    """
    pass


@overload(give_smem)
def give_smem_ol(smem):
    if not isinstance(smem, Array) or smem.layout != "C":
        return None

    def impl(smem):
        smem_ptr = _array_data_voidptr(smem)
        bindings.give_smem(smem_ptr, uint64(smem.nbytes))

    return impl


def release_smem():
    """Release TMA shared-memory registration for the calling CTA.

    This is the no-argument counterpart to :func:`give_smem`: registration is
    recorded per CTA rather than per array. All threads in every CTA that
    registers a shared-memory array must release it before the kernel exits.
    """
    pass


@overload(release_smem)
def release_smem_ol():

    def impl():
        bindings.release_smem()

    return impl


def my_pe():
    pass


@overload(my_pe)
def my_pe_ol():

    def impl():
        return bindings.my_pe()

    return impl


def team_my_pe():
    pass


@overload(team_my_pe)
def team_my_pe_ol(team):

    def impl(team):
        return bindings.team_my_pe(team)

    return impl


def team_n_pes():
    pass


@overload(team_n_pes)
def team_n_pes_ol(team):

    def impl(team):
        return bindings.team_n_pes(team)

    return impl


def n_pes():
    pass


@overload(n_pes)
def n_pes_ol():

    def impl():
        return bindings.n_pes()

    return impl


def barrier_all():
    pass


@overload(barrier_all)
def barrier_all_ol():

    def impl():
        return bindings.barrier_all()

    return impl


def sync_all():
    pass


@overload(sync_all)
def sync_all_ol():

    def impl():
        return bindings.sync_all()

    return impl


def signal_op():
    pass


@overload(signal_op)
def signal_op_ol(signal_var, signal_val, signal_op, pe):

    def impl(signal_var, signal_val, signal_op, pe):
        signal_varptr = ffi.from_buffer(signal_var)
        bindings.signal_op(signal_varptr, signal_val, signal_op, pe)

    return impl


def signal_wait():
    pass


@overload(signal_wait)
def signal_wait_ol(signal_var, signal_op, signal_val):

    def impl(signal_var, signal_op, signal_val):
        signal_varptr = ffi.from_buffer(signal_var)
        return bindings.signal_wait_until(signal_varptr, signal_op, signal_val)

    return impl
