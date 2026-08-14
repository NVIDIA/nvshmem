# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nvshmem.bindings.device.cute as bindings
from nvshmem.core import SmemAmount, Teams

__all__ = [
    "SmemAmount", "ask_smem", "give_smem", "release_smem", "my_pe", "team_my_pe", "team_n_pes", "n_pes", "barrier_all",
    "sync_all", "signal_op", "signal_wait"
]

from cutlass import cute
import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import cast as cute_cast
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@dsl_user_op
def _smem_ptr_as_generic(smem_ptr, *, loc=None, ip=None):
    """Widen a shared-memory pointer into a generic 16B-aligned ``Int8`` pointer.

    A CuTe shared pointer is a 32-bit shared-window offset, while the native
    ``void *`` parameter is a 64-bit generic address, so the pointer needs an
    address-space cast (``cvta.shared.u64``) before it crosses the FFI boundary.
    """
    shared = llvm.inttoptr(
        llvm.PointerType.get(int(AddressSpace.smem)),
        smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    generic = llvm.addrspacecast(llvm.PointerType.get(int(AddressSpace.generic)), shared, loc=loc, ip=ip)
    address = llvm.ptrtoint(T.i64(), generic, loc=loc, ip=ip)
    return cute.make_ptr(cutlass.Int8, address, AddressSpace.generic, assumed_align=16, loc=loc, ip=ip)


@cute.jit
def ask_smem(amount):
    """Return the TMA dynamic shared-memory requirement for ``amount``.

    Use the result as the dynamic shared-memory size when launching a kernel
    that calls :func:`give_smem`.
    """
    return bindings.ask_smem(cute_cast(amount, cutlass.Int32))


@cute.jit
def give_smem(smem: cute.Tensor):
    """Register a dynamic shared-memory tensor for TMA transfers in this CTA.

    ``smem`` must cover the full dynamic shared-memory allocation and have a
    16-byte-aligned base, for example a tensor constructed from
    ``cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)``. Its byte size is
    derived from its element type and layout, so callers do not pass a
    separate size. The allocation must be at least
    ``ask_smem(SmemAmount.SMEM_MINIMUM)`` bytes and all threads in every CTA
    must register an equally sized allocation. Pair every call with
    :func:`release_smem` before the kernel returns.
    """
    # The generated binding uses an Int8 pointer for C ``void*``.
    smem_ptr = _smem_ptr_as_generic(_resolve_ptr(smem))
    size = cute.size_in_bytes(smem.element_type, smem.layout)
    return bindings.give_smem(smem_ptr, cute_cast(size, cutlass.Uint64))


@cute.jit
def release_smem():
    """Release the calling CTA's TMA shared-memory registration.

    This is the no-argument counterpart to :func:`give_smem`: the native API
    records registration per CTA rather than per array. All threads in every
    CTA that registers a shared-memory tensor must release it before the
    kernel exits.
    """
    return bindings.release_smem()


@cute.jit
def my_pe():
    """
    Returns the PE number of the calling PE within the default (world) team.

    This is a thread-level query that returns the rank of the current PE in
    the global NVSHMEM PE space (``TEAM_WORLD``), ranging from ``0`` to
    ``n_pes() - 1``.

    Returns:
        The integer PE rank of the calling PE (``int32``).

    Note:
        May be called from any thread; all threads in a kernel return the same
        PE number.  Use ``team_my_pe`` to query the rank within a sub-team.
    """
    return bindings.my_pe()


@cute.jit
def team_my_pe(team):
    """
    Returns the PE number of the calling PE within ``team``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle.  Use
          ``nvshmem.core.Teams.TEAM_WORLD`` for the global team.

    Returns:
        The integer rank of the calling PE within ``team`` (``int32``),
        ranging from ``0`` to ``team_n_pes(team) - 1``.

    Note:
        The returned rank is relative to ``team`` and may differ from the
        global PE number returned by ``my_pe``.
    """
    return bindings.team_my_pe(team)


@cute.jit
def team_n_pes(team):
    """
    Returns the total number of PEs in ``team``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle.  Use
          ``nvshmem.core.Teams.TEAM_WORLD`` for the global team.

    Returns:
        The number of PEs participating in ``team`` (``int32``).

    Note:
        Equivalent to querying the size of the communicator associated with
        ``team``.  Use ``n_pes`` to obtain the global PE count.
    """
    return bindings.team_n_pes(team)


@cute.jit
def n_pes():
    """
    Returns the total number of PEs in the NVSHMEM runtime (``TEAM_WORLD``).

    This is a thread-level query equivalent to ``team_n_pes(TEAM_WORLD)``.

    Returns:
        The total number of PEs running in the NVSHMEM program (``int32``).

    Note:
        May be called from any thread; all threads in a kernel return the same
        value.  Use ``team_n_pes`` to query the size of a sub-team.
    """
    return bindings.n_pes()


@cute.jit
def barrier_all():
    """
    Executes a thread-level barrier across all PEs in the NVSHMEM runtime
    (equivalent to ``barrier(TEAM_WORLD)``).

    A barrier combines synchronization with a full memory fence, ensuring that
    all outstanding NVSHMEM memory operations (puts, gets, atomics) issued
    before the barrier are complete and visible before any PE proceeds past it.

    Note:
        All PEs must call ``barrier_all`` before any PE can proceed past it.
        For CTA-wide or warp-wide variants use ``barrier_all_block`` /
        ``barrier_all_warp`` from the bindings in collective.py.
        ``barrier_all`` provides stronger ordering guarantees than ``sync_all``.
    """
    # Use TEAM_WORLD for all PEs; match collective.py's pattern
    team = Teams.TEAM_WORLD
    return bindings.team_barrier(team)


@cute.jit
def sync_all():
    """
    Executes a thread-level synchronization across all PEs in the NVSHMEM
    runtime (equivalent to ``sync(TEAM_WORLD)``).

    This is a lightweight synchronization point that guarantees all PEs have
    reached it before any PE proceeds.  It does not provide memory-ordering or
    memory-visibility guarantees; use ``barrier_all`` when a memory fence is
    also required.  All PEs must call this function before any PE can proceed
    past it.

    Note:
        All PEs must call ``sync_all`` before any PE can proceed past it.
        For CTA-wide or warp-wide variants use ``sync_all_block`` /
        ``sync_all_warp`` from the bindings in collective.py.
        Use ``barrier_all`` when a full memory fence is also required.
    """
    team = Teams.TEAM_WORLD
    return bindings.team_sync(team)


@cute.jit
def signal_op(signal_var, signal_val, signal_op, pe):
    """
    Performs a signal operation on the signal variable ``signal_var`` on PE ``pe``.

    Atomically updates the remote signal variable according to ``signal_op``:
    either sets it to ``signal_val`` (``NVSHMEM_SIGNAL_SET``) or adds
    ``signal_val`` to its current value (``NVSHMEM_SIGNAL_ADD``).  This
    allows a local PE to notify a remote PE of an event without transferring
    bulk data.

    Args:
        - ``signal_var``: CuTe tensor view pointing to a symmetric signal
          variable (dtype ``uint64``) on PE ``pe``.  Must be a 1-element
          NVSHMEM-allocated tensor.
        - ``signal_val`` (``int``): Value used to update the signal variable.
          Cast to ``uint64`` before the operation.
        - ``signal_op``: Signal operation type.  Supported values are
          ``nvshmem.core.SignalOp.SIGNAL_SET`` (overwrite with ``signal_val``)
          and ``nvshmem.core.SignalOp.SIGNAL_ADD`` (atomically add
          ``signal_val``).
        - ``pe`` (``int``): Target PE that owns ``signal_var``.

    Note:
        This is a thread-level, non-blocking operation.  The update may not
        be immediately visible to the remote PE.  Pair with ``signal_wait``
        on the remote side to coordinate.
    """
    signal_var_ptr = _resolve_ptr(signal_var)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    return bindings.signal_op(signal_var_ptr, signal_val, signal_op, pe)


@cute.jit
def signal_wait(signal_var, signal_op, signal_val):
    """
    Waits until the local signal variable ``signal_var`` satisfies the
    condition ``signal_var <signal_op> signal_val``.

    Spins (busy-waits) until the condition is true, then returns.  This is
    typically used on the receiving PE to detect that a remote PE has
    completed a data transfer or a signaling operation.

    Args:
        - ``signal_var``: CuTe tensor view pointing to a local signal
          variable (dtype ``uint64``).  Must be a 1-element
          NVSHMEM-allocated tensor on the calling PE.
        - ``signal_op``: Comparison operator.  Supported values are
          ``nvshmem.core.CMP_EQ``, ``nvshmem.core.CMP_NE``, ``nvshmem.core.CMP_GT``,
          ``nvshmem.core.CMP_GE``, ``nvshmem.core.CMP_LT``, and ``nvshmem.core.CMP_LE``.
        - ``signal_val`` (``int``): Threshold value for the comparison.
          Cast to ``uint64`` before the operation.

    Note:
        This is a thread-level blocking operation.  The calling thread spins
        until the condition is satisfied.  Pair with ``signal_op`` on the
        sending PE to coordinate.
    """
    signal_var_ptr = _resolve_ptr(signal_var)
    signal_val = cute_cast(signal_val, cutlass.Uint64)
    return bindings.signal_wait_until(signal_var_ptr, signal_op, signal_val)
