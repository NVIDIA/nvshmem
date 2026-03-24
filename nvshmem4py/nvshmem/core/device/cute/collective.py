# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

from nvshmem.bindings.device.cute import *
from nvshmem.bindings.device.cute import (
    barrier as _nvshmem_barrier,
    barrier_block as _nvshmem_barrier_block,
    barrier_warp as _nvshmem_barrier_warp,
)

import nvshmem.core
import cutlass
from cutlass import cute
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.base_dsl.typing import cast as cute_cast

__all__ = [
    "sync_block",
    "sync_warp",
    "sync",
    "sync_all",
    "sync_all_block",
    "sync_all_warp",
    "barrier",
    "barrier_block",
    "barrier_warp",
    "barrier_all",
    "barrier_all_block",
    "barrier_all_warp",
    "reduce",
    "reduce_block",
    "reduce_warp",
    "reducescatter",
    "reducescatter_block",
    "reducescatter_warp",
    "fcollect",
    "fcollect_block",
    "fcollect_warp",
    "broadcast",
    "broadcast_block",
    "broadcast_warp",
    "alltoall",
    "alltoall_block",
    "alltoall_warp",
]


@cute.jit
def _resolve_ptr(arg):
    return arg.iterator


@cute.jit
def _size_of(obj):
    return cute.size(obj)


@cute.jit
def _resolve_dtype(dst):
    return dst.element_type


@cute.jit
def _resolve_team(team):
    return cute_cast(team, cutlass.Int32)


@cute.jit
def _resolve_root(root):
    return cute_cast(root, cutlass.Int32)


@cute.jit
def _resolve_nelems_dst(dst):
    return cute_cast(_size_of(dst), cutlass.Uint64)


@cute.jit
def _resolve_nelems_src(src):
    return cute_cast(_size_of(src), cutlass.Uint64)


@cute.jit
def _resolve_nelems_min(dst, src):
    dst_nelems = _size_of(dst)
    src_nelems = _size_of(src)
    return cute_cast(dst_nelems if dst_nelems < src_nelems else src_nelems, cutlass.Uint64)


@cute.jit
def _resolve_nelems_alltoall(src, team):
    src_nelems = cute_cast(_size_of(src), cutlass.Uint64)
    team_size = cute_cast(team_n_pes(team), cutlass.Uint64)
    return cute_cast(src_nelems // team_size, cutlass.Uint64)


# sync variations


@cute.jit
def sync_block(team):
    """
    Executes a CTA-level synchronization across all PEs in ``team``. All threads in the CTA must call this function.

    This is a lightweight synchronization point that guarantees all PEs in the team have
    reached it before any PE proceeds.  It does not provide memory-ordering or
    memory-visibility guarantees; use ``barrier`` when a memory fence is also required.
    All PEs in the team must call this function before any PE can proceed past it.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to synchronize.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to synchronize all PEs.

    Note:
        All PEs in ``team`` must call ``sync_block`` with the same ``team`` argument.
        Use ``sync_all_block`` to synchronize across all PEs without specifying a team.
    """
    team = _resolve_team(team)
    return team_sync_block(team)


@cute.jit
def sync_warp(team):
    """
    Executes a warp-level synchronization across all PEs in ``team``. All threads in the warp must call this function.

    This is a lightweight synchronization point that guarantees all PEs in the team have
    reached it before any PE proceeds.  It does not provide memory-ordering or
    memory-visibility guarantees; use ``barrier`` when a memory fence is also required.
    All PEs in the team must call this function before any PE can proceed past it.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to synchronize.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to synchronize all PEs.

    Note:
        All PEs in ``team`` must call ``sync_warp`` with the same ``team`` argument.
        Use ``sync_all_warp`` to synchronize across all PEs without specifying a team.
    """
    team = _resolve_team(team)
    return team_sync_warp(team)


@cute.jit
def sync(team):
    """
    Executes a thread-level synchronization across all PEs in ``team``.

    This is a lightweight synchronization point that guarantees all PEs in the team have
    reached it before any PE proceeds.  It does not provide memory-ordering or
    memory-visibility guarantees; use ``barrier`` when a memory fence is also required.
    All PEs in the team must call this function before any PE can proceed past it.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to synchronize.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to synchronize all PEs.

    Note:
        All PEs in ``team`` must call ``sync`` with the same ``team`` argument.
        Use ``sync_all`` to synchronize across all PEs without specifying a team.
    """
    team = _resolve_team(team)
    return team_sync(team)


# sync_all variations


@cute.jit
def sync_all_block():
    """
    Executes a CTA-level synchronization across all PEs in the NVSHMEM runtime
    (equivalent to ``sync_block(TEAM_WORLD)``). All threads in the CTA must call this function.

    This is a convenience wrapper around ``sync_block`` that automatically uses
    ``TEAM_WORLD`` as the team, covering all PEs participating in the NVSHMEM job.

    Note:
        All PEs must call ``sync_all_block`` before any PE can proceed past it.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync_block(team)


@cute.jit
def sync_all_warp():
    """
    Executes a warp-level synchronization across all PEs in the NVSHMEM runtime
    (equivalent to ``sync_warp(TEAM_WORLD)``). All threads in the warp must call this function.

    This is a convenience wrapper around ``sync_warp`` that automatically uses
    ``TEAM_WORLD`` as the team, covering all PEs participating in the NVSHMEM job.

    Note:
        All PEs must call ``sync_all_warp`` before any PE can proceed past it.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync_warp(team)


@cute.jit
def sync_all():
    """
    Executes a thread-level synchronization across all PEs in the NVSHMEM runtime
    (equivalent to ``sync(TEAM_WORLD)``).

    This is a convenience wrapper around ``sync`` that automatically uses
    ``TEAM_WORLD`` as the team, covering all PEs participating in the NVSHMEM job.

    Note:
        All PEs must call ``sync_all`` before any PE can proceed past it.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return team_sync(team)


# barrier variations


@cute.jit
def barrier_block(team):
    """
    Executes a CTA-level barrier across all PEs in ``team``. All threads in the CTA must call this function.

    A barrier combines synchronization with a full memory fence, ensuring that all
    outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier
    are complete and visible before any PE in the team proceeds past the barrier.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to barrier on.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to barrier across all PEs.

    Note:
        All PEs in ``team`` must call ``barrier_block`` with the same ``team`` argument.
        Use ``barrier_all_block`` to barrier across all PEs without specifying a team.
        ``barrier`` provides stronger ordering guarantees than ``sync``.
    """
    team = _resolve_team(team)
    return _nvshmem_barrier_block(team)


@cute.jit
def barrier_warp(team):
    """
    Executes a warp-level barrier across all PEs in ``team``. All threads in the warp must call this function.

    A barrier combines synchronization with a full memory fence, ensuring that all
    outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier
    are complete and visible before any PE in the team proceeds past the barrier.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to barrier on.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to barrier across all PEs.

    Note:
        All PEs in ``team`` must call ``barrier_warp`` with the same ``team`` argument.
        Use ``barrier_all_warp`` to barrier across all PEs without specifying a team.
        ``barrier`` provides stronger ordering guarantees than ``sync``.
    """
    team = _resolve_team(team)
    return _nvshmem_barrier_warp(team)


@cute.jit
def barrier(team):
    """
    Executes a thread-level barrier across all PEs in ``team``.

    A barrier combines synchronization with a full memory fence, ensuring that all
    outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier
    are complete and visible before any PE in the team proceeds past the barrier.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the set of PEs to barrier on.
          Use ``nvshmem.core.Teams.TEAM_WORLD`` to barrier across all PEs.

    Note:
        All PEs in ``team`` must call ``barrier`` with the same ``team`` argument.
        Use ``barrier_all`` to barrier across all PEs without specifying a team.
        ``barrier`` provides stronger ordering guarantees than ``sync``.
    """
    team = _resolve_team(team)
    return _nvshmem_barrier(team)


# barrier_all variations


@cute.jit
def barrier_all_block():
    """
    Executes a CTA-level barrier across all PEs in the NVSHMEM runtime
    (equivalent to ``barrier_block(TEAM_WORLD)``). All threads in the CTA must call this function.

    This is a convenience wrapper around ``barrier_block`` that automatically uses
    ``TEAM_WORLD`` as the team. It combines synchronization with a full memory fence,
    ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

    Note:
        All PEs must call ``barrier_all_block`` before any PE can proceed past it.
        ``barrier_all`` provides stronger ordering guarantees than ``sync_all``.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier_block(team)


@cute.jit
def barrier_all_warp():
    """
    Executes a warp-level barrier across all PEs in the NVSHMEM runtime
    (equivalent to ``barrier_warp(TEAM_WORLD)``). All threads in the warp must call this function.

    This is a convenience wrapper around ``barrier_warp`` that automatically uses
    ``TEAM_WORLD`` as the team. It combines synchronization with a full memory fence,
    ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

    Note:
        All PEs must call ``barrier_all_warp`` before any PE can proceed past it.
        ``barrier_all`` provides stronger ordering guarantees than ``sync_all``.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier_warp(team)


@cute.jit
def barrier_all():
    """
    Executes a thread-level barrier across all PEs in the NVSHMEM runtime
    (equivalent to ``barrier(TEAM_WORLD)``).

    This is a convenience wrapper around ``barrier`` that automatically uses
    ``TEAM_WORLD`` as the team. It combines synchronization with a full memory fence,
    ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

    Note:
        All PEs must call ``barrier_all`` before any PE can proceed past it.
        ``barrier_all`` provides stronger ordering guarantees than ``sync_all``.
    """
    team = _resolve_team(nvshmem.core.Teams.TEAM_WORLD)
    return _nvshmem_barrier(team)


# reduce variations


@cute.jit
def reduce_block(team, dst, src, op):
    """
    Performs a CTA-scoped all-reduce from ``src`` to ``dst`` across all PEs in ``team``. All threads in the CTA must call this function with the same arguments.

    Each PE contributes ``size(dst)`` elements from ``src``, and the result of applying
    the reduction operator ``op`` element-wise across all PEs is written to ``dst`` on
    every PE in the team (all-reduce semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced
          is ``size(dst)``.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least ``size(dst)`` elements.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reduce_block`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce_block(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce_block(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")


@cute.jit
def reduce_warp(team, dst, src, op):
    """
    Performs a warp-scoped all-reduce from ``src`` to ``dst`` across all PEs in ``team``. All threads in the warp must call this function with the same arguments.

    Each PE contributes ``size(dst)`` elements from ``src``, and the result of applying
    the reduction operator ``op`` element-wise across all PEs is written to ``dst`` on
    every PE in the team (all-reduce semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced
          is ``size(dst)``.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least ``size(dst)`` elements.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reduce_warp`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce_warp(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce_warp(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")


@cute.jit
def reduce(team, dst, src, op):
    """
    Performs a thread-scoped all-reduce from ``src`` to ``dst`` across all PEs in ``team``.

    Each PE contributes ``size(dst)`` elements from ``src``, and the result of applying
    the reduction operator ``op`` element-wise across all PEs is written to ``dst`` on
    every PE in the team (all-reduce semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced
          is ``size(dst)``.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least ``size(dst)`` elements.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reduce`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reduce(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reduce(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reduce(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reduce(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reduce op/dtype combination: op={op}, dtype={dtype}")


# reducescatter variations


@cute.jit
def reducescatter_block(team, dst, src, op):
    """
    Performs a CTA-scoped reduce-scatter from ``src`` to ``dst`` across all PEs in ``team``. All threads in the CTA must call this function with the same arguments.

    In a reduce-scatter, each PE contributes elements from ``src``, and the result of
    applying the reduction operator element-wise across all PEs is divided into equal
    portions, each portion written to ``dst`` on a different PE (scatter semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements
          per PE is ``size(dst)``. The ``src`` array should have at least
          ``size(dst) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reducescatter_block`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter_block(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter_block(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")


@cute.jit
def reducescatter_warp(team, dst, src, op):
    """
    Performs a warp-scoped reduce-scatter from ``src`` to ``dst`` across all PEs in ``team``. All threads in the warp must call this function with the same arguments.

    In a reduce-scatter, each PE contributes elements from ``src``, and the result of
    applying the reduction operator element-wise across all PEs is divided into equal
    portions, each portion written to ``dst`` on a different PE (scatter semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements
          per PE is ``size(dst)``. The ``src`` array should have at least
          ``size(dst) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reducescatter_warp`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter_warp(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter_warp(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")


@cute.jit
def reducescatter(team, dst, src, op):
    """
    Performs a thread-scoped reduce-scatter from ``src`` to ``dst`` across all PEs in ``team``.

    In a reduce-scatter, each PE contributes elements from ``src``, and the result of
    applying the reduction operator element-wise across all PEs is divided into equal
    portions, each portion written to ``dst`` on a different PE (scatter semantics).

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements
          per PE is ``size(dst)``. The ``src`` array should have at least
          ``size(dst) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array.
          Must be a symmetric (NVSHMEM-allocated) tensor.
        - ``op`` (``str``): Reduction operator string. Supported operators for numeric types:
          ``"sum"``, ``"prod"``, ``"min"``, ``"max"``. Additional bitwise operators for
          integral types: ``"and"``, ``"or"``, ``"xor"``.

    Note:
        All PEs in ``team`` must call ``reducescatter`` before any PE can proceed past it.
        The element count is taken from ``dst``. Passing an unsupported op/dtype combination
        raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_dst(dst)
    dtype = _resolve_dtype(dst)

    if const_expr(op == "min"):

        if const_expr(dtype == cutlass.Int8):
            return int8_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_min_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_min_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "max"):

        if const_expr(dtype == cutlass.Int8):
            return int8_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_max_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_max_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "sum"):

        if const_expr(dtype == cutlass.Int8):
            return int8_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_sum_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_sum_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "prod"):

        if const_expr(dtype == cutlass.Int8):
            return int8_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float32):
            return float_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float64):
            return double_prod_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Float16):
            return half_prod_reducescatter(team, dst_ptr, src_ptr, nelem)



    elif const_expr(op == "and"):

        if const_expr(dtype == cutlass.Int8):
            return int8_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_and_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_and_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "or"):

        if const_expr(dtype == cutlass.Int8):
            return int8_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_or_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_or_reducescatter(team, dst_ptr, src_ptr, nelem)


    elif const_expr(op == "xor"):

        if const_expr(dtype == cutlass.Int8):
            return int8_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int16):
            return int16_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int32):
            return int32_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Int64):
            return int64_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint8):
            return uint8_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint16):
            return uint16_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint32):
            return uint32_xor_reducescatter(team, dst_ptr, src_ptr, nelem)

        elif const_expr(dtype == cutlass.Uint64):
            return uint64_xor_reducescatter(team, dst_ptr, src_ptr, nelem)


    raise RuntimeError(f"Unsupported CuTe reducescatter op/dtype combination: op={op}, dtype={dtype}")


# fcollect variations


@cute.jit
def fcollect_block(team, dst, src):
    """
    Performs a CTA-scoped fcollect (all-gather) from ``src`` to ``dst`` across all PEs in ``team``. All threads in the CTA must call this function with the same arguments.

    Each PE contributes ``size(src)`` elements, and the concatenated result from all PEs
    is written to ``dst`` on every PE. The ``dst`` array must be large enough to hold
    contributions from all PEs: ``size(dst) >= size(src) * team_n_pes(team)``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least
          ``size(src) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements
          contributed per PE is ``size(src)``.

    Note:
        All PEs in ``team`` must call ``fcollect_block`` before any PE can proceed past it.
        The element count per PE is taken from ``src``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect_block(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")


@cute.jit
def fcollect_warp(team, dst, src):
    """
    Performs a warp-scoped fcollect (all-gather) from ``src`` to ``dst`` across all PEs in ``team``. All threads in the warp must call this function with the same arguments.

    Each PE contributes ``size(src)`` elements, and the concatenated result from all PEs
    is written to ``dst`` on every PE. The ``dst`` array must be large enough to hold
    contributions from all PEs: ``size(dst) >= size(src) * team_n_pes(team)``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least
          ``size(src) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements
          contributed per PE is ``size(src)``.

    Note:
        All PEs in ``team`` must call ``fcollect_warp`` before any PE can proceed past it.
        The element count per PE is taken from ``src``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect_warp(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")


@cute.jit
def fcollect(team, dst, src):
    """
    Performs a thread-scoped fcollect (all-gather) from ``src`` to ``dst`` across all PEs in ``team``.

    Each PE contributes ``size(src)`` elements, and the concatenated result from all PEs
    is written to ``dst`` on every PE. The ``dst`` array must be large enough to hold
    contributions from all PEs: ``size(dst) >= size(src) * team_n_pes(team)``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor with at least
          ``size(src) * team_n_pes(team)`` elements.
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements
          contributed per PE is ``size(src)``.

    Note:
        All PEs in ``team`` must call ``fcollect`` before any PE can proceed past it.
        The element count per PE is taken from ``src``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_src(src)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_fcollect(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_fcollect(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for fcollect: {dtype}")


# broadcast variations


@cute.jit
def broadcast_block(team, dst, src, root=0):
    """
    Performs a CTA-scoped broadcast from ``src`` on ``root`` PE to ``dst`` on all PEs in ``team``. All threads in the CTA must call this function with the same arguments.

    The root PE broadcasts the contents of its ``src`` array to the ``dst`` array on
    every PE in the team (including the root itself). Non-root PEs ignore their own ``src``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
        - ``src``: CuTe tensor view pointing to the symmetric source array on the root PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE's ``src`` is used.
        - ``root`` (``int``, optional): PE rank within ``team`` that serves as the broadcast source.
          Defaults to ``0`` (the first PE in the team).

    Note:
        All PEs in ``team`` must call ``broadcast_block`` before any PE can proceed past it.
        The number of elements broadcast is ``min(size(dst), size(src))``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast_block(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")


@cute.jit
def broadcast_warp(team, dst, src, root=0):
    """
    Performs a warp-scoped broadcast from ``src`` on ``root`` PE to ``dst`` on all PEs in ``team``. All threads in the warp must call this function with the same arguments.

    The root PE broadcasts the contents of its ``src`` array to the ``dst`` array on
    every PE in the team (including the root itself). Non-root PEs ignore their own ``src``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
        - ``src``: CuTe tensor view pointing to the symmetric source array on the root PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE's ``src`` is used.
        - ``root`` (``int``, optional): PE rank within ``team`` that serves as the broadcast source.
          Defaults to ``0`` (the first PE in the team).

    Note:
        All PEs in ``team`` must call ``broadcast_warp`` before any PE can proceed past it.
        The number of elements broadcast is ``min(size(dst), size(src))``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast_warp(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")


@cute.jit
def broadcast(team, dst, src, root=0):
    """
    Performs a thread-scoped broadcast from ``src`` on ``root`` PE to ``dst`` on all PEs in ``team``.

    The root PE broadcasts the contents of its ``src`` array to the ``dst`` array on
    every PE in the team (including the root itself). Non-root PEs ignore their own ``src``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array.
          Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
        - ``src``: CuTe tensor view pointing to the symmetric source array on the root PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE's ``src`` is used.
        - ``root`` (``int``, optional): PE rank within ``team`` that serves as the broadcast source.
          Defaults to ``0`` (the first PE in the team).

    Note:
        All PEs in ``team`` must call ``broadcast`` before any PE can proceed past it.
        The number of elements broadcast is ``min(size(dst), size(src))``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_min(dst, src)
    root = _resolve_root(root)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int16):
        return int16_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int32):
        return int32_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Int64):
        return int64_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float32):
        return float_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float64):
        return double_broadcast(team, dst_ptr, src_ptr, nelem, root)

    elif const_expr(dtype == cutlass.Float16):
        return half_broadcast(team, dst_ptr, src_ptr, nelem, root)

    raise RuntimeError(f"Unsupported CuTe dtype for broadcast: {dtype}")


# alltoall variations


@cute.jit
def alltoall_block(team, dst, src):
    """
    Performs a CTA-scoped all-to-all exchange from ``src`` to ``dst`` across all PEs in ``team``. All threads in the CTA must call this function with the same arguments.

    In an all-to-all operation, each PE sends a distinct portion of its ``src`` array to
    every other PE, and receives a portion from every PE into its ``dst`` array.
    The ``src`` array is logically divided into ``team_n_pes(team)`` equal segments;
    segment ``i`` is sent to PE ``i``. Each PE receives one segment from every PE into
    the corresponding portion of ``dst``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least
          ``size(src)`` elements (same total size as ``src``).
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is
          ``size(src) // team_n_pes(team)`` elements.

    Note:
        All PEs in ``team`` must call ``alltoall_block`` before any PE can proceed past it.
        ``size(src)`` must be evenly divisible by ``team_n_pes(team)``.
        The per-PE element count passed to NVSHMEM is ``size(src) // team_n_pes(team)``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall_block(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall_block(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")


@cute.jit
def alltoall_warp(team, dst, src):
    """
    Performs a warp-scoped all-to-all exchange from ``src`` to ``dst`` across all PEs in ``team``. All threads in the warp must call this function with the same arguments.

    In an all-to-all operation, each PE sends a distinct portion of its ``src`` array to
    every other PE, and receives a portion from every PE into its ``dst`` array.
    The ``src`` array is logically divided into ``team_n_pes(team)`` equal segments;
    segment ``i`` is sent to PE ``i``. Each PE receives one segment from every PE into
    the corresponding portion of ``dst``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least
          ``size(src)`` elements (same total size as ``src``).
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is
          ``size(src) // team_n_pes(team)`` elements.

    Note:
        All PEs in ``team`` must call ``alltoall_warp`` before any PE can proceed past it.
        ``size(src)`` must be evenly divisible by ``team_n_pes(team)``.
        The per-PE element count passed to NVSHMEM is ``size(src) // team_n_pes(team)``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall_warp(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")


@cute.jit
def alltoall(team, dst, src):
    """
    Performs a thread-scoped all-to-all exchange from ``src`` to ``dst`` across all PEs in ``team``.

    In an all-to-all operation, each PE sends a distinct portion of its ``src`` array to
    every other PE, and receives a portion from every PE into its ``dst`` array.
    The ``src`` array is logically divided into ``team_n_pes(team)`` equal segments;
    segment ``i`` is sent to PE ``i``. Each PE receives one segment from every PE into
    the corresponding portion of ``dst``.

    Args:
        - ``team`` (``int``): NVSHMEM team handle identifying the participating PEs.
        - ``dst``: CuTe tensor view pointing to the symmetric destination array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least
          ``size(src)`` elements (same total size as ``src``).
        - ``src``: CuTe tensor view pointing to the symmetric source array on this PE.
          Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is
          ``size(src) // team_n_pes(team)`` elements.

    Note:
        All PEs in ``team`` must call ``alltoall`` before any PE can proceed past it.
        ``size(src)`` must be evenly divisible by ``team_n_pes(team)``.
        The per-PE element count passed to NVSHMEM is ``size(src) // team_n_pes(team)``.
        Passing an unsupported dtype raises ``RuntimeError`` at JIT compile time.
    """
    team = _resolve_team(team)
    dst_ptr = _resolve_ptr(dst)
    src_ptr = _resolve_ptr(src)
    nelem = _resolve_nelems_alltoall(src, team)
    dtype = _resolve_dtype(dst)

    if const_expr(dtype == cutlass.Int8):
        return int8_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int16):
        return int16_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int32):
        return int32_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Int64):
        return int64_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint8):
        return uint8_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint16):
        return uint16_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint32):
        return uint32_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Uint64):
        return uint64_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float32):
        return float_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float64):
        return double_alltoall(team, dst_ptr, src_ptr, nelem)

    elif const_expr(dtype == cutlass.Float16):
        return half_alltoall(team, dst_ptr, src_ptr, nelem)

    raise RuntimeError(f"Unsupported CuTe dtype for alltoall: {dtype}")
