# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
The following are nvshmem.core APIs that can be used as-is
from their bindings
"""

import nvshmem.bindings as bindings
from nvshmem.bindings.ask_smem import ask_smem as _ask_smem
from cuda.pathfinder import load_nvidia_dynamic_lib
import ctypes

__all__ = [
    "ComparisonType", "SignalOp", "InitStatus", "ThreadSupport", "SmemAmount", "ask_smem", "query_thread", "my_pe",
    "team_my_pe", "team_n_pes", "n_pes", "init_status"
]
"""
IntEnum which matches 1:1 with ``nvshmemx_cmp_type_t``
"""
ComparisonType = bindings.Cmp_type
"""
IntEnum which matches 1:1 with the ``nvshmem_signal_op_t``
"""
SignalOp = bindings.Signal_op
"""
IntEnum which matches 1:1 with ``nvshmem_init_status_t``
"""
InitStatus = bindings.Init_status
"""
IntEnum which matches 1:1 with ``nvshmemx_thread_support_t``.
"""
ThreadSupport = bindings.Thread_support

# IntEnum matching ``nvshmemx_smem_amount_t``.
SmemAmount = bindings.Smem_amount


def ask_smem(amount: SmemAmount = SmemAmount.SMEM_RECOMMENDED) -> int:
    """Return the dynamic shared-memory size requested by NVSHMEM TMA.

    The value can be supplied as the dynamic shared-memory launch size for a
    kernel that calls the device-side :func:`nvshmemx_give_smem` wrapper.
    Invalid values follow the native helper and return the recommended size.
    """
    return _ask_smem(amount)


def query_thread() -> ThreadSupport:
    """Return the NVSHMEM host-library thread-support level.

    This mirrors ``nvshmem_query_thread`` and can be called before NVSHMEM
    initialization. It reports the library capability, not a guarantee about
    the thread safety of Python code that calls it.
    """
    load_nvidia_dynamic_lib("nvshmem_host")
    provided = ctypes.c_int()
    bindings.query_thread(ctypes.addressof(provided))
    return ThreadSupport(provided.value)


def my_pe() -> int:
    """Get the current Processing Element (PE) ID of this process.

    Returns:
        int: The PE ID of the calling process within ``TEAM_WORLD``.
    """
    return bindings.my_pe()


def n_pes() -> int:
    """Get the total number of Processing Elements (PEs) in ``TEAM_WORLD``.

    Returns:
        int: The total number of PEs in the default global team (``TEAM_WORLD``).
    """
    return bindings.n_pes()


def team_my_pe(team) -> int:
    """Get the PE ID of this process within a specified team.

    Args:
        team: The team handle (e.g., ``nvshmem.core.Teams.TEAM_NODE``).

    Returns:
        int: The PE ID of the calling process within the specified team.
    """
    return bindings.team_my_pe(team)


def team_n_pes(team) -> int:
    """Get the number of Processing Elements (PEs) in a specified team.

    Args:
        team: The team handle (e.g., ``nvshmem.core.Teams.TEAM_NODE``).

    Returns:
        int: The total number of PEs in the specified team.
    """
    return bindings.team_n_pes(team)


def init_status() -> InitStatus:
    """Get the current initialization status

    Returns:
        InitStatus: An enum representing the status of initialization.
    """
    return bindings.init_status()
