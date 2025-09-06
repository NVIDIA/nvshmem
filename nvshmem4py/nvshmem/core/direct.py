# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

"""
The following are nvshmem.core APIs that can be used as-is
from their bindings
"""

import nvshmem.bindings as bindings

__all__ = ["Teams", "my_pe", "team_my_pe", "team_n_pes", "n_pes", "team_n_pes", "ComparisonType", "SignalOp", "InitStatus"]

"""
IntEnum which matches 1:1 with ``nvshmem_team_id_t``
"""
Teams = bindings.Team_id

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
