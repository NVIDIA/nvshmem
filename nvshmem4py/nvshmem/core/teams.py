# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

"""
The following are NVSHMEM functions used for team management
"""

from nvshmem.core.nvshmem_types import Teams, NvshmemError

import nvshmem.bindings as bindings
from nvshmem.bindings import TeamConfig

import ctypes
from typing import Union
import logging

logger = logging.getLogger("nvshmem")

__all__ = ["team_split_strided", "team_split_2d", "team_destroy", "team_init", "team_translate_pe", "TeamConfig", "TeamUniqueId", "get_team_unique_id"]

TeamUniqueId = bindings.team_uniqueid

def get_team_unique_id() -> TeamUniqueId:
    """Get a new team unique ID.
    
    This function calls the underlying NVSHMEM function to generate a unique ID
    for team creation. The unique ID is guaranteed to be unique across all processes.
    
    Returns:
        TeamUniqueId: A new team unique ID object.
        
    Raises:
        NvshmemError: If getting the unique ID fails.
    """
    unique_id = TeamUniqueId()
    bindings.team_get_uniqueid(unique_id.ptr)
    return unique_id

def team_split_strided(parent_team: Teams, start: int, stride: int, size: int, config: TeamConfig, config_mask: int, new_team_name=None) -> Teams:
    """Split a parent team into a new team using strided distribution.
    
    This function creates a new team by selecting a subset of processes from the parent team
    using a strided distribution pattern. The new team will contain processes at positions
    start, start + stride, start + 2*stride, ..., start + (size-1)*stride.
    
    Args:
        parent_team: The parent team to split from.
        start: The starting process index in the parent team.
        stride: The stride between selected processes.
        size: The number of processes to include in the new team.
        config: Configuration for the new team.
        config_mask: Mask specifying which configuration options to apply.
        new_team_name: Optional name for the new team. If None, a default name is generated.
        
    Returns:
        Teams: The newly created team.
        
    Raises:
        NvshmemError: If team splitting fails.
        
    Note:
        If new_team_name is None, the default name will be "TEAM_{handle_value}" where
        handle_value is the numeric handle returned by the underlying NVSHMEM function.
    """
    new_team_handle = ctypes.c_int()

    bindings.team_split_strided(parent_team, start, stride, size, config.ptr, config_mask, ctypes.addressof(new_team_handle))

    if new_team_name is None:
        new_team_name = f"TEAM_{new_team_handle.value}"
    
    # team_split_strided behaves like team_split_2d for non-participating PEs
    # Therefore it's not an error to get a -1 handle. The return code of the binding is the error condition. However, we want to print a debug log.
    if new_team_handle.value == -1:
        logger.debug(f"Got -1 back as team handle for team {new_team_name}. This PE is not part of the team")
        return None

    Teams.add(new_team_name, new_team_handle.value)
    return Teams[new_team_name]

def team_split_2d(parent_team: Teams, xrange: int, xaxis_config: TeamConfig, xaxis_mask: int, yaxis_config: TeamConfig, yaxis_mask: int, new_team_name=None) -> tuple[Teams, Teams]:
    """Split a parent team into two 2D teams.
    
    This function creates two new teams from a parent team, organizing processes in a 2D grid.
    The first team represents the x-axis with xrange processes, and the second team represents
    the y-axis with the remaining processes.
    
    Args:
        parent_team: The parent team to split from.
        xrange: The number of processes to include in the x-axis team.
        xaxis_config: Configuration for the x-axis team.
        xaxis_mask: Mask specifying which x-axis configuration options to apply.
        yaxis_config: Configuration for the y-axis team.
        yaxis_mask: Mask specifying which y-axis configuration options to apply.
        new_team_name: Optional base name for the new teams. If None, default names are generated.
        
    Returns:
        tuple[Teams, Teams]: A tuple containing (x_axis_team, y_axis_team).
        
    Raises:
        NvshmemError: If team splitting fails.
        
    Note:
        If new_team_name is None, the default names will be "TEAM_X_{handle_value}" and 
        "TEAM_Y_{handle_value}" where handle_value is the numeric handle returned by the 
        underlying NVSHMEM function. If new_team_name is provided, the teams will be named
        "{new_team_name}_X" and "{new_team_name}_Y".
    """
    xaxis_team_handle = ctypes.c_int()
    yaxis_team_handle = ctypes.c_int()

    bindings.team_split_2d(parent_team, xrange, xaxis_config.ptr, xaxis_mask, yaxis_config.ptr, yaxis_mask, ctypes.addressof(xaxis_team_handle), ctypes.addressof(yaxis_team_handle))

    if new_team_name is None:
        xaxis_team_name = f"TEAM_X_{xaxis_team_handle.value}"
        yaxis_team_name = f"TEAM_Y_{yaxis_team_handle.value}"
    else:
        xaxis_team_name = f"{new_team_name}_X"
        yaxis_team_name = f"{new_team_name}_Y"

    # This will happen for PEs that are not part of the team
    # Therefore it's not an error. The return code of the binding is the error condition. However, we want to print a debug log.
    if xaxis_team_handle.value == -1 or yaxis_team_handle.value == -1:
        logger.debug(f"Got -1 back as team handle for team {xaxis_team_name} or {yaxis_team_name}. This PE is not part of one of the two teams")

    if xaxis_team_handle.value == -1:
        logger.debug(f"Got -1 back as team handle for team {xaxis_team_name}. This PE is not part of the x-axis team")
        xaxis_team = None
    else:
        Teams.add(xaxis_team_name, xaxis_team_handle.value)
        xaxis_team = Teams[xaxis_team_name]
    if yaxis_team_handle.value == -1:
        logger.debug(f"Got -1 back as team handle for team {yaxis_team_name}. This PE is not part of the y-axis team")
        yaxis_team = None
    else:
        Teams.add(yaxis_team_name, yaxis_team_handle.value)
        yaxis_team = Teams[yaxis_team_name]
    
    return xaxis_team, yaxis_team

def team_destroy(team: Union[int, str]):
    """Destroy a team and remove it from the team registry.
    
    This function destroys the specified team and removes it from the internal team registry.
    The team can be specified either by its handle (integer) or by its name (string).
    
    Args:
        team: The team to destroy, specified either as an integer handle or string name.
        
    Note:
        If a team handle is provided, the function will attempt to destroy it regardless
        of whether it exists in the registry. The underlying C API gracefully handles
        failures for non-existent teams.
        
    Raises:
        KeyError: If a team name is provided but the team is not found in the registry.
    """
    if isinstance(team, int):
        # Accepting the team handle regardless of what it is is desired behavior
        # The C API gracefully handles failure
        bindings.team_destroy(team)
        Teams.remove_by_value(team)
        return
    team_name = team
    # Using Teams[team_name] will Except if the team is not found
    bindings.team_destroy(Teams[team_name])
    Teams.remove(team_name)

def team_init(team_config: TeamConfig, config_mask: int, npes: int, pe_idx_in_team: int, new_team_name=None) -> Teams:
    """Initialize a new team with the specified configuration.
    
    This function creates a new team with the given configuration and adds it to the
    internal team registry. The team will contain npes processes, and the current
    process will have index pe_idx_in_team within the team.
    
    Args:
        team_config: Configuration for the new team.
        config_mask: Mask specifying which configuration options to apply.
        npes: Number of processes in the team.
        pe_idx_in_team: Index of the current process within the team.
        new_team_name: Optional name for the new team. If None, a default name is generated.
        
    Returns:
        Teams: The newly created team.
        
    Raises:
        NvshmemError: If team initialization fails
        
    Note:
        If new_team_name is None, the default name will be "TEAM_{handle_value}" where
        handle_value is the numeric handle returned by the underlying NVSHMEM function.
    """
    new_team_handle = ctypes.c_int()
    bindings.team_init(ctypes.addressof(new_team_handle), team_config.ptr, config_mask, npes, pe_idx_in_team)
    if new_team_name is None:
        new_team_name = f"TEAM_{new_team_handle.value}"
    
    # For team_init, only PEs who are part of the team call this function
    # So this will always be an error
    if new_team_handle.value == -1:
        raise NvshmemError(f"Failed to create team {new_team_name}")

    Teams.add(new_team_name, new_team_handle.value)

    return Teams[new_team_name]

def team_translate_pe(src_team: Teams, src_pe: int, dest_team: Teams) -> int:
    """Translate a process index from one team to another.
    
    This function maps a process index from a source team to the corresponding
    process index in a destination team. This is useful for communication
    between different teams that may have different process ordering.
    
    Args:
        src_team: The source team containing the process to translate.
        src_pe: The process index in the source team.
        dest_team: The destination team to translate the process index to.
        
    Returns:
        int: The process index in the destination team corresponding to src_pe in src_team.
        
    Raises:
        NvshmemError: If the translation operation fails.
    """
    return bindings.team_translate_pe(src_team, src_pe, dest_team)
