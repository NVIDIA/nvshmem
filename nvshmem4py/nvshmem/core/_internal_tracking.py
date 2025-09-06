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
Internal tracking for NVShmem

This file contains things like buffer management, status, etc.
"""
from enum import IntEnum

"""
Map of Device IDs from cuda.core to MemoryResource (NvshmemResource) objects
Used to avoid re-creating NvshmemResources every time someone calls nvshmem.core.allocate()
"""
_mr_references = {}

"""
class for Internal Init Status
"""
class InternalInitStatus(IntEnum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    DE_INITIALIZED = 2 # Keeps bootstrap

"""
Set to True after initializing. Used for safety checks before functions
"""
_is_initialized = {"status": InternalInitStatus.UNINITIALIZED}

"""
Each NVSHMEM process needs to be assocaited with a device. We cache that here.
"""
_cached_device = {"device": None}

"""
Debug mode is used to avoid redundant calls to Device()
"""
_debug_mode = False
