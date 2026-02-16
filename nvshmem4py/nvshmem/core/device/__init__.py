# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

__all__ = ["numba", "cute"]

from . import numba

# Lazy import for cute - only import when accessed
_cute_module = None

def __getattr__(name):
    """Lazy import of cute module."""
    if name == "cute":
        global _cute_module
        if _cute_module is None:
            from . import cute
            _cute_module = cute
        return _cute_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
