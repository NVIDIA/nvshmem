# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

__all__ = ["numba", "cute"]

# Lazy import for numba and cute - only import when accessed
_numba_module = None
_cute_module = None


def __getattr__(name):
    """Lazy import of numba and cute modules."""
    if name == "numba":
        global _numba_module
        if _numba_module is None:
            from . import numba
            _numba_module = numba
        return _numba_module
    if name == "cute":
        global _cute_module
        if _cute_module is None:
            from . import cute
            _cute_module = cute
        return _cute_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
