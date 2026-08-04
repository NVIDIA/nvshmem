# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVSHMEM device bitcode used by CuTe DSL ``@cute.extern`` declarations."""

import cutlass.cute as cute

from nvshmem.core.init_fini import find_device_bitcode_library


def _device_bitcode_path() -> str:
    """Locate the architecture-matched NVSHMEM device bitcode library."""
    return find_device_bitcode_library()


class _NVSHMEMDeviceBitCode(cute.BitCode):
    """Resolve the bitcode path when CuTe traces an extern call.

    ``find_device_bitcode_library`` selects a per-SM library from the current
    CUDA device.  Binding modules can be imported before rank-local device
    selection, so resolving it eagerly at import time can link the wrong SM.
    """

    def __init__(self):
        # ``BitCode`` is a frozen dataclass. Its ``path`` property below is
        # intentionally lazy, so there is no value to initialize here.
        pass

    @property
    def path(self) -> str:
        return _device_bitcode_path()


NVSHMEM_DEVICE_BITCODE = _NVSHMEMDeviceBitCode()
