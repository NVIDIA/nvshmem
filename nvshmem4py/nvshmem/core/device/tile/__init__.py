# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""NVSHMEM device API for cuTile kernels.

    from nvshmem.core.device.tile import put_signal

``generated`` and the ``direct``, ``rma``, ``collective``, ``amo``, and ``mem``
modules are generated from the NVSHMEM device headers when nvshmem4py is built
with ``NVSHMEM4PY_BUILD_CUTILE_BINDINGS=ON``.
"""

from importlib.util import find_spec

if find_spec(__name__ + ".generated") is None:
    raise ImportError(f"{__name__} was not generated; build nvshmem4py with "
                      "NVSHMEM4PY_BUILD_CUTILE_BINDINGS=ON")

from . import amo, collective, direct, generated, mem, rma
from .generated import *
from .direct import *
from .rma import *
from .collective import *
from .amo import *
from .mem import *

__all__ = [
    *generated.__all__,
    *direct.__all__,
    *rma.__all__,
    *collective.__all__,
    *amo.__all__,
    *mem.__all__,
]
