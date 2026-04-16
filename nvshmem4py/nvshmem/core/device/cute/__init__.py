# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning

if os.path.exists(os.path.join(os.path.dirname(__file__), "rma.py")):
    from .rma import *
    from .direct import *
    from .amo import *
    from .collective import *
    from .mem import *
    __all__ = rma.__all__ + direct.__all__ + amo.__all__ + collective.__all__
else:
    warnings.warn("Numba device bindings are not enabled", NvshmemWarning)
    rma = None
    direct = None
    amo = None
