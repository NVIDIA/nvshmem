# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning

if os.path.exists(os.path.join(os.path.dirname(__file__), "rma.py")):
    from .rma import *
    from .direct import *
    from .amo import *
    from .collective import *
    __all__ = rma.__all__ + direct.__all__ + amo.__all__ + collective.__all__
else:
    warnings.warn("Numba device bindings are not enabled", NvshmemWarning)
    rma = None
    direct = None
    amo = None
