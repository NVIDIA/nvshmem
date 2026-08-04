# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.pathfinder import find_nvidia_header_directory

from cutlass import cute
from cutlass.cute import compile

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cuteast.py")):
    if not hasattr(cute, "extern"):
        raise RuntimeError("NVSHMEM CuTe DSL bindings require nvidia-cutlass-dsl>=4.5.2; "
                           "the installed version does not provide cute.extern.")

    from ._cuteast import *
    from nvshmem.core.nvshmem_types import Teams

    INCLUDE_PATH = find_nvidia_header_directory("nvshmem")
    if not os.path.isdir(INCLUDE_PATH):
        raise RuntimeError(
            f"NVSHMEM headers not found at {INCLUDE_PATH}. Please confirm that nvshmem is installed correctly.")

    if "nvshmem.h" not in os.listdir(INCLUDE_PATH):
        raise RuntimeError("nvshmem.h not found, package may not be properly installed")

    CCCL_INCLUDE_PATH = find_nvidia_header_directory("cccl")

    if not os.path.exists(CCCL_INCLUDE_PATH):
        raise RuntimeError(
            f"CCCL headers not found at {CCCL_INCLUDE_PATH}. Please confirm that cccl is installed correctly.")

else:
    warnings.warn("CuTe DSL device bindings are not enabled", NvshmemWarning)
    _cuteast = None
