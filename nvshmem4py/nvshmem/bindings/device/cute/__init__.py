# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

from cuda.pathfinder import find_nvidia_header_directory

from cutlass import cute
from cutlass.cute import compile

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cuteast.py")):
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

    # Path to this folder to look for entry point file
    this_folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(this_folder, "entry_point.h")):
        raise RuntimeError("entry_point.h not found, package may not be properly installed")

else:
    warnings.warn("CuTe DSL device bindings are not enabled", NvshmemWarning)
    _cuteast = None
