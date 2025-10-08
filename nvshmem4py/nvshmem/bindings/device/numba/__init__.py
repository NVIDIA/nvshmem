# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

from cuda.pathfinder import find_nvidia_header_directory

from numba import cuda 

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning

if os.path.exists(os.path.join(os.path.dirname(__file__), "_numbast.py")):
    from ._numbast import *

    INCLUDE_PATH = find_nvidia_header_directory("nvshmem")
    if "nvshmem.h" not in os.listdir(INCLUDE_PATH):
        raise RuntimeError("nvshmem.h not found, package may not be properly installed")

    if not os.path.exists(INCLUDE_PATH):
        raise RuntimeError(f"NVSHMEM headers not found at {INCLUDE_PATH}. Please confirm that nvshmem is installed correctly.")

    # Path to this folder to look for entry point file
    this_folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(this_folder, "entry_point.h")):
        raise RuntimeError("entry_point.h not found, package may not be properly installed")

    cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = ":".join([INCLUDE_PATH, this_folder])

else:
    warnings.warn("Numba device bindings are not enabled", NvshmemWarning)
    _numbast = None
