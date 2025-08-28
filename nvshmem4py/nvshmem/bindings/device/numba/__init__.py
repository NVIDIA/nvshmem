from numba import config
from ._numbast import *  # noqa: F403

import os

# TODO: use this when find_nvidia_header_directory is available
# from cuda.pathfinder import find_nvidia_header_directory

# INCLUDE_PATH = find_nvidia_header_directory("nvshmem")
# if "nvshmem.h" not in os.listdir(INCLUDE_PATH):
#     raise RuntimeError("nvshmem.h not found, package may not be properly installed")

# $1/nvshmem_pkg/lib/libnvshmem_host.so.3
LD_PRELOAD = os.environ.get("LD_PRELOAD", "")
if not LD_PRELOAD:
    raise RuntimeError("LD_PRELOAD is not set. Please set it to the path of the nvshmem library.")
if "libnvshmem_host.so" not in LD_PRELOAD:
    raise RuntimeError("libnvshmem_host.so not found in LD_PRELOAD, package may not be properly installed")

# $1/nvshmem_pkg/
PACKAGE_PATH = os.path.dirname(os.path.dirname(LD_PRELOAD))
# $1/nvshmem_pkg/include 
INCLUDE_PATH = os.path.join(PACKAGE_PATH, "include")

if not os.path.exists(INCLUDE_PATH):
    raise RuntimeError(f"NVSHMEM headers not found at {INCLUDE_PATH}. Please confirm that nvshmem is installed correctly.")


# Path to this folder to look for entry point file
this_folder = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(this_folder, "entry_point.h")):
    raise RuntimeError("entry_point.h not found, package may not be properly installed")

config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = ":".join([INCLUDE_PATH, this_folder])
