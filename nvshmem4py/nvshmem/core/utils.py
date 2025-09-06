# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

"""
The following are utility functions for NVSHMEM4Py
"""

import logging
import sys
import os
import socket
import threading
from enum import IntEnum

import numpy as np

import nvshmem.bindings as bindings
from nvshmem.core._internal_tracking import _mr_references, _cached_device, _debug_mode

from cuda.core.experimental import Device, system
from cuda.core.experimental._memory import Buffer

try:
    import torch
    torch_to_numpy = {
            torch.float16: np.float16,
            torch.bfloat16: np.float16,  # Note: bfloat16 isn't supported by default in NumPy
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bool: np.bool_,
        }
    _torch_enabled = True
except:
    torch_to_numpy = None
    _torch_enabled = False

def _get_device() -> Device:
    """
    Perform a pre-flight check on the cuda Device in use, making sure it's ready for use in nvshmem
    This function returns the Device that is current before it was called, and sets the current device 
    to the cached device that this NVSHMEM PE was registered against at init time.

    The caller MUST set the same device back to current at the end of the API calling this.

    This function raises an Exception if no device is set to current.

    The expected, required flow for an NVSHMEM4Py.Core API is:
        Call ``_get_device()``, save old device
                internally, this sets current device to the correct device this PE was initialized with
        Perform API call
        Set old device back to current. If the old device is the same as the current one, this is idempotent
        If not, it matches the NVSHMEM4Py and Libnvshmem contract - return the process to the caller as they gave it to us
    """
    if _debug_mode:
        # Device() excepts if no device is current
        old_device = Device()
    else:
        old_device = None

    if not _debug_mode and _cached_device["device"] is not None:
        return _cached_device["device"], None
        
    # If the device is None here, we want to do the second half of the two-stage init.
    if _cached_device["device"] is None:
        _cached_device["device"] = Device()

    if _cached_device["device"].device_id != Device().device_id:
        _cached_device["device"].set_current()

    return _cached_device["device"], old_device

def _configure_logging(level="WARNING", logfile=None, mype=None):
    """
    Configure the Python logger for NVSHMEM or related modules.

    NOTE: mype can be retrieved with ``nvshmem.core.my_pe()``

    Args:
        level (str): Logging level as a string. Options are 'DEBUG', 'INFO', 
                     'WARNING', 'ERROR', or 'CRITICAL'. Defaults to 'WARNING'.
        logfile (str, optional): If provided, logs will also be written to this file.

    Example:
        >>> configure_logging("DEBUG")
        >>> logging.debug("This is a debug message.")
    """
    if mype is None:
        try:
            mype = bindings.my_pe()
        except:
            print("Unable to retrieve PE. Ensure NVSHMEM is initialized")
            mype = -1
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # C NVSHMEM Log Format: host:pid:tid [PE] NVSHMEM <log_level> 
    host = socket.getfqdn().split(".")[0]
    pid = os.getpid()
    tid = threading.get_native_id()
    # Note, my preference is to include time in the log fmt, but libnvshmem doesn't do this.
    formatter = logging.Formatter(f"{host}:{pid}:{tid} [{mype}] " + "NVSHMEM %(levelname)s : %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Set up the root logger
    logger = logging.getLogger("nvshmem")
    logger.setLevel(numeric_level)
    logger.handlers = []  # Remove any default handlers
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def get_size(shape, dtype):
    """
    Calculate the total size in bytes of an array with the given shape and dtype.

    Parameters:
        shape (tuple or list of int): Shape of the array (e.g., (64, 64)).
        
        dtype (str, np.dtype, or torch.dtype): Data type (e.g., 'float32', torch.float64).

    Returns:
        
        int: Total size in bytes.
    """
    # Normalize dtype to np.dtype

    if _torch_enabled and isinstance(dtype, torch.dtype):
        if not torch_to_numpy or dtype not in torch_to_numpy:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        dtype = np.dtype(torch_to_numpy[dtype])
    else:
        # Special case BF16 because numpy doesn't support it
        if dtype == "bfloat16":
            dtype = np.float16
        dtype = np.dtype(dtype)

    num_elements = np.prod(shape)
    return int(num_elements * dtype.itemsize)

def dtype_nbytes(dtype: str) -> int:
    """Return the size in bytes of a single element of the given NVSHMEM dtype.

    Accepts NVSHMEM dtype names such as ``'float'``, ``'int64'``, ``'half'``, etc.
    Raises a ``ValueError`` if the dtype is not recognized.

    Args:
        dtype (str): The NVSHMEM data type name.

    Returns:
        int: The number of bytes required to store a single element of the given dtype.

    Raises:
        ``ValueError``: If the specified dtype is not supported.
    """
    dtype_sizes = {
        "half": 2,
        "bfloat16": 2,
        "float": 4,
        "double": 8,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "int64_t": 8,
        "uint64_t": 8,
        "long": 8,
        "longlong": 8,
        "ulonglong": 8,
        "ptrdiff": 8,
        "fp16": 2,
        "bf16": 2,
        "int": 4,
        "int64": 8,
        "char": 1,
        "schar": 1,
        "size": 8,  # assuming size_t is 64-bit
    }

    try:
        return dtype_sizes[dtype]
    except KeyError:
        raise ValueError(f"Unsupported NVSHMEM dtype: {dtype}")
