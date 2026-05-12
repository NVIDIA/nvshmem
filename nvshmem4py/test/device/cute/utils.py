# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import cuda.bindings.driver as cudrv
from cuda.core import Device
import cutlass.cute as cute
import numpy as np
import pytest

import nvshmem.core.interop.cute as cute_interop

_UTILS_PATH = Path(__file__).resolve().parents[2] / "utils.py"
_spec = importlib.util.spec_from_file_location("_nvshmem_test_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

uid_init = _utils.uid_init
mpi_init = _utils.mpi_init
get_local_rank_per_node = _utils.get_local_rank_per_node

_CUTE_DTYPE_MAP = {
    "float32": cute.Float32,
    "float64": cute.Float64,
    "int8": cute.Int8,
    "int16": cute.Int16,
    "int32": cute.Int32,
    "int64": cute.Int64,
    "uint8": cute.Uint8,
    "uint16": cute.Uint16,
    "uint32": cute.Uint32,
    "uint64": cute.Uint64,
}

_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _cute_dtype(dtype_name):
    dtype = _CUTE_DTYPE_MAP.get(dtype_name)
    if dtype is None:
        pytest.skip(f"CuTe dtype not supported for CuTe test: {dtype_name}")
    return dtype


def _fill_cute_tensor(tensor, dtype_name, value):
    np_dtype = _NUMPY_DTYPE_MAP[dtype_name]
    host = np.full(tuple(tensor.shape), np_dtype(value), dtype=np_dtype)
    buf, _, _ = cute_interop.tensor_get_buffer(tensor)
    cudrv.cuMemcpyHtoD(buf.handle, host, host.nbytes)
    Device().sync()


def _read_cute_tensor(tensor, dtype_name):
    np_dtype = _NUMPY_DTYPE_MAP[dtype_name]
    host = np.empty(tuple(tensor.shape), dtype=np_dtype)
    buf, _, _ = cute_interop.tensor_get_buffer(tensor)
    cudrv.cuMemcpyDtoH(host, buf.handle, host.nbytes)
    Device().sync()
    return host


__all__ = [
    "uid_init",
    "mpi_init",
    "get_local_rank_per_node",
    "_cute_dtype",
    "_fill_cute_tensor",
    "_read_cute_tensor",
]
