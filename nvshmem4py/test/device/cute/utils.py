# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

from cuda.core import Device
import cutlass.cute as cute
import pytest
import torch

import nvshmem.core.interop.cute as cute_interop

_UTILS_PATH = Path(__file__).resolve().parents[2] / "utils.py"
_spec = importlib.util.spec_from_file_location("_nvshmem_test_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

uid_init = _utils.uid_init
mpi_init = _utils.mpi_init
get_local_rank_per_node = _utils.get_local_rank_per_node

_CUTE_DTYPE_MAP = {
    "bfloat16": cute.BFloat16,
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

_TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}


def _cute_dtype(dtype_name):
    dtype = _CUTE_DTYPE_MAP.get(dtype_name)
    if dtype is None:
        pytest.skip(f"CuTe dtype not supported for CuTe test: {dtype_name}")
    return dtype


def _torch_view_of_cute_tensor(tensor, dtype_name):
    # Wrap the NVSHMEM-backed buffer behind a CuTe DSL tensor as a Torch view
    # via DLPack -- the same path ``nvshmem.core.interop.torch.tensor()`` uses
    # to expose NVSHMEM symmetric memory to Torch.  Routing host I/O through
    # Torch instead of raw ``cuMemcpy{HtoD,DtoH}`` avoids segfaults on
    # libcuda's VMM-mapped symmetric heap; raw cuMemcpyDtoH against that region
    # traps inside ``cuMemcpyDtoH_v2`` while libcuda walks its allocation table.
    buf, _, _ = cute_interop.tensor_get_buffer(tensor)
    torch_dtype = _TORCH_DTYPE_MAP[dtype_name]
    return torch.utils.dlpack.from_dlpack(buf).view(torch_dtype).view(tuple(tensor.shape))


def _fill_cute_tensor(tensor, dtype_name, value):
    view = _torch_view_of_cute_tensor(tensor, dtype_name)
    view.fill_(value)
    Device().sync()


def _read_cute_tensor(tensor, dtype_name):
    view = _torch_view_of_cute_tensor(tensor, dtype_name)
    if view.dtype is torch.bfloat16:
        # NumPy has no bfloat16; hand back the exact float32 promotion instead.
        view = view.float()
    return view.detach().cpu().numpy()


__all__ = [
    "uid_init",
    "mpi_init",
    "get_local_rank_per_node",
    "_cute_dtype",
    "_fill_cute_tensor",
    "_read_cute_tensor",
]
