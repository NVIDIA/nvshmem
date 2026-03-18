# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import numpy as np
import pytest

import cutlass.cute as cute
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute
import nvshmem.core.interop.cute as cute_interop

from cuda.core import Device, system

from test_device_rma import (
    _compile_kernel,
    _nvshmem_stream,
    _fill_cute_tensor,
    _read_cute_tensor,
    _cute_dtype,
)

coll_dtypes = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

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


def _assert_tensor_equals(tensor, dtype, expected):
    host = _read_cute_tensor(tensor, dtype)
    if np.issubdtype(host.dtype, np.floating):
        assert np.allclose(host, expected)
    else:
        assert (host == expected).all()


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_device_reduce(nvshmem_init_fini, team, dtype, op):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    nelems = 16
    src = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    dest = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    _fill_cute_tensor(src, dtype, nvshmem.core.my_pe() + 1)
    _fill_cute_tensor(dest, dtype, 0)
    dev.sync()  # Sync after filling tensors, before kernel launch

    @cute.kernel
    def test_reduce_kernel(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.reduce(team, dest, src, op)

    @cute.jit
    def test_reduce_launcher(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        test_reduce_kernel(team, dest, src).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    compiled = _compile_kernel(test_reduce_launcher, team, dest, src)
    compiled(team, dest, src)
    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(team, stream=stream)
    stream.sync()  # Sync stream after barrier
    if op == "sum":
        expected = sum(range(1, nvshmem.core.n_pes() + 1))
    elif op == "min":
        expected = min(range(1, nvshmem.core.n_pes() + 1))
    else:
        expected = max(range(1, nvshmem.core.n_pes() + 1))

    _assert_tensor_equals(dest, dtype, expected)
    cute_interop.free_tensor(src)
    cute_interop.free_tensor(dest)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_device_reducescatter(nvshmem_init_fini, team, dtype, op):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    nelems = 16
    src = cute_interop.tensor((nelems * nvshmem.core.n_pes(), ), dtype=cute_dtype)
    dest = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    _fill_cute_tensor(src, dtype, nvshmem.core.my_pe() + 1)
    _fill_cute_tensor(dest, dtype, 0)
    dev.sync()  # Sync after filling tensors, before kernel launch

    @cute.kernel
    def test_reducescatter_kernel(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.reducescatter(team, dest, src, op)

    @cute.jit
    def test_reducescatter_launcher(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        test_reducescatter_kernel(team, dest, src).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    compiled = _compile_kernel(test_reducescatter_launcher, team, dest, src)
    compiled(team, dest, src)
    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(team, stream=stream)
    stream.sync()  # Sync stream after barrier
    if op == "sum":
        expected = sum(range(1, nvshmem.core.n_pes() + 1))
    elif op == "min":
        expected = min(range(1, nvshmem.core.n_pes() + 1))
    else:
        expected = max(range(1, nvshmem.core.n_pes() + 1))

    _assert_tensor_equals(dest, dtype, expected)
    cute_interop.free_tensor(src)
    cute_interop.free_tensor(dest)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_fcollect(nvshmem_init_fini, team, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    nelems = 16
    src = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    team_n = nvshmem.core.team_n_pes(team)
    dest = cute_interop.tensor((nelems * team_n, ), dtype=cute_dtype)
    _fill_cute_tensor(src, dtype, nvshmem.core.my_pe() + 1)
    _fill_cute_tensor(dest, dtype, 0)
    dev.sync()  # Sync after filling tensors, before kernel launch

    @cute.kernel
    def test_fcollect_kernel(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.fcollect(team, dest, src)

    @cute.jit
    def test_fcollect_launcher(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        test_fcollect_kernel(team, dest, src).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    compiled = _compile_kernel(test_fcollect_launcher, team, dest, src)
    compiled(team, dest, src)
    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(team, stream=stream)
    stream.sync()  # Sync stream after barrier
    expected = np.concatenate([np.full(nelems, pe + 1, dtype=_NUMPY_DTYPE_MAP[dtype]) for pe in range(team_n)])
    _assert_tensor_equals(dest, dtype, expected)
    cute_interop.free_tensor(src)
    cute_interop.free_tensor(dest)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_alltoall(nvshmem_init_fini, team, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    nelems = 16
    src = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    dest = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    _fill_cute_tensor(src, dtype, nvshmem.core.my_pe() + 1)
    _fill_cute_tensor(dest, dtype, 0)
    dev.sync()  # Sync after filling tensors, before kernel launch

    @cute.kernel
    def test_alltoall_kernel(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.alltoall(team, dest, src)

    @cute.jit
    def test_alltoall_launcher(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        test_alltoall_kernel(team, dest, src).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    compiled = _compile_kernel(test_alltoall_launcher, team, dest, src)
    compiled(team, dest, src)
    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(team, stream=stream)
    stream.sync()  # Sync stream after barrier
    chunk = nelems // nvshmem.core.n_pes()
    expected = np.concatenate(
        [np.full(chunk, pe + 1, dtype=_NUMPY_DTYPE_MAP[dtype]) for pe in range(nvshmem.core.n_pes())])
    _assert_tensor_equals(dest, dtype, expected)
    cute_interop.free_tensor(src)
    cute_interop.free_tensor(dest)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_NODE])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_broadcast(nvshmem_init_fini, team, dtype):
    if nvshmem.core.team_n_pes(team) < 2:
        pytest.skip("Need >1 PE in team for broadcast test")
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    nelems = 16
    src = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    dest = cute_interop.tensor((nelems, ), dtype=cute_dtype)
    _fill_cute_tensor(src, dtype, nvshmem.core.my_pe() + 1)
    _fill_cute_tensor(dest, dtype, 0)
    dev.sync()  # Sync after filling tensors, before kernel launch

    @cute.kernel
    def test_broadcast_kernel(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.broadcast(team, dest, src, root=0)

    @cute.jit
    def test_broadcast_launcher(team: Int32, dest: cute.Tensor, src: cute.Tensor):
        test_broadcast_kernel(team, dest, src).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    compiled = _compile_kernel(test_broadcast_launcher, team, dest, src)
    compiled(team, dest, src)
    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(team, stream=stream)
    stream.sync()  # Sync stream after barrier
    _assert_tensor_equals(dest, dtype, 1)
    cute_interop.free_tensor(src)
    cute_interop.free_tensor(dest)
