# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core import Device
from numba_cuda_mlir import cuda
import nvshmem.core
import nvshmem.core.device.numba_cuda_mlir

import pytest

_sync_teams = [
    nvshmem.core.Teams.TEAM_NODE,
    pytest.param(nvshmem.core.Teams.TEAM_WORLD,
                 marks=pytest.mark.xfail(reason="proxy timeout on PCIe 2-node (Bug TBD)", strict=False)),
    nvshmem.core.Teams.TEAM_SHARED,
]


@pytest.mark.mpi
@pytest.mark.parametrize("teams", _sync_teams)
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba_cuda_mlir.sync, nvshmem.core.device.numba_cuda_mlir.sync_block,
    nvshmem.core.device.numba_cuda_mlir.sync_warp
])
def test_device_sync(nvshmem_init_fini, teams, func):
    print(f"Testing {func.__name__} on team {teams}")

    nblocks = 1
    nthreads = 1
    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit(lto=True)
    def test_sync(teams):
        func(teams)

    stream = dev.create_stream()

    test_sync[nblocks, nthreads, stream](teams)
    nvshmem.core.barrier(teams, stream=stream)
    stream.sync()
    dev.sync()
    print("Done testing sync")


@pytest.mark.mpi
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba_cuda_mlir.sync_all, nvshmem.core.device.numba_cuda_mlir.sync_all_block,
    nvshmem.core.device.numba_cuda_mlir.sync_all_warp
])
def test_device_sync_all(nvshmem_init_fini, func):
    print(f"Testing {func.__name__}")

    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit(lto=True)
    def test_sync_all():
        func()

    stream = dev.create_stream()

    test_sync_all[nblocks, nthreads, stream]()

    stream.sync()
    dev.sync()
    print("Done testing sync_all")
