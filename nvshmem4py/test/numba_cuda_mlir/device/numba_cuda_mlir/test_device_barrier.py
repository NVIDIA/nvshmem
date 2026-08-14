# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core import Device
from numba_cuda_mlir import cuda
import nvshmem.core
import nvshmem.core.device.numba_cuda_mlir

import pytest

_barrier_teams = [
    nvshmem.core.Teams.TEAM_NODE,
    pytest.param(nvshmem.core.Teams.TEAM_WORLD,
                 marks=pytest.mark.xfail(reason="proxy timeout on PCIe 2-node (Bug TBD)", strict=False)),
    nvshmem.core.Teams.TEAM_SHARED,
]


@pytest.mark.mpi
@pytest.mark.parametrize("teams", _barrier_teams)
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba_cuda_mlir.barrier, nvshmem.core.device.numba_cuda_mlir.barrier_block,
    nvshmem.core.device.numba_cuda_mlir.barrier_warp
])
def test_device_barrier(nvshmem_init_fini, teams, func):
    print(f"Testing {func.__name__} on team {teams}")

    nblocks = 1
    nthreads = 32  # Full warp required for barrier_warp
    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit(lto=True)
    def test_barrier(teams):
        func(teams)

    stream = dev.create_stream()

    test_barrier[nblocks, nthreads, stream](teams)
    nvshmem.core.barrier(teams, stream=stream)
    stream.sync()
    dev.sync()
    print("Done testing barrier")


@pytest.mark.mpi
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba_cuda_mlir.barrier_all, nvshmem.core.device.numba_cuda_mlir.barrier_all_block,
    nvshmem.core.device.numba_cuda_mlir.barrier_all_warp
])
def test_device_barrier_all(nvshmem_init_fini, func):
    print(f"Testing {func.__name__}")

    nblocks = 1
    nthreads = 32  # Full warp required for barrier_all_warp

    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit(lto=True)
    def test_barrier_all():
        func()

    stream = dev.create_stream()

    test_barrier_all[nblocks, nthreads, stream]()

    stream.sync()
    dev.sync()
    print("Done testing barrier_all")
