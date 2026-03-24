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

from cuda.core import Device

from test_device_rma import (
    _compile_kernel,
)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_NODE])
def test_device_sync(nvshmem_init_fini, team):
    dev = Device()
    dev.set_current()

    @cute.kernel
    def test_sync_kernel(team: Int32):
        nvshmem_cute.sync(team)

    @cute.jit
    def test_sync_launcher(team: Int32):
        test_sync_kernel(team).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    nvshmem.core.barrier(team)
    print(f"Before sync from {nvshmem.core.my_pe()}")
    compiled = _compile_kernel(test_sync_launcher, team)
    dev.sync()
    compiled(team)
    dev.sync()
    nvshmem.core.barrier(team)


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
def test_device_barrier(nvshmem_init_fini, team):
    dev = Device()
    dev.set_current()

    @cute.kernel
    def test_barrier_kernel(team: Int32):
        nvshmem_cute.barrier(team)

    @cute.jit
    def test_barrier_launcher(team: Int32):
        test_barrier_kernel(team).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            cooperative=True,
        )

    nvshmem.core.barrier(team)
    dev.sync()
    compiled = _compile_kernel(test_barrier_launcher, team)
    compiled(team)
    dev.sync()
    print(f"After barrier from {nvshmem.core.my_pe()}")
    nvshmem.core.barrier(team)
