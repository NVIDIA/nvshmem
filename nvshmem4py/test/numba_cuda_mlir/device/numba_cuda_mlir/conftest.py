# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from utils import uid_init, mpi_init, get_local_rank_per_node
from nvshmem.core import finalize, barrier, Teams
from cuda.core import Device


def pytest_addoption(parser):
    parser.addoption("--init-type",
                     action="store",
                     default="uid",
                     help="Method to initialize NVSHMEM",
                     choices=["uid", "mpi"])


@pytest.fixture(scope="session", autouse=True)
def nvshmem_init_fini(request):
    init_type = request.config.getoption("--init-type")
    if init_type == "uid":
        uid_init()
    elif init_type == "mpi":
        mpi_init()

    # Sync device before first test to ensure init is complete
    local_rank = get_local_rank_per_node()
    dev = Device(local_rank)
    dev.set_current()
    dev.sync()

    yield

    barrier(Teams.TEAM_WORLD, stream=dev.create_stream())
    dev.sync()  # Ensure all kernels are complete on this device before finalization
    finalize()


@pytest.fixture(autouse=True)
def mpi_test_sync():
    """MPI barrier before and after each test to keep PEs synchronized.

    Without this, faster PEs can advance to the next test while slower ones
    are still finishing, causing nvshmem barrier counter mismatches on
    multi-node PCIe systems.
    """
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    yield
    MPI.COMM_WORLD.Barrier()
