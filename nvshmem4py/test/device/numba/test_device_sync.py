from cuda.core import Device
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba

import pytest


@pytest.mark.mpi
@pytest.mark.parametrize("teams",
                         [nvshmem.core.Teams.TEAM_NODE, nvshmem.core.Teams.TEAM_WORLD, nvshmem.core.Teams.TEAM_SHARED])
@pytest.mark.parametrize(
    "func", [nvshmem.core.device.numba.sync, nvshmem.core.device.numba.sync_block, nvshmem.core.device.numba.sync_warp])
def test_device_sync(nvshmem_init_fini, teams, func):
    print(f"Testing {func.__name__} on team {teams}")

    nblocks = 1
    nthreads = 1
    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit
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
    nvshmem.core.device.numba.sync_all, nvshmem.core.device.numba.sync_all_block,
    nvshmem.core.device.numba.sync_all_warp
])
def test_device_sync_all(nvshmem_init_fini, func):
    print(f"Testing {func.__name__}")

    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit
    def test_sync_all():
        func()

    stream = dev.create_stream()

    test_sync_all[nblocks, nthreads, stream]()

    stream.sync()
    dev.sync()
    print("Done testing sync_all")
