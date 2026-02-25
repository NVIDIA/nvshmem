from cuda.core import Device, Stream
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba

import pytest


@pytest.mark.mpi
@pytest.mark.parametrize("teams",
                         [nvshmem.core.Teams.TEAM_NODE, nvshmem.core.Teams.TEAM_WORLD, nvshmem.core.Teams.TEAM_SHARED])
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba.barrier, nvshmem.core.device.numba.barrier_block, nvshmem.core.device.numba.barrier_warp
])
def test_device_barrier(nvshmem_init_fini, teams, func):
    print(f"Testing {func.__name__} on team {teams}")

    nblocks = 1
    nthreads = 1
    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit
    def test_barrier(teams):
        func(teams)

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(nb_stream.handle.value)

    test_barrier[nblocks, nthreads, nb_stream](teams)
    nvshmem.core.barrier(teams, stream=cu_stream_ref)
    cu_stream_ref.sync()
    dev.sync()
    print("Done testing barrier")


@pytest.mark.mpi
@pytest.mark.parametrize("func", [
    nvshmem.core.device.numba.barrier_all, nvshmem.core.device.numba.barrier_all_block,
    nvshmem.core.device.numba.barrier_all_warp
])
def test_device_barrier_all(nvshmem_init_fini, func):
    print(f"Testing {func.__name__}")

    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit
    def test_barrier_all():
        func()

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(nb_stream.handle.value)

    test_barrier_all[nblocks, nthreads, nb_stream]()

    cu_stream_ref.sync()
    dev.sync()
    print("Done testing barrier_all")
