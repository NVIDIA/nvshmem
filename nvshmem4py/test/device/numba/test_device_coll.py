from cuda.core import Device
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba
import cupy

import pytest

coll_dtypes = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
coll_scopes = ["", "_block", "_warp"]


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_device_reduce(nvshmem_init_fini, team, dtype, op):
    print(f"Testing {dtype} reduce on team {team}")

    nblocks = 1
    nthreads = 1
    nelems = 16
    src = nvshmem.core.array((nelems, ), dtype=dtype)
    src[:] = nvshmem.core.my_pe() + 1
    dest = nvshmem.core.array((nelems, ), dtype=dtype)
    dest[:] = 0
    cuda.synchronize()

    print(f"Src after init: {src}")
    print(f"Dest after init: {dest}")
    print(f"From PE {nvshmem.core.my_pe()}")

    @cuda.jit
    def test_reduce(team, dest, src):
        nvshmem.core.device.numba.reduce(team, dest, src, op)

    dev = Device()
    stream = dev.create_stream()

    test_reduce[nblocks, nthreads, stream](team, dest, src)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    print(f"Dest after reduce: {dest}")
    if op == "sum":
        expected = sum(range(1, nvshmem.core.n_pes() + 1))
    elif op == "min":
        expected = min(range(1, nvshmem.core.n_pes() + 1))
    elif op == "max":
        expected = max(range(1, nvshmem.core.n_pes() + 1))

    assert (dest == expected).all()

    nvshmem.core.free_array(src)
    nvshmem.core.free_array(dest)
    print("Done testing reduce")


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_device_reducescatter(nvshmem_init_fini, team, dtype, op):
    print(f"Testing {dtype} reducescatter on team {team}")

    nblocks = 1
    nthreads = 1
    nelems = 16
    src = nvshmem.core.array((nelems * nvshmem.core.n_pes(), ), dtype=dtype)
    src[:] = nvshmem.core.my_pe() + 1
    dest = nvshmem.core.array((nelems, ), dtype=dtype)
    dest[:] = 0
    cuda.synchronize()

    print(f"From PE {nvshmem.core.my_pe()}")
    print(f"Src after init: {src}")
    print(f"Dest after init: {dest}")

    @cuda.jit()
    def test_reducescatter(team, dest, src):
        nvshmem.core.device.numba.reducescatter(team, dest, src, op)

    dev = Device()
    stream = dev.create_stream()

    test_reducescatter[nblocks, nthreads, stream](team, dest, src)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    print(f"Dest after reducescatter: {dest}")
    if op == "sum":
        expected = sum(range(1, nvshmem.core.n_pes() + 1))
    elif op == "min":
        expected = min(range(1, nvshmem.core.n_pes() + 1))
    elif op == "max":
        expected = max(range(1, nvshmem.core.n_pes() + 1))

    assert (dest == expected).all()

    nvshmem.core.free_array(src)
    nvshmem.core.free_array(dest)
    print("Done testing reducescatter")


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_fcollect(nvshmem_init_fini, team, dtype):
    team_n = nvshmem.core.team_n_pes(team)
    nelems = 16
    src = nvshmem.core.array((nelems, ), dtype=dtype)
    dest = nvshmem.core.array((nelems * team_n, ), dtype=dtype)
    src[:] = nvshmem.core.my_pe() + 1
    dest[:] = 0
    cuda.synchronize()

    print(f"Src after init: {src}")
    print(f"Dest after init: {dest}")

    @cuda.jit
    def k(team, dest, src):
        nvshmem.core.device.numba.fcollect(team, dest, src)

    dev = Device()
    stream = dev.create_stream()
    k[1, 128, stream](team, dest, src)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    expected = []
    for pe in range(nvshmem.core.n_pes()):
        expected.extend([pe + 1] * nelems)
    print(f"Expected: {expected}")
    print(f"Dest: {dest}")
    assert (dest == cupy.asarray(expected)).all()


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_alltoall(nvshmem_init_fini, team, dtype):
    print(f"Testing {dtype} alltoall on team {team}")

    nblocks = 1
    nthreads = 1
    nelems = 16
    src = nvshmem.core.array((nelems, ), dtype=dtype)
    src[:] = nvshmem.core.my_pe() + 1
    dest = nvshmem.core.array((nelems), dtype=dtype)
    dest[:] = 0
    cuda.synchronize()

    print(f"From PE {nvshmem.core.my_pe()}")
    print(f"Src after init: {src}")
    print(f"Dest after init: {dest}")

    @cuda.jit
    def test_alltoall(dest, src, team):
        nvshmem.core.device.numba.alltoall(team, dest, src)

    dev = Device()
    stream = dev.create_stream()
    test_alltoall[nblocks, nthreads, stream](dest, src, team)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    expected = []
    for i in range(1, nvshmem.core.n_pes() + 1):
        expected.extend([i] * (nelems // nvshmem.core.n_pes()))
    print(f"Expected: {expected}")
    print(f"Dest: {dest}")
    assert (dest == cupy.asarray(expected)).all()

    nvshmem.core.free_array(src)
    nvshmem.core.free_array(dest)
    print("Done testing alltoall")


@pytest.mark.mpi
@pytest.mark.parametrize("team", [nvshmem.core.Teams.TEAM_WORLD])
@pytest.mark.parametrize("dtype", coll_dtypes)
def test_device_broadcast(nvshmem_init_fini, team, dtype):
    if nvshmem.core.team_n_pes(team) < 2:
        pytest.skip("Need >1 PE in team for broadcast test")
    print(f"Testing {dtype} broadcast on team {team}")

    nblocks = 1
    nthreads = 1
    nelems = 16
    src = nvshmem.core.array((nelems, ), dtype=dtype)
    src[:] = nvshmem.core.my_pe() + 1
    dest = nvshmem.core.array((nelems), dtype=dtype)
    dest[:] = 0
    cuda.synchronize()

    print(f"From PE {nvshmem.core.my_pe()}")
    print(f"Src after init: {src}")
    print(f"Dest after init: {dest}")

    @cuda.jit
    def test_broadcast(dest, src, team):
        nvshmem.core.device.numba.broadcast(team, dest, src, root=0)

    dev = Device()
    stream = dev.create_stream()
    test_broadcast[nblocks, nthreads, stream](dest, src, team)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    # Expect 1 (sent by PE 0) on all PEs
    print(f"Dest: {dest}")
    assert (dest == 1).all()

    nvshmem.core.free_array(src)
    nvshmem.core.free_array(dest)
    print("Done testing broadcast")
