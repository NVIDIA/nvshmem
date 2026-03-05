import cupy as cp
from cuda.core import Device, Stream
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba

import pytest


@pytest.mark.mpi
def test_device_get_peer_array(nvshmem_init_fini):
    """
    Test device-side get_peer_array for inter-PE access via Numba kernel
    """
    # Only test when at least 2 PEs
    if nvshmem.core.n_pes() < 2:
        pytest.skip("Need at least 2 PEs for peer access")

    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    # CuPy array allocated with NVSHMEM backend
    arr = nvshmem.core.array((4, ), dtype="int32")
    arr[:] = nvshmem.core.my_pe()

    @cuda.jit
    def peer_fetch_kernel(in_arr, pe):
        peer_arr = nvshmem.core.device.numba.get_peer_array(in_arr, pe)
        for i in range(in_arr.shape[0]):
            peer_arr[i] = nvshmem.core.device.numba.my_pe()

    # choose src_pe/peer
    my_pe = nvshmem.core.my_pe()
    peer_pe = (my_pe + 1) % nvshmem.core.n_pes()

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    peer_fetch_kernel[nblocks, nthreads, nb_stream](arr, peer_pe)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=cu_stream_ref)
    cu_stream_ref.sync()
    dev.sync()
    assert (arr == peer_pe).all(), f"Result {arr} did not match expected {peer_pe}"


@pytest.mark.mpi
def test_device_get_multicast_array(nvshmem_init_fini):
    """
    Test device-side get_multicast_array for multicast access via Numba kernel
    """
    # Only test if multicast teams are available (skip if not supported)
    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    if not dev.properties.multicast_supported or nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping MC memory test because Multicast memory is not supported on this platform")
        pytest.skip("Skipping MC memory test because Multicast memory is not supported on this platform")

    # CuPy array allocated with NVSHMEM backend
    arr = nvshmem.core.array((4, ), dtype="float32")
    arr[:] = nvshmem.core.my_pe()

    @cuda.jit
    def multicast_fetch_kernel(team, in_arr):
        mc_arr = nvshmem.core.device.numba.get_multicast_array(team, in_arr)
        if nvshmem.core.device.numba.my_pe() == 0:
            for i in range(in_arr.shape[0]):
                in_arr[i] = nvshmem.core.device.numba.my_pe() + 1

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    multicast_fetch_kernel[nblocks, nthreads, nb_stream](nvshmem.core.Teams.TEAM_WORLD, arr)
    cu_stream_ref.sync()
    dev.sync()
    assert (arr == 1).all(), f"Multicast array result {arr} did not match expected {1}"
