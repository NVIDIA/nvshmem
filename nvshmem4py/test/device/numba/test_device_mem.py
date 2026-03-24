import cupy as cp
from cuda.core import Device
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

    stream = dev.create_stream()

    peer_fetch_kernel[nblocks, nthreads, stream](arr, peer_pe)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    dev.sync()
    assert (arr == peer_pe).all(), f"Result {arr} did not match expected {peer_pe}"


@pytest.mark.mpi
def test_device_get_multicast_array(nvshmem_init_fini):
    """
    Test device-side get_multicast_array for multicast access via Numba kernel.

    Uses TEAM_NODE (NVLink domain) instead of TEAM_WORLD to support platforms
    such as H20 (X84, CUDA 12) that have multiple NVLink switch domains.  On
    those platforms TEAM_WORLD may span domain boundaries where multicast is
    unsupported, while TEAM_NODE correctly identifies each NVLink domain.  On
    GB200/GB300 (aarch64, CUDA 13) all PEs are in a single NVLink domain so
    TEAM_NODE == TEAM_WORLD and behaviour is unchanged.
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
        # Use team-relative rank so that the rank-0 PE within each NVLink
        # domain (TEAM_NODE) performs the write, making the test correct
        # across topologies with multiple NVLink switch domains.
        if nvshmem.core.device.numba.team_my_pe(team) == 0:
            for i in range(in_arr.shape[0]):
                mc_arr[i] = 1.0

    stream = dev.create_stream()

    multicast_fetch_kernel[nblocks, nthreads, stream](nvshmem.core.Teams.TEAM_NODE, arr)
    stream.sync()
    dev.sync()
    assert (arr == 1).all(), f"Multicast array result {arr} did not match expected {1}"
