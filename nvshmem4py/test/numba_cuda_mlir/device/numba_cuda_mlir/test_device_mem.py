# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.core import Device
from numba_cuda_mlir import cuda
import nvshmem.core
import nvshmem.core.device.numba_cuda_mlir

import pytest


@pytest.mark.mpi
def test_device_get_peer_array(nvshmem_init_fini):
    """
    Test device-side get_peer_array for inter-PE access via Numba kernel.

    ``get_peer_array`` is a thin wrapper over ``nvshmem_ptr``, which only
    returns a valid VA when the peer PE is reachable over NVLink (i.e. lives
    in the same NVLink domain as the local PE).  On multi-node testbeds that
    are connected only via TCP/IB (no cross-node NVLink fabric), ``TEAM_NODE``
    degenerates to a single PE per node and ``nvshmem_ptr`` is undefined for
    any other PE.  Skip in that case to avoid issuing an illegal load/store
    that would poison the CUDA context for every subsequent test.
    """
    # Only test when at least 2 PEs
    if nvshmem.core.n_pes() < 2:
        pytest.skip("Need at least 2 PEs for peer access")
    if nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        pytest.skip("Need >1 PE in NVLink domain (TEAM_NODE) for peer access test")

    nblocks = 1
    nthreads = 1

    dev = Device()
    dev.sync()

    # CuPy array allocated with NVSHMEM backend
    arr = nvshmem.core.array((4, ), dtype="int32")
    arr[:] = nvshmem.core.my_pe()

    @cuda.jit(lto=True)
    def peer_fetch_kernel(in_arr, pe):
        peer_arr = nvshmem.core.device.numba_cuda_mlir.get_peer_array(in_arr, pe)
        for i in range(in_arr.shape[0]):
            peer_arr[i] = nvshmem.core.device.numba_cuda_mlir.my_pe()

    node_team = nvshmem.core.Teams.TEAM_NODE
    node_rank = nvshmem.core.team_my_pe(node_team)
    node_size = nvshmem.core.team_n_pes(node_team)
    peer_node_rank = (node_rank + 1) % node_size
    peer_pe = nvshmem.core.team_translate_pe(node_team, peer_node_rank, nvshmem.core.Teams.TEAM_WORLD)
    predecessor_node_rank = (node_rank - 1) % node_size
    expected_pe = nvshmem.core.team_translate_pe(node_team, predecessor_node_rank, nvshmem.core.Teams.TEAM_WORLD)

    stream = dev.create_stream()

    peer_fetch_kernel[nblocks, nthreads, stream](arr, peer_pe)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    dev.sync()
    assert (arr == expected_pe).all(), f"Result {arr} did not match expected {expected_pe}"
    nvshmem.core.free_array(arr)


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

    On testbeds without a cross-node NVLink fabric (e.g. dual-node TCP/IB
    setups), TEAM_NODE collapses to a single PE per node and multicast is
    inherently unavailable; skip cleanly in that case.  The skip gate is
    evaluated before any CUDA work so that the test does not get tripped by
    a poisoned context inherited from a previous failure.
    """
    nblocks = 1
    nthreads = 1

    dev = Device()
    if not dev.properties.multicast_supported:
        pytest.skip("Multicast not supported on this platform")
    if nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        pytest.skip("Need >1 PE in NVLink domain (TEAM_NODE) for multicast test")
    dev.sync()

    # CuPy array allocated with NVSHMEM backend
    arr = nvshmem.core.array((4, ), dtype="float32")
    arr[:] = nvshmem.core.my_pe()

    @cuda.jit(lto=True)
    def multicast_fetch_kernel(team, in_arr):
        mc_arr = nvshmem.core.device.numba_cuda_mlir.get_multicast_array(team, in_arr)
        # Use team-relative rank so that the rank-0 PE within each NVLink
        # domain (TEAM_NODE) performs the write, making the test correct
        # across topologies with multiple NVLink switch domains.
        if nvshmem.core.device.numba_cuda_mlir.team_my_pe(team) == 0:
            for i in range(in_arr.shape[0]):
                mc_arr[i] = 1.0

    stream = dev.create_stream()

    multicast_fetch_kernel[nblocks, nthreads, stream](nvshmem.core.Teams.TEAM_NODE, arr)
    stream.sync()
    dev.sync()
    assert (arr == 1).all(), f"Multicast array result {arr} did not match expected {1}"
    nvshmem.core.free_array(arr)
