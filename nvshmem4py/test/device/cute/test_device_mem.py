import numpy as np
import pytest
import torch

import cutlass.cute as cute
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE

import nvshmem.core
import nvshmem.core.device.cute.mem as nvshmem_cute_mem
import nvshmem.core.interop.cute as cute_interop

from cuda.core.experimental import Device, system

from test_device_rma import (
    _compile_kernel,
    _nvshmem_stream,
    _fill_cute_tensor,
    _read_cute_tensor,
    _cute_dtype,
)


@pytest.mark.mpi
def test_device_get_peer_tensor(nvshmem_init_fini):
    if nvshmem.core.n_pes() < 2:
        pytest.skip("Need at least 2 PEs for peer access")

    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.num_devices
    Device(local_rank).set_current()
    buf = cute_interop.tensor((4,), dtype=cute.Int32)
    _fill_cute_tensor(buf, "int32", nvshmem.core.my_pe())

    @cute.kernel
    def peer_fetch_kernel(arr: cute.Tensor, pe: Int32):
        peer_arr = nvshmem_cute_mem.get_peer_tensor(arr, pe)
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for i in range(4):
                peer_arr[i] = nvshmem.core.my_pe()

    @cute.jit
    def peer_fetch_launcher(arr: cute.Tensor, pe: Int32):
        peer_fetch_kernel(arr, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    peer_pe = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    compiled = _compile_kernel(peer_fetch_launcher, buf, 0)
    compiled(buf, peer_pe)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    torch.cuda.synchronize()

    expected = np.full((4,), peer_pe, dtype=np.int32)
    host = _read_cute_tensor(buf, "int32")
    assert (host == expected).all()

    cute_interop.free_tensor(buf)


@pytest.mark.mpi
def test_device_get_multicast_tensor(nvshmem_init_fini):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.num_devices
    Device(local_rank).set_current()
    if not Device().properties.multicast_supported:
        pytest.skip("Multicast not supported on this platform")
    if nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        pytest.skip("Need >1 PE for multicast test")

    buf = cute_interop.tensor((4,), dtype=cute.Float32)
    _fill_cute_tensor(buf, "float32", 0)

    @cute.kernel
    def multicast_fetch_kernel(team: Int32, arr: cute.Tensor):
        mc_arr = nvshmem_cute_mem.get_multicast_tensor(team, arr)
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0 and nvshmem.core.my_pe() == 0:
            for i in range(4):
                mc_arr[i] = 1.0

    @cute.jit
    def multicast_fetch_launcher(team: Int32, arr: cute.Tensor):
        multicast_fetch_kernel(team, arr).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(multicast_fetch_launcher, nvshmem.core.Teams.TEAM_WORLD, buf)
    compiled(nvshmem.core.Teams.TEAM_WORLD, buf)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    torch.cuda.synchronize()

    expected = np.full((4,), 1.0, dtype=np.float32)
    host = _read_cute_tensor(buf, "float32")
    assert (host == expected).all()

    cute_interop.free_tensor(buf)
