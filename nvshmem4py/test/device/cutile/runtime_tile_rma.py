# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank runtime gate for shared-memory Tile operands of the CuTile TileLib.

Run from an environment with a TileIR 13.5 CuTile tuple, a CuTile-enabled
NVSHMEM4Py wheel, and two visible GPUs:

    NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED=1 \
      mpirun -np 2 python nvshmem4py/test/device/cutile/runtime_tile_rma.py

A Tile operand's extent is part of its Tile IR type, so the compiler names
each extent's specialization separately while the NVSHMEM shim receives the
element count at runtime. One kernel puts two Tiles of different extents; each
destination must receive exactly its Tile and nothing past it. A second kernel
gets into a Tile destination, which the API returns as the updated Tile.
"""

import gc
import os
from typing import Any

EXTENTS = (32, 64)
SIZE = 128
NUM_WORKER_WARPS = 4


def _make_tile_put_kernel(ct: Any, my_pe: Any, n_pes: Any, put_block: Any) -> Any:

    @ct.kernel(num_worker_warps=NUM_WORKER_WARPS)
    def tile_put_two_extents(
        source: ct.Array,
        narrow_destination: ct.Array,
        wide_destination: ct.Array,
        narrow_extent: ct.Constant[int],
        wide_extent: ct.Constant[int],
    ):
        peer = (my_pe() + 1) % n_pes()
        narrow = ct.load(source, (0, ), shape=(narrow_extent, ))
        wide = ct.load(source, (0, ), shape=(wide_extent, ))
        put_block(narrow_destination, narrow, peer)
        put_block(wide_destination, wide, peer)

    return tile_put_two_extents


def _make_tile_get_kernel(ct: Any, my_pe: Any, n_pes: Any, get_block: Any) -> Any:

    @ct.kernel(num_worker_warps=NUM_WORKER_WARPS)
    def tile_get(source: ct.Array, fetched: ct.Array, extent: ct.Constant[int]):
        peer = (my_pe() + 1) % n_pes()
        tile = ct.full((extent, ), -2.0, dtype=ct.float32)
        tile = get_block(tile, source, peer)
        ct.store(fetched, (0, ), tile)

    return tile_get


def run() -> None:
    """Put two Tiles of different extents to, and get a Tile from, the next rank."""
    from cuda.core.experimental import Device, system
    from mpi4py import MPI
    import cuda.tile as ct
    import nvshmem.core
    from nvshmem.core.device.tile import get_block, my_pe, n_pes, put_block
    import torch

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        raise RuntimeError(f"CuTile Tile runtime test requires exactly two MPI ranks, got {comm.Get_size()}")
    rank = comm.Get_rank()
    device = Device(rank % system.get_num_devices())
    device.set_current()

    initialized = False
    kernels: list[Any] = []
    buffers: list[Any] = []
    try:
        nvshmem.core.init(device=device, mpi_comm=comm, initializer_method="mpi")
        initialized = True
        source, narrow, wide, fetched = (nvshmem.core.tensor((SIZE, ), dtype=torch.float32) for _ in range(4))
        buffers = [source, narrow, wide, fetched]
        source[:] = float(rank + 1)
        for destination in (narrow, wide, fetched):
            destination[:] = -1.0
        device.sync()
        comm.Barrier()

        kernels = [
            _make_tile_put_kernel(ct, my_pe, n_pes, put_block),
            _make_tile_get_kernel(ct, my_pe, n_pes, get_block),
        ]
        stream = torch.cuda.current_stream()
        ct.launch(stream, (1, 1), kernels[0], (source, narrow, wide, *EXTENTS))
        ct.launch(stream, (1, 1), kernels[1], (source, fetched, EXTENTS[0]))
        device.sync()
        comm.Barrier()
        device.sync()

        predecessor = (rank - 1) % comm.Get_size()
        successor = (rank + 1) % comm.Get_size()
        for extent, destination, peer, operation in (
            (EXTENTS[0], narrow, predecessor, "put"),
            (EXTENTS[1], wide, predecessor, "put"),
            (EXTENTS[0], fetched, successor, "get"),
        ):
            expected = torch.full((SIZE, ), -1.0, dtype=torch.float32)
            expected[:extent] = float(peer + 1)
            torch.testing.assert_close(
                destination.detach().cpu(),
                expected,
                msg=f"rank {rank}: extent-{extent} Tile {operation} delivered the wrong elements")
        print(f"rank {rank}: Tile puts of extents {EXTENTS} and Tile get of extent {EXTENTS[0]} passed")
    finally:
        if initialized:
            device.sync()
        # Release the CUDA library before freeing NVSHMEM state.
        kernels.clear()
        gc.collect()
        if initialized:
            for tensor in buffers:
                nvshmem.core.free_tensor(tensor)
            nvshmem.core.finalize()


if __name__ == "__main__":
    if os.environ.get("NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED") != "1":
        raise SystemExit("Set NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED=1 before running this distributed test")
    run()
