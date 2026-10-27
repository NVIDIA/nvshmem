# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank runtime gate for NVSHMEM operations on type-erased ``void *`` operands.

Run from an environment with a TileIR 13.5 CuTile tuple, a CuTile-enabled
NVSHMEM4Py wheel, and two P2P-connected GPUs:

    NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED=1 \
      mpirun -np 2 python nvshmem4py/test/device/cutile/runtime_pointer_views.py

Each such operation is declared once per element type and selected from its
operands' dtype, and its count is inferred from the operands. One kernel
stores through ``get_peer_array``'s view of the next rank's float32 Array;
another puts bfloat16 data with ``putmem_block`` from an Array and from a Tile
and with the 64-bit-unit ``put64_block``. A float32 ``alltoall_block`` covers a
count inferred per PE of the team.
"""

import gc
import os
from typing import Any

ELEMENTS = 64
NUM_WORKER_WARPS = 4


def _make_kernels(ct: Any, api: Any) -> Any:
    my_pe, n_pes = api.my_pe, api.n_pes

    @ct.kernel(num_worker_warps=NUM_WORKER_WARPS)
    def store_through_peer_view(destination: ct.Array, elements: ct.Constant[int]):
        peer = api.get_peer_array(destination, (my_pe() + 1) % n_pes())
        value = ct.astype(my_pe() + 1, ct.float32)
        ct.store(peer, (0, ), ct.full((elements, ), 0.0, dtype=ct.float32) + value)

    @ct.kernel(num_worker_warps=NUM_WORKER_WARPS)
    def put_bfloat16(source: ct.Array, array_destination: ct.Array, tile_destination: ct.Array,
                     sized_destination: ct.Array, elements: ct.Constant[int]):
        peer = (my_pe() + 1) % n_pes()
        api.putmem_block(array_destination, source, peer)
        api.putmem_block(tile_destination, ct.load(source, (0, ), shape=(elements, )), peer)
        api.put64_block(sized_destination, source, peer)

    @ct.kernel(num_worker_warps=NUM_WORKER_WARPS)
    def exchange(send: ct.Array, receive: ct.Array):
        api.alltoall_block(api.Teams.TEAM_WORLD, receive, send)

    return store_through_peer_view, put_bfloat16, exchange


def run() -> None:
    """Store through a peer view and put bfloat16 data to the next rank."""
    from cuda.core.experimental import Device, system
    from mpi4py import MPI
    import cuda.tile as ct
    import nvshmem.core
    import nvshmem.core.device.tile as api
    import torch

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        raise RuntimeError(f"CuTile pointer-view runtime test requires exactly two MPI ranks, got {comm.Get_size()}")
    rank = comm.Get_rank()
    device = Device(rank % system.get_num_devices())
    device.set_current()

    initialized = False
    kernels: list[Any] = []
    buffers: list[Any] = []
    try:
        nvshmem.core.init(device=device, mpi_comm=comm, initializer_method="mpi")
        initialized = True
        viewed = nvshmem.core.tensor((ELEMENTS, ), dtype=torch.float32)
        source, array_destination, tile_destination, sized_destination = (nvshmem.core.tensor((ELEMENTS, ),
                                                                                              dtype=torch.bfloat16)
                                                                          for _ in range(4))
        size = comm.Get_size()
        send, receive = (nvshmem.core.tensor((size * ELEMENTS, ), dtype=torch.float32) for _ in range(2))
        buffers = [viewed, source, array_destination, tile_destination, sized_destination, send, receive]
        source[:] = float(rank + 1)
        send.copy_(torch.arange(size * ELEMENTS, dtype=torch.float32) + 1000.0 * rank)
        for destination in (viewed, array_destination, tile_destination, sized_destination, receive):
            destination[:] = -1.0
        device.sync()
        comm.Barrier()

        kernels = list(_make_kernels(ct, api))
        stream = torch.cuda.current_stream()
        ct.launch(stream, (1, 1), kernels[0], (viewed, ELEMENTS))
        ct.launch(stream, (1, 1), kernels[1],
                  (source, array_destination, tile_destination, sized_destination, ELEMENTS))
        ct.launch(stream, (1, 1), kernels[2], (send, receive))
        device.sync()
        comm.Barrier()
        device.sync()

        predecessor = float((rank - 1) % comm.Get_size() + 1)
        for name, destination in (("peer view", viewed), ("bfloat16 Array putmem", array_destination),
                                  ("bfloat16 Tile putmem", tile_destination), ("bfloat16 put64", sized_destination)):
            expected = torch.full((ELEMENTS, ), predecessor, dtype=destination.dtype)
            torch.testing.assert_close(destination.detach().cpu(),
                                       expected,
                                       msg=f"rank {rank}: {name} delivered the wrong elements")
        # Block ``pe`` of the result is block ``rank`` of PE ``pe``'s send buffer.
        expected = torch.cat([
            torch.arange(rank * ELEMENTS, (rank + 1) * ELEMENTS, dtype=torch.float32) + 1000.0 * pe
            for pe in range(size)
        ])
        torch.testing.assert_close(receive.detach().cpu(),
                                   expected,
                                   msg=f"rank {rank}: alltoall exchanged wrong blocks")
        print(f"rank {rank}: peer-view store, bfloat16 putmem and put64, and alltoall passed")
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
