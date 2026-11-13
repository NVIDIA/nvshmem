#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""All-gather of a sharded input fused with a matrix multiply.

Each PE owns an M x K shard of the input. One CTA pushes the shard into slot
``rank`` of every peer's ``gathered`` buffer with put_signal_nbi_block, which
sets the peer's ``arrived[rank]`` flag once the data has landed. Each output
tile of ``gathered @ weight`` waits only for the shard it reads, so no quiet is
needed, and tiles of shards that have arrived proceed while others are in
flight. Every PE checks its result against torch.

    mpirun -np 2 python 04_allgather_matmul.py
"""

import cuda.tile as ct
import nvshmem.core
import torch

from _common import finalize, initialize, synchronize
from nvshmem.core.device.tile import ComparisonType, SignalOp, my_pe, put_signal_nbi_block, signal_wait_until

M, K, N = 256, 256, 128
TILE_M, TILE_N, TILE_K = 64, 64, 64


@ct.kernel(num_worker_warps=4)
def push_shard(gathered: ct.Array, arrived: ct.Array, pes: ct.Constant[int]):
    pe = my_pe()
    shard = gathered.slice(axis=0, start=pe, stop=pe + 1)
    flag = arrived.slice(axis=0, start=pe, stop=pe + 1)
    for shift in range(1, pes):
        put_signal_nbi_block(shard, shard, flag, 1, SignalOp.SIGNAL_SET, (pe + shift) % pes)


@ct.kernel(num_worker_warps=4)
def gathered_matmul(gathered: ct.Array, weight: ct.Array, out: ct.Array, arrived: ct.Array, tile_m: ct.Constant[int],
                    tile_n: ct.Constant[int], tile_k: ct.Constant[int]):
    tiles_per_shard = gathered.shape[1] // tile_m
    shard = ct.bid(0) // tiles_per_shard
    row = ct.bid(0) % tiles_per_shard
    signal_wait_until(arrived.slice(axis=0, start=shard, stop=shard + 1), ComparisonType.CMP_EQ, 1)
    accumulator = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
    for k in range(ct.num_tiles(weight, axis=0, shape=(tile_k, tile_n))):
        a = ct.load(gathered, (shard, row, k), shape=(1, tile_m, tile_k)).reshape((tile_m, tile_k))
        b = ct.load(weight, (k, ct.bid(1)), shape=(tile_k, tile_n))
        accumulator = ct.mma(a, b, accumulator)
    ct.store(out, (ct.bid(0), ct.bid(1)), accumulator)


def main():
    comm, device = initialize()
    rank, size = comm.Get_rank(), comm.Get_size()
    shard = torch.rand((M, K), generator=torch.Generator().manual_seed(rank)).half()
    weight = torch.rand((K, N), generator=torch.Generator().manual_seed(size)).half()

    gathered = nvshmem.core.tensor((size, M, K), dtype=torch.float16)
    arrived = nvshmem.core.tensor((size, ), dtype=torch.int64)
    gathered.zero_()
    gathered[rank].copy_(shard)
    arrived.zero_()
    arrived[rank] = 1
    weight = weight.to(gathered.device)
    out = torch.empty((size * M, N), dtype=torch.float32, device=gathered.device)
    synchronize(comm, device)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1), push_shard, (gathered, arrived, size))
    ct.launch(stream, (size * M // TILE_M, N // TILE_N), gathered_matmul,
              (gathered, weight, out, arrived, TILE_M, TILE_N, TILE_K))
    synchronize(comm, device)

    expected = torch.cat(comm.allgather(shard)).float() @ weight.cpu().float()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-3)
    print(f"PE {rank}: gathered {size} shards and matched torch")

    finalize(device, gathered, arrived)


if __name__ == "__main__":
    main()
