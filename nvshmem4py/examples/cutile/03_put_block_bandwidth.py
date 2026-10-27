#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bandwidth of put_block from an Array and from a Tile, then a sum-reduce.

Each CTA puts its share of a buffer to the next PE in CHUNK-element pieces,
either straight from the Array or after loading each piece as a Tile, and
completes the puts with quiet. Every timed run also checks the delivered data.
A sum_reduce_block over all PEs follows.

    mpirun -np 2 python 03_put_block_bandwidth.py --sizes 16k,256k --ctas 1,4
"""

import argparse

import cuda.tile as ct
import nvshmem.core
import torch

from _common import finalize, initialize, synchronize
from nvshmem.core.device.tile import Teams, my_pe, n_pes, put_block, quiet, sum_reduce_block

CHUNK = 4096
ELEMENT_BYTES = 4
REDUCE_ELEMENTS = 1024


@ct.kernel(num_worker_warps=4)
def put_from_array(src: ct.Array, dst: ct.Array, chunks: ct.Constant[int], chunk: ct.Constant[int]):
    next_pe = (my_pe() + 1) % n_pes()
    for i in range(chunks):
        start = (ct.bid(0) * chunks + i) * chunk
        put_block(dst.slice(axis=0, start=start, stop=start + chunk), src.slice(axis=0, start=start,
                                                                                stop=start + chunk), next_pe)
    quiet()


@ct.kernel(num_worker_warps=4)
def put_from_tile(src: ct.Array, dst: ct.Array, chunks: ct.Constant[int], chunk: ct.Constant[int]):
    next_pe = (my_pe() + 1) % n_pes()
    for i in range(chunks):
        index = ct.bid(0) * chunks + i
        tile = ct.load(src, (index, ), shape=(chunk, ))
        put_block(dst.slice(axis=0, start=index * chunk, stop=(index + 1) * chunk), tile, next_pe)
    quiet()


@ct.kernel(num_worker_warps=4)
def sum_reduce(src: ct.Array, dst: ct.Array):
    sum_reduce_block(Teams.TEAM_WORLD, dst, src)


PUT_KERNELS = {"array": put_from_array, "tile": put_from_tile}


def parse_bytes(text):
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    unit = units.get(text[-1].lower())
    return int(text[:-1]) * unit if unit else int(text)


def time_puts(comm, device, kernel, elements, ctas, iterations, warmup):
    """Return the mean time of one launch in milliseconds, after checking the data."""
    rank, size = comm.Get_rank(), comm.Get_size()
    chunk = min(elements, CHUNK)
    src = nvshmem.core.tensor((elements * ctas, ), dtype=torch.float32)
    dst = nvshmem.core.tensor((elements * ctas, ), dtype=torch.float32)
    src.fill_(rank + 1)
    dst.fill_(-1)
    args = (src, dst, elements // chunk, chunk)
    stream = torch.cuda.current_stream()
    synchronize(comm, device)

    for _ in range(warmup):
        ct.launch(stream, (ctas, 1), kernel, args)
    synchronize(comm, device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iterations):
        ct.launch(stream, (ctas, 1), kernel, args)
    end.record(stream)
    synchronize(comm, device)

    torch.testing.assert_close(dst, torch.full_like(dst, (rank - 1) % size + 1))
    nvshmem.core.free_tensor(src)
    nvshmem.core.free_tensor(dst)
    return start.elapsed_time(end) / iterations


def check_sum_reduce(comm, device, kernel):
    rank, size = comm.Get_rank(), comm.Get_size()
    src = nvshmem.core.tensor((REDUCE_ELEMENTS, ), dtype=torch.float32)
    dst = nvshmem.core.tensor((REDUCE_ELEMENTS, ), dtype=torch.float32)
    src.fill_(rank + 1)
    dst.fill_(-1)
    synchronize(comm, device)

    ct.launch(torch.cuda.current_stream(), (1, 1), kernel, (src, dst))
    synchronize(comm, device)

    torch.testing.assert_close(dst, torch.full_like(dst, size * (size + 1) // 2))
    nvshmem.core.free_tensor(src)
    nvshmem.core.free_tensor(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="16k,256k,4m", help="bytes each CTA puts, comma separated")
    parser.add_argument("--ctas", default="1,4", help="CTA counts, comma separated")
    parser.add_argument("--sources", default="array,tile", help="put sources: array, tile, or both")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    sizes = [parse_bytes(text) // ELEMENT_BYTES for text in args.sizes.split(",")]
    ctas = [int(text) for text in args.ctas.split(",")]
    for elements in sizes:
        chunk = min(elements, CHUNK)
        if elements % chunk or chunk & (chunk - 1):
            parser.error(f"{elements * ELEMENT_BYTES} bytes per CTA must be a power of two up to "
                         f"{CHUNK * ELEMENT_BYTES} or a multiple of {CHUNK * ELEMENT_BYTES}")

    comm, device = initialize()
    for source in args.sources.split(","):
        for count in ctas:
            for elements in sizes:
                ms = time_puts(comm, device, PUT_KERNELS[source], elements, count, args.iterations, args.warmup)
                if comm.Get_rank() == 0:
                    total_bytes = elements * count * ELEMENT_BYTES
                    print(f"put_block {source:5s} ctas={count:3d} bytes/CTA={elements * ELEMENT_BYTES:9d} "
                          f"{ms * 1e3:9.2f} us {total_bytes / ms / 1e6:8.2f} GB/s")
    check_sum_reduce(comm, device, sum_reduce)
    if comm.Get_rank() == 0:
        print("sum_reduce_block matched the expected sum")

    finalize(device)


if __name__ == "__main__":
    main()
