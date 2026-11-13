#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""2D Jacobi iteration with a device-initiated halo exchange.

The PEs split the rows of a grid whose edges are held at 1.0. Each PE owns
ROWS rows plus a halo row above and below them. Every iteration, one CTA puts
the PE's first and last rows into its neighbors' halos as Tiles, and a tiled
kernel applies the five-point stencil. PE 0 checks the result against a
single-GPU solve.

    mpirun -np 2 python 02_jacobi_halo_exchange.py
"""

import cuda.tile as ct
import nvshmem.core
import torch

from _common import finalize, initialize, synchronize
from nvshmem.core.device.tile import barrier_all_block, my_pe, n_pes, put_block

ROWS, WIDTH = 64, 128
TILE_ROWS, TILE_COLS = 16, 64
ITERATIONS = 50
EDGE = 1.0


@ct.kernel(num_worker_warps=4)
def exchange_halos(u: ct.Array, rows: ct.Constant[int], width: ct.Constant[int]):
    pe = my_pe()
    if pe > 0:
        first_row = ct.load(u, (1, 0), shape=(1, width)).reshape((width, ))
        put_block(u.slice(axis=0, start=rows + 1, stop=rows + 2), first_row, pe - 1)
    if pe < n_pes() - 1:
        last_row = ct.load(u, (rows, 0), shape=(1, width)).reshape((width, ))
        put_block(u.slice(axis=0, start=0, stop=1), last_row, pe + 1)
    barrier_all_block()


@ct.kernel
def stencil(u: ct.Array, u_new: ct.Array, tile_rows: ct.Constant[int], tile_cols: ct.Constant[int]):
    rows = 1 + ct.bid(0) * tile_rows + ct.arange(tile_rows, dtype=ct.int32)[:, None]
    cols = ct.bid(1) * tile_cols + ct.arange(tile_cols, dtype=ct.int32)[None, :]
    up = ct.gather(u, (rows - 1, cols))
    down = ct.gather(u, (rows + 1, cols))
    left = ct.gather(u, (rows, cols - 1))
    right = ct.gather(u, (rows, cols + 1))
    centre = ct.gather(u, (rows, cols))
    interior = (cols > 0) & (cols < u.shape[1] - 1)
    ct.scatter(u_new, (rows, cols), ct.where(interior, (up + down + left + right) * 0.25, centre))


def initial_grid(rows):
    grid = torch.zeros((rows + 2, WIDTH))
    grid[:, [0, -1]] = EDGE
    return grid


def reference_solve(rows):
    u = initial_grid(rows)
    u[[0, -1], :] = EDGE
    for _ in range(ITERATIONS):
        u_new = u.clone()
        u_new[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]) * 0.25
        u = u_new
    return u[1:-1]


def main():
    comm, device = initialize()
    rank, size = comm.Get_rank(), comm.Get_size()

    grid = initial_grid(ROWS)
    if rank == 0:
        grid[0] = EDGE
    if rank == size - 1:
        grid[-1] = EDGE
    u = nvshmem.core.tensor(tuple(grid.shape), dtype=torch.float32)
    u_new = nvshmem.core.tensor(tuple(grid.shape), dtype=torch.float32)
    u.copy_(grid)
    u_new.copy_(grid)
    synchronize(comm, device)

    stream = torch.cuda.current_stream()
    tiles = (ROWS // TILE_ROWS, WIDTH // TILE_COLS)
    for _ in range(ITERATIONS):
        ct.launch(stream, (1, 1), exchange_halos, (u, ROWS, WIDTH))
        ct.launch(stream, tiles, stencil, (u, u_new, TILE_ROWS, TILE_COLS))
        # Every PE swaps in step, so u names the same symmetric buffer on every PE.
        u, u_new = u_new, u
    synchronize(comm, device)

    owned_rows = comm.gather(u[1:-1].cpu(), root=0)
    if rank == 0:
        torch.testing.assert_close(torch.cat(owned_rows), reference_solve(size * ROWS))
        print(f"{size} PEs matched the single-GPU solve after {ITERATIONS} iterations")

    finalize(device, u, u_new)


if __name__ == "__main__":
    main()
