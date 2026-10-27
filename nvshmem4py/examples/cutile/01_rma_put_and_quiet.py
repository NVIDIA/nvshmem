#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Put a block to the next PE, blocking and non-blocking, and complete both with quiet.

    mpirun -np 2 python 01_rma_put_and_quiet.py
"""

import cuda.tile as ct
import nvshmem.core
import torch

from _common import finalize, initialize, synchronize
from nvshmem.core.device.tile import my_pe, n_pes, put_block, put_nbi_block, quiet

ELEMENTS = 32


@ct.kernel(num_worker_warps=4)
def ring_put(src: ct.Array, dst: ct.Array, nbi_dst: ct.Array):
    next_pe = (my_pe() + 1) % n_pes()
    put_block(dst, src, next_pe)
    put_nbi_block(nbi_dst, src, next_pe)
    quiet()


def main():
    comm, device = initialize()
    rank, size = comm.Get_rank(), comm.Get_size()

    src, dst, nbi_dst = (nvshmem.core.tensor((ELEMENTS, ), dtype=torch.float32) for _ in range(3))
    src.fill_(rank + 1)
    dst.fill_(-1)
    nbi_dst.fill_(-1)
    synchronize(comm, device)

    ct.launch(torch.cuda.current_stream(), (1, 1), ring_put, (src, dst, nbi_dst))
    synchronize(comm, device)

    previous_pe = (rank - 1) % size
    expected = torch.full((ELEMENTS, ), previous_pe + 1, dtype=torch.float32)
    torch.testing.assert_close(dst.cpu(), expected)
    torch.testing.assert_close(nbi_dst.cpu(), expected)
    print(f"PE {rank}: received both puts from PE {previous_pe}")

    finalize(device, src, dst, nbi_dst)


if __name__ == "__main__":
    main()
