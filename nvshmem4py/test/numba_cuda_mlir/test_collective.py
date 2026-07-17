# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cffi
import argparse

from cuda.core import Device

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import int32

import nvshmem
from nvshmem.bindings import my_pe, n_pes
from nvshmem.device.bindings.numba_cuda_mlir import int_p, barrier_all

from utils import uid_init, mpi_init


def test_collective(dev: Device):

    ffi = cffi.FFI()

    @cuda.jit(lto=True)
    def reduce_ring(dest, mype, npes):
        target = ffi.from_buffer(dest)
        peer = int32((mype + 1) % npes)
        lvalue = int32(mype)

        for _ in range(npes):
            int_p(target, lvalue, peer)
            barrier_all()
            lvalue = int32(target[0] + mype)
            barrier_all()

    mype = my_pe()
    npes = n_pes()

    dest = nvshmem.core.array((1, ), dtype="int32")

    reduce_ring[1, 1, 0](dest, mype, npes)

    dev.sync()

    print(f"{my_pe()}: received message {dest[0]}")

    nvshmem.core.free_array(dest)
    nvshmem.core.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        dev = uid_init()
    elif args.init_type == "mpi":
        dev = mpi_init()

    test_collective(dev)
