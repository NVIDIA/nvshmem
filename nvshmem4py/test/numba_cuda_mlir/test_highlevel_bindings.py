# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cffi
import argparse

from cuda.core import Device

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import Array, float32, int32
from numba_cuda_mlir.extending import overload, typing_registry

import nvshmem
from nvshmem.bindings import barrier_all
from nvshmem.device.bindings.numba_cuda_mlir import my_pe, n_pes, int_p, float_p
from utils import uid_init, mpi_init


def test_highlevel_bindings(dev: Device):

    ffi = cffi.FFI()

    def p():
        pass

    @overload(p, inline="always", typing_registry=typing_registry)
    def p_ol(arr, mype, peer):
        if arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):

            def impl(arr, mype, peer):
                ptr = ffi.from_buffer(arr)
                int_p(ptr, mype, peer)

            return impl
        elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):

            def impl(arr, mype, peer):
                ptr = ffi.from_buffer(arr)
                float_p(ptr, float32(mype), peer)

            return impl

    @cuda.jit(lto=True)
    def app_kernel(dest):
        mype = my_pe()
        npes = n_pes()
        peer = int32((mype + 1) % npes)

        p(dest, mype, peer)

    dest = nvshmem.core.array((1,), dtype="float32")

    app_kernel[1, 1, 0](dest)

    dev.sync()
    barrier_all()

    nvshmem.core.free_array(dest)
    nvshmem.core.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-type",
        "-i",
        type=str,
        help="Init type to use",
        choices=["mpi", "uid"],
        default="uid",
    )
    args = parser.parse_args()
    if args.init_type == "uid":
        dev = uid_init()
    elif args.init_type == "mpi":
        dev = mpi_init()

    test_highlevel_bindings(dev)
