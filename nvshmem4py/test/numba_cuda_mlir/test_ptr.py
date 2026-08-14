# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import int32

import argparse

from utils import uid_init, mpi_init

import nvshmem
from nvshmem.device.bindings.numba_cuda_mlir import ptr

import cffi

ffi = cffi.FFI()


def test_ptr():

    @cuda.jit(lto=True)
    def kernel_nvshmem(arr):
        arr_ptr = ffi.from_buffer(arr)
        other_ptr_typed = ptr(arr_ptr, int32(1))
        _ = other_ptr_typed

    dest = nvshmem.core.array((1, ), dtype="int16")

    kernel_nvshmem[1, 1](dest)

    cuda.synchronize()
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
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_ptr()
