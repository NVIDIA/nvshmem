# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import numpy as np
import pytest

import nvshmem4py.nvshmem as nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_broadcast(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, root, stream):
        nvshmem.broadcast(dst, src, root, nvshmem.TEAM_WORLD, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = nvshmem.array((1,), dtype=dtype)

    kernel[1, 1, s](dst, src, 0, s)
    s.sync()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_alltoall(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, stream):
        nvshmem.alltoall(dst, src, nvshmem.TEAM_WORLD, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = nvshmem.array((1,), dtype=dtype)

    kernel[1, 1, s](dst, src, s)
    s.sync()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_allreduce(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, stream):
        nvshmem.sum_reduce(dst, src, nvshmem.TEAM_WORLD, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = nvshmem.array((1,), dtype=dtype)

    kernel[1, 1, s](dst, src, s)
    s.sync()
