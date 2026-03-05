# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import numpy as np
import pytest

import nvshmem4py.nvshmem as nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_put(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, pe, stream):
        nvshmem.put(dst, src, pe, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = np.array([1], dtype=dtype)

    kernel[1, 1, s](dst, src, 0, s)
    s.sync()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_get(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, pe, stream):
        nvshmem.get(dst, src, pe, stream=stream)

    dst = np.array([0], dtype=dtype)
    src = nvshmem.array((1,), dtype=dtype)

    kernel[1, 1, s](dst, src, 0, s)
    s.sync()
