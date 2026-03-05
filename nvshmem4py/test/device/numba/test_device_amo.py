# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import numpy as np
import pytest

import nvshmem4py.nvshmem as nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_amo_add(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, pe, stream):
        nvshmem.int_p(dst, src, pe, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = np.array([1], dtype=dtype)

    kernel[1, 1, s](dst, src[0], 0, s)
    s.sync()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_amo_fetch_add(dtype):
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(dst, src, pe, stream):
        nvshmem.int_atomic_fetch_add(dst, src, pe, stream=stream)

    dst = nvshmem.array((1,), dtype=dtype)
    src = np.array([1], dtype=dtype)

    kernel[1, 1, s](dst, src[0], 0, s)
    s.sync()
