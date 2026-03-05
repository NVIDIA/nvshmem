# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import numpy as np
import pytest

import nvshmem4py.nvshmem as nvshmem


def test_device_malloc_free():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(stream):
        ptr = nvshmem.malloc(8, stream=stream)
        nvshmem.free(ptr, stream=stream)

    kernel[1, 1, s](s)
    s.sync()


def test_device_calloc_free():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(stream):
        ptr = nvshmem.calloc(1, 8, stream=stream)
        nvshmem.free(ptr, stream=stream)

    kernel[1, 1, s](s)
    s.sync()
