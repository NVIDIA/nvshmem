# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import pytest

import nvshmem4py.nvshmem as nvshmem


def test_device_quiet():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(stream):
        nvshmem.quiet(stream=stream)

    kernel[1, 1, s](s)
    s.sync()


def test_device_fence():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(stream):
        nvshmem.fence(stream=stream)

    kernel[1, 1, s](s)
    s.sync()
