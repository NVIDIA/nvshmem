# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import cuda.core as cc
import numba.cuda as cuda
import pytest

import nvshmem4py.nvshmem as nvshmem


def test_device_barrier_all():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(stream):
        nvshmem.barrier_all(stream=stream)

    kernel[1, 1, s](s)
    s.sync()


def test_device_barrier():
    dev = cc.Device()
    s = dev.default_stream

    @cuda.jit
    def kernel(team, stream):
        nvshmem.team_barrier(team, stream=stream)

    kernel[1, 1, s](nvshmem.TEAM_WORLD, s)
    s.sync()
