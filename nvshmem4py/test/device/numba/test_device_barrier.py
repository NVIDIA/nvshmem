# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information

import numpy as np
import pytest

import cuda.core as cc
import numba.cuda as cuda

import nvshmem.core


@cuda.jit
def test_barrier_all_kernel():
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.barrier_all()


@cuda.jit
def test_sync_all_kernel():
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.sync_all()


@pytest.mark.parametrize("kernel", [test_barrier_all_kernel, test_sync_all_kernel])
def test_device_barrier(kernel):
    nvshmem.core.init()

    dev = cc.Device()
    cu_stream = dev.create_stream()

    kernel[1, 1, cu_stream]()
    cu_stream.sync()

    nvshmem.core.barrier_all(stream=cu_stream)

    nvshmem.core.finalize()
