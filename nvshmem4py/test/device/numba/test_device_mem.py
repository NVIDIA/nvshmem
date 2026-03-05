# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information

import numpy as np
import pytest

import cuda.core as cc
import numba.cuda as cuda

import nvshmem.core


@cuda.jit
def test_malloc_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.put(dest, source, 1, pe)


@cuda.jit
def test_free_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.put(dest, source, 1, pe)


@pytest.mark.parametrize("kernel", [test_malloc_kernel, test_free_kernel])
def test_device_mem(kernel):
    nvshmem.core.init()

    mype = nvshmem.core.my_pe()

    dev = cc.Device()
    cu_stream = dev.create_stream()

    source = np.array([1], dtype=np.uint64)
    dest = nvshmem.core.malloc(source.nbytes)

    kernel[1, 1, cu_stream](dest, source, mype)
    cu_stream.sync()

    nvshmem.core.barrier_all(stream=cu_stream)

    nvshmem.core.free(dest)
    nvshmem.core.finalize()
