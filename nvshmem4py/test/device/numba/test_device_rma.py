# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information

import numpy as np
import pytest

import cuda.core as cc
import numba.cuda as cuda

import nvshmem.core


@cuda.jit
def test_put_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.put(dest, source, 1, pe)


@cuda.jit
def test_get_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.get(dest, source, 1, pe)


@cuda.jit
def test_p_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.p(dest, source[0], pe)


@cuda.jit
def test_g_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.g(dest, pe)


@pytest.mark.parametrize("kernel", [test_put_kernel, test_get_kernel, test_p_kernel, test_g_kernel])
def test_device_rma(kernel):
    nvshmem.core.init()

    mype = nvshmem.core.my_pe()
    npes = nvshmem.core.n_pes()

    dev = cc.Device()
    cu_stream = dev.create_stream()

    source = np.array([1], dtype=np.uint64)
    dest = nvshmem.core.malloc(source.nbytes)

    if mype == 0:
        kernel[1, 1, cu_stream](dest, source, 1)
    else:
        kernel[1, 1, cu_stream](dest, source, 0)

    cu_stream.sync()

    nvshmem.core.barrier_all(stream=cu_stream)

    nvshmem.core.free(dest)
    nvshmem.core.finalize()
