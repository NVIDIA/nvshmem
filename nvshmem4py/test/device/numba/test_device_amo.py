# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# See COPYRIGHT for license information

import numpy as np
import pytest

import cuda.core as cc
import numba.cuda as cuda

import nvshmem.core
from nvshmem.core.nvshmem_types import *


@cuda.jit
def test_atomic_add_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_add(dest, source[0], pe)


@cuda.jit
def test_atomic_inc_kernel(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_add_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_add(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_inc_kernel(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_and_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_and(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_or_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_or(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_xor_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_xor(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_kernel(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch(dest, pe)


@cuda.jit
def test_atomic_set_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_set(dest, source[0], pe)


@cuda.jit
def test_atomic_swap_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_swap(dest, source[0], pe)


@cuda.jit
def test_atomic_compare_swap_kernel(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_compare_swap(dest, source[0], source[1], pe)


@cuda.jit
def test_atomic_add_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_add(dest, source[0], pe)


@cuda.jit
def test_atomic_inc_kernel_32(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_add_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch_add(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_inc_kernel_32(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_and_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch_and(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_or_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch_or(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_xor_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch_xor(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_kernel_32(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_fetch(dest, pe)


@cuda.jit
def test_atomic_set_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_set(dest, source[0], pe)


@cuda.jit
def test_atomic_swap_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_swap(dest, source[0], pe)


@cuda.jit
def test_atomic_compare_swap_kernel_32(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint32_atomic_compare_swap(dest, source[0], source[1], pe)


@cuda.jit
def test_atomic_add_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_add(dest, source[0], pe)


@cuda.jit
def test_atomic_inc_kernel_64(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_add_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_add(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_inc_kernel_64(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_inc(dest, pe)


@cuda.jit
def test_atomic_fetch_and_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_and(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_or_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_or(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_xor_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch_xor(dest, source[0], pe)


@cuda.jit
def test_atomic_fetch_kernel_64(dest, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_fetch(dest, pe)


@cuda.jit
def test_atomic_set_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_set(dest, source[0], pe)


@cuda.jit
def test_atomic_swap_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_swap(dest, source[0], pe)


@cuda.jit
def test_atomic_compare_swap_kernel_64(dest, source, pe):
    tid = cuda.threadIdx.x
    if tid == 0:
        nvshmem.core.uint64_atomic_compare_swap(dest, source[0], source[1], pe)


@pytest.mark.parametrize(
    "kernel",
    [
        test_atomic_add_kernel,
        test_atomic_inc_kernel,
        test_atomic_fetch_add_kernel,
        test_atomic_fetch_inc_kernel,
        test_atomic_fetch_and_kernel,
        test_atomic_fetch_or_kernel,
        test_atomic_fetch_xor_kernel,
        test_atomic_fetch_kernel,
        test_atomic_set_kernel,
        test_atomic_swap_kernel,
        test_atomic_compare_swap_kernel,
        test_atomic_add_kernel_32,
        test_atomic_inc_kernel_32,
        test_atomic_fetch_add_kernel_32,
        test_atomic_fetch_inc_kernel_32,
        test_atomic_fetch_and_kernel_32,
        test_atomic_fetch_or_kernel_32,
        test_atomic_fetch_xor_kernel_32,
        test_atomic_fetch_kernel_32,
        test_atomic_set_kernel_32,
        test_atomic_swap_kernel_32,
        test_atomic_compare_swap_kernel_32,
        test_atomic_add_kernel_64,
        test_atomic_inc_kernel_64,
        test_atomic_fetch_add_kernel_64,
        test_atomic_fetch_inc_kernel_64,
        test_atomic_fetch_and_kernel_64,
        test_atomic_fetch_or_kernel_64,
        test_atomic_fetch_xor_kernel_64,
        test_atomic_fetch_kernel_64,
        test_atomic_set_kernel_64,
        test_atomic_swap_kernel_64,
        test_atomic_compare_swap_kernel_64,
    ],
)
def test_device_amo(kernel):
    nvshmem.core.init()

    mype = nvshmem.core.my_pe()
    npes = nvshmem.core.n_pes()

    dev = cc.Device()
    cu_stream = dev.create_stream()

    source = np.array([1, 2], dtype=np.uint64)
    dest = nvshmem.core.malloc(source.nbytes)

    if mype == 0:
        kernel[1, 1, cu_stream](dest, source, 1)
    else:
        kernel[1, 1, cu_stream](dest, source, 0)

    cu_stream.sync()

    nvshmem.core.barrier_all(stream=cu_stream)

    nvshmem.core.free(dest)
    nvshmem.core.finalize()
