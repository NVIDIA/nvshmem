#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

from cuda.core.experimental import Stream
from numba import cuda

import nvshmem.core


def test_barrier(init_type, teams):
    nvshmem.core.init(init_type)

    def func(teams):
        nvshmem.core.device.numba.barrier(teams)

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    test_barrier_kernel = cuda.jit(func)
    test_barrier_kernel[1, 1, nb_stream](teams)

    nvshmem.core.barrier_all(stream=cu_stream_ref)
    nvshmem.core.finalize()


def test_barrier_all(init_type):
    nvshmem.core.init(init_type)

    def func():
        nvshmem.core.device.numba.barrier_all()

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    test_barrier_all_kernel = cuda.jit(func)
    test_barrier_all_kernel[1, 1, nb_stream]()

    nvshmem.core.barrier_all(stream=cu_stream_ref)
    nvshmem.core.finalize()
