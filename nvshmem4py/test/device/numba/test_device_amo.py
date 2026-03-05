#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

import numpy as np
from cuda.core.experimental import Stream
from numba import cuda

import nvshmem.core


def test_amo_add(init_type):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    # Launch kernel to add 5 atomically
    buf = nvshmem.core.device_array((1,), dtype=np.int32)
    buf[:] = 0

    @cuda.jit
    def kernel(x):
        nvshmem.core.device.numba.atomic_add(x, 0, 5, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())

    kernel[1, 1, nb_stream](buf)
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()


def test_amo_cas(init_type):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    buf = nvshmem.core.device_array((1,), dtype=np.int32)
    buf[:] = 0

    @cuda.jit
    def kernel(x):
        nvshmem.core.device.numba.atomic_compare_swap(
            x, 0, 0, 1, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
        )

    kernel[1, 1, nb_stream](buf)
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()
