#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

import numpy as np
from cuda.core.experimental import Stream
from numba import cuda

import nvshmem.core


@cuda.jit
def test_put(buf_dst, buf_src, peer_pe):
    nvshmem.core.device.numba.put(buf_dst, buf_src, peer_pe)


@cuda.jit
def test_get(buf_dst, buf_src, peer_pe):
    nvshmem.core.device.numba.get(buf_dst, buf_src, peer_pe)


def test_device_put(init_type):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    buf_src = nvshmem.core.device_array((16,), dtype=np.int32)
    buf_dst = nvshmem.core.device_array((16,), dtype=np.int32)

    buf_src[:] = nvshmem.core.my_pe() + 1

    test_put[1, 16, nb_stream](buf_dst, buf_src, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()


def test_device_get(init_type):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    buf_src = nvshmem.core.device_array((16,), dtype=np.int32)
    buf_dst = nvshmem.core.device_array((16,), dtype=np.int32)

    buf_src[:] = nvshmem.core.my_pe() + 1

    test_get[1, 16, nb_stream](buf_dst, buf_src, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()
