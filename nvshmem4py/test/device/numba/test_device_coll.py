#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

import numpy as np
from numba import cuda

import nvshmem.core


class NumbaStreamWrapper:
    def __init__(self, nb_stream):
        self._nb_stream = nb_stream

    def __cuda_stream__(self):
        return (0, int(self._nb_stream.handle))


def test_reduce(init_type, teams):
    nvshmem.core.init(init_type)

    op = nvshmem.core.ReduceOp.SUM

    @cuda.jit
    def kernel(team, dest, src):
        nvshmem.core.device.numba.reduce(team, dest, src, op)

    nb_stream = cuda.stream()
    stream_wrapper = NumbaStreamWrapper(nb_stream)

    src = nvshmem.core.device_array((16,), dtype=np.int32)
    dest = nvshmem.core.device_array((16,), dtype=np.int32)

    src[:] = nvshmem.core.my_pe() + 1

    kernel[1, 16, stream_wrapper](teams, dest, src)

    nvshmem.core.barrier_all()
    nvshmem.core.finalize()
