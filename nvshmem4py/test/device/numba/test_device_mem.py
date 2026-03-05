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
def peer_fetch_kernel(arr, peer_pe):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = nvshmem.core.device.numba.get(arr, i, peer_pe)


@cuda.jit
def multicast_fetch_kernel(team, arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = nvshmem.core.device.numba.get(team, arr, i)


def test_peer_get(init_type):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    arr = nvshmem.core.device_array((128,), dtype=np.int32)
    arr[:] = nvshmem.core.my_pe()

    peer_fetch_kernel[1, 128, nb_stream](arr, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()


def test_multicast_get(init_type, teams):
    nvshmem.core.init(init_type)

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    arr = nvshmem.core.device_array((128,), dtype=np.int32)
    arr[:] = nvshmem.core.my_pe()

    multicast_fetch_kernel[1, 128, nb_stream](nvshmem.core.Teams.TEAM_WORLD, arr)
    nvshmem.core.barrier_all(stream=cu_stream_ref)

    nvshmem.core.finalize()
