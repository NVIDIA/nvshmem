# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import numpy as np
import pytest

from cuda import cuda
from cuda.core.experimental import Device, Stream

import nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_wait_until(dtype):
    local_rank_per_node = int(nvshmem.core.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    dev = Device(local_rank_per_node)
    dev.set_current()

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    nvshmem.core.init(
        attr=nvshmem.core.Attr(
            mpi_comm=nvshmem.core.MPIComm.from_mpi4py(),
            cuda_stream=cu_stream_ref,
        )
    )

    # Allocate and initialize symmetric memory
    src = np.array([0], dtype=dtype)
    dst = np.array([0], dtype=dtype)

    d_src = nvshmem.core.malloc(src.nbytes)
    d_dst = nvshmem.core.malloc(dst.nbytes)

    cuda.memcpy_htod_async(d_src, src, nb_stream)
    cuda.memcpy_htod_async(d_dst, dst, nb_stream)

    # Put value 1 to PE 0
    if nvshmem.core.my_pe() == 1:
        one = np.array([1], dtype=dtype)
        d_one = nvshmem.core.malloc(one.nbytes)
        cuda.memcpy_htod_async(d_one, one, nb_stream)
        nvshmem.core.put(d_dst, d_one, 1, 0, stream=cu_stream_ref)
        nvshmem.core.free(d_one)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    # Wait until dst becomes 1 on PE 0
    if nvshmem.core.my_pe() == 0:
        nvshmem.core.wait_until(d_dst, 1, stream=cu_stream_ref)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    cuda.memcpy_dtoh_async(dst, d_dst, nb_stream)
    nb_stream.synchronize()

    # Verify result
    assert dst[0] == 1

    nvshmem.core.free(d_src)
    nvshmem.core.free(d_dst)

    nvshmem.core.finalize()
