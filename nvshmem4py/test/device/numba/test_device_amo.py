# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import numpy as np
import pytest

from cuda import cuda
from cuda.core.experimental import Device, Stream

import nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_amo_add(dtype):
    local_rank_per_node = int(nvshmem.core.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    dev = Device(local_rank_per_node)
    dev.set_current()

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    # Launch kernel to add 5 atomically
    src = np.array([5], dtype=dtype)
    dst = np.zeros(1, dtype=dtype)

    nvshmem.core.init(
        attr=nvshmem.core.Attr(
            mpi_comm=nvshmem.core.MPIComm.from_mpi4py(),
            cuda_stream=cu_stream_ref,
        )
    )

    d_dst = nvshmem.core.malloc(dst.nbytes)
    d_src = nvshmem.core.malloc(src.nbytes)

    cuda.memcpy_htod_async(d_src, src, nb_stream)

    # Perform atomic add on PE 0
    nvshmem.core.int_atomic_add(d_dst, d_src, 0, stream=cu_stream_ref)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    cuda.memcpy_dtoh_async(dst, d_dst, nb_stream)
    nb_stream.synchronize()

    # Verify result
    if nvshmem.core.my_pe() == 0:
        assert dst[0] == 5

    nvshmem.core.free(d_dst)
    nvshmem.core.free(d_src)

    nvshmem.core.finalize()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_device_amo_fetch_add(dtype):
    local_rank_per_node = int(nvshmem.core.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    dev = Device(local_rank_per_node)
    dev.set_current()

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    # Launch kernel to fetch-add 5 atomically
    src = np.array([5], dtype=dtype)
    dst = np.zeros(1, dtype=dtype)

    nvshmem.core.init(
        attr=nvshmem.core.Attr(
            mpi_comm=nvshmem.core.MPIComm.from_mpi4py(),
            cuda_stream=cu_stream_ref,
        )
    )

    d_dst = nvshmem.core.malloc(dst.nbytes)
    d_src = nvshmem.core.malloc(src.nbytes)

    cuda.memcpy_htod_async(d_src, src, nb_stream)

    # Perform atomic fetch-add on PE 0
    old = nvshmem.core.int_atomic_fetch_add(d_dst, d_src, 0, stream=cu_stream_ref)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    cuda.memcpy_dtoh_async(dst, d_dst, nb_stream)
    nb_stream.synchronize()

    # Verify result
    if nvshmem.core.my_pe() == 0:
        assert old == 0
        assert dst[0] == 5

    nvshmem.core.free(d_dst)
    nvshmem.core.free(d_src)

    nvshmem.core.finalize()
