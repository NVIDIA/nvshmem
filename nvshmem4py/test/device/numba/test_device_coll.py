# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import numpy as np
import pytest

from cuda import cuda
from cuda.core.experimental import Device, Stream

import nvshmem


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_device_broadcast(dtype):
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

    # Allocate and initialize source and destination buffers
    src = np.arange(10, dtype=dtype)
    dst = np.zeros_like(src)

    d_src = nvshmem.core.malloc(src.nbytes)
    d_dst = nvshmem.core.malloc(dst.nbytes)

    cuda.memcpy_htod_async(d_src, src, nb_stream)

    # Broadcast from PE 0
    nvshmem.core.broadcast(d_dst, d_src, 10, 0, stream=cu_stream_ref)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    cuda.memcpy_dtoh_async(dst, d_dst, nb_stream)
    nb_stream.synchronize()

    # Verify result
    assert np.array_equal(dst, src)

    nvshmem.core.free(d_src)
    nvshmem.core.free(d_dst)

    nvshmem.core.finalize()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_device_allreduce(dtype):
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

    # Allocate and initialize source and destination buffers
    src = np.arange(10, dtype=dtype)
    dst = np.zeros_like(src)

    d_src = nvshmem.core.malloc(src.nbytes)
    d_dst = nvshmem.core.malloc(dst.nbytes)

    cuda.memcpy_htod_async(d_src, src, nb_stream)

    # Allreduce sum
    nvshmem.core.allreduce_sum(d_dst, d_src, 10, stream=cu_stream_ref)

    nvshmem.core.barrier_all(stream=cu_stream_ref)

    cuda.memcpy_dtoh_async(dst, d_dst, nb_stream)
    nb_stream.synchronize()

    # Verify result
    npes = nvshmem.core.n_pes()
    assert np.array_equal(dst, src * npes)

    nvshmem.core.free(d_src)
    nvshmem.core.free(d_dst)

    nvshmem.core.finalize()
