# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information

import pytest

from cuda import cuda
from cuda.core.experimental import Device, Stream

import nvshmem


@pytest.mark.parametrize("barrier", [nvshmem.core.barrier_all, nvshmem.core.sync_all])
def test_device_barrier(barrier):
    local_rank_per_node = int(nvshmem.core.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    dev = Device(local_rank_per_node)
    dev.set_current()

    nb_stream = cuda.stream(
        flags=cuda.stream_flags.NON_BLOCKING,
    )
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    nvshmem.core.init(
        attr=nvshmem.core.Attr(
            mpi_comm=nvshmem.core.MPIComm.from_mpi4py(),
            cuda_stream=cu_stream_ref,
        )
    )

    barrier(stream=cu_stream_ref)

    nvshmem.core.finalize()
