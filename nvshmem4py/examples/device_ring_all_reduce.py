#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

import numpy as np
from cuda.core.experimental import Device, Stream
from numba import cuda

import nvshmem.core


def main():
    # Initialize NVSHMEM
    nvshmem.core.init()

    # Set up device
    local_rank_per_node = nvshmem.core.my_pe() % nvshmem.core.n_pes()
    dev = Device(local_rank_per_node)
    dev.set_current()

    nb_stream = cuda.stream()  # WAR: Numba-CUDA takes numba stream object or int
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    nvshmem.core.init(
        attr=nvshmem.core.NvshmemInitAttr(
            mpi_comm=nvshmem.core.MPI.COMM_WORLD,
            cuda_stream=cu_stream_ref,
        )
    )

    # Create source and destination arrays
    src = np.ones(1024, dtype=np.float32) * (nvshmem.core.my_pe() + 1)
    dst = np.zeros_like(src)

    # Perform ring all-reduce
    for _ in range(nvshmem.core.n_pes()):
        nvshmem.core.put(dst, src, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())
        nvshmem.core.barrier_all(stream=cu_stream_ref)
        src += dst

    # Print result
    print(f"PE {nvshmem.core.my_pe()} result: {src[0]}")

    # Finalize NVSHMEM
    nvshmem.core.finalize()


if __name__ == "__main__":
    main()
