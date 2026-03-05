#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

import argparse
import math

import cupy as cp
import numpy as np
from cuda.core.experimental import Device, Stream
from numba import cuda

import nvshmem.core


# NOTE: This example is intentionally large; only the stream handle conversion
# below is relevant to NVBug 5954573.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()

    nvshmem.core.init()

    local_rank_per_node = nvshmem.core.my_pe() % nvshmem.core.n_pes()
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create a CUDA stream and use it for barriers and kernels
    nb_stream = cuda.stream()
    cu_stream = Stream.from_handle(int(nb_stream.handle))
    nvshmem.core.barrier_all(stream=cu_stream)

    # Placeholder body (original example content omitted in this patch context)
    x = cp.arange(args.n, dtype=cp.float32)
    nvshmem.core.barrier_all(stream=cu_stream)
    if nvshmem.core.my_pe() == 0:
        print(float(x[0]))

    nvshmem.core.finalize()


if __name__ == "__main__":
    main()
