# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write to a peer's symmetric array from a Numba kernel on peer-accessible GPUs."""

from cuda.core import Device, system
from mpi4py import MPI
from numba import cuda

import nvshmem.core as nvshmem


@cuda.jit
def write_value(array, value):
    array[0] = value


def main() -> None:
    # Create a node-local MPI communicator
    world = MPI.COMM_WORLD
    local = world.Split_type(MPI.COMM_TYPE_SHARED)
    initialized = False
    array = None
    peer_array = None
    try:
        # Validate that CUDA devices are available
        device_count = system.get_num_devices()
        if device_count == 0:
            raise RuntimeError("No CUDA devices are available")

        # Assign one GPU to each process based on its node-local rank
        device = Device(local.Get_rank() % device_count)
        device.set_current()
        stream = device.create_stream()

        # Initialize NVSHMEM using MPI
        nvshmem.init(device=device, mpi_comm=world, initializer_method="mpi")
        initialized = True

        # Allocate one integer into symmetric memory, returning a CuPy ArrayView
        array = nvshmem.array((1, ), dtype="int32")

        # Form a ring in which each PE writes to its right-hand neighbor
        my_pe = nvshmem.my_pe()
        n_pes = nvshmem.n_pes()
        peer = (my_pe + 1) % n_pes

        array[0] = -1
        # Peer arrays require direct GPU peer access between the participating PEs, otherwise an
        # Exception is raised
        peer_array = nvshmem.get_peer_array(array, peer)
        write_value[1, 1, stream, 0](peer_array, my_pe)

        # Wait for every PE's write before validating the local symmetric array
        nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=stream)
        device.sync()

        expected = (my_pe - 1 + n_pes) % n_pes
        received = int(array[0])
        if received != expected:
            raise RuntimeError(f"PE {my_pe}: received {received}, expected {expected}")
        print(f"PE {my_pe}: received {received} from PE {expected}", flush=True)
    finally:
        # Release array views and symmetric memory before finalizing NVSHMEM
        peer_array = None
        try:
            if initialized and array is not None:
                nvshmem.free_array(array)
        finally:
            array = None
            try:
                if initialized:
                    nvshmem.finalize()
            finally:
                local.Free()


if __name__ == "__main__":
    main()
