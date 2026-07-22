# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal NVSHMEM4Py initialization example using MPI."""

from cuda.core import Device, system
from mpi4py import MPI

import nvshmem.core as nvshmem


def main() -> None:
    # Create a node-local MPI communicator
    world = MPI.COMM_WORLD
    local = world.Split_type(MPI.COMM_TYPE_SHARED)
    initialized = False
    try:
        # Validate that CUDA devices are available
        device_count = system.get_num_devices()
        if device_count == 0:
            raise RuntimeError("No CUDA devices are available")

        # Assign one GPU to each process based on its node-local rank
        device = Device(local.Get_rank() % device_count)
        device.set_current()

        # Initialize NVSHMEM using MPI
        nvshmem.init(device=device, mpi_comm=world, initializer_method="mpi")
        initialized = True

        # Output my global PE and global number of PEs
        print(f"Hello from PE {nvshmem.my_pe()} of {nvshmem.n_pes()}", flush=True)
    finally:
        try:
            if initialized:
                nvshmem.finalize()
        finally:
            local.Free()


if __name__ == "__main__":
    main()
