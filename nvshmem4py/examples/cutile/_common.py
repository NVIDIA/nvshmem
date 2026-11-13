# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Setup and teardown shared by the cuTile NVSHMEM examples."""

from cuda.core import Device, system
from mpi4py import MPI

import nvshmem.core


def initialize():
    """Give each MPI rank a GPU and initialize NVSHMEM on it."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        raise SystemExit("Run the cuTile NVSHMEM examples with at least two MPI ranks.")
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    device = Device(local.Get_rank() % system.get_num_devices())
    local.Free()
    device.set_current()
    nvshmem.core.init(device=device, mpi_comm=comm, initializer_method="mpi")
    return comm, device


def synchronize(comm, device):
    """Wait for this PE's GPU work, then for every PE to get here."""
    device.sync()
    comm.Barrier()


def finalize(device, *tensors):
    """Free the symmetric tensors and finalize NVSHMEM."""
    device.sync()
    for tensor in tensors:
        nvshmem.core.free_tensor(tensor)
    nvshmem.core.finalize()
