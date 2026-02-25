"""
Utilities for NVSHMEM4Py unit tests

TODO: this is vended from top level utils.py, let's find a way to centralize these functions
"""

from mpi4py import MPI
import numpy as np

import nvshmem.core

from cuda.core import Device, system

import os


def get_local_rank_per_node():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split COMM_WORLD into sub-communicators of processes on the same node
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    local_rank = node_comm.Get_rank()
    local_size = node_comm.Get_size()

    return local_rank


def uid_init():
    # This will use mpi4py to perform a UID based init with bcast.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    local_rank_per_node = get_local_rank_per_node()
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # Broadcast UID to all ranks
    comm.Bcast(uniqueid._data.view(np.int8), root=0)

    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks, mpi_comm=None, initializer_method="uid")

    return dev


def mpi_init():
    local_rank_per_node = get_local_rank_per_node()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    return dev
