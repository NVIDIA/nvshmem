"""
Utilities for NVSHMEM4Py unit tests
"""

from mpi4py import MPI
import numpy as np

import nvshmem.core
from nvshmem.core.nvshmem_types import *

from cuda.core import Device, system
from cuda.core import Stream
from cuda.core import MemoryResource, Buffer
import cuda.bindings.driver as driver
import ctypes

import os


def get_local_rank_per_node():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split COMM_WORLD into sub-communicators of processes on the same node
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    local_rank = node_comm.Get_rank()
    local_size = node_comm.Get_size()
    print(f"Local rank {local_rank} global rank {rank} and node size {local_size} of global size {size} ranks")
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
    print(f"MPI initialized on device {dev.device_id}")
    return dev
