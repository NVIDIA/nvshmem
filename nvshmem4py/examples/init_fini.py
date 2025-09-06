"""
This file contains examples of initialization and finalization of NVSHMEM through various launching methods
"""
import numpy as np
import nvshmem.core
from cuda.core.experimental import Device, system
import os

from mpi4py import MPI
import torch
import torch.distributed as dist

def mpi_uid_init():
    from mpi4py import MPI
    # This will use mpi4py to perform a UID based init with bcast.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # Broadcast UID to all ranks
    comm.Bcast(uniqueid._data.view(np.int8), root=0)

    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks,
                      mpi_comm=None, initializer_method="uid")
    nvshmem.core.finalize()

def torchrun_uid_init_bcast():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher

    It uses torch.distributed.broadcast on a NumPy array to handle the broadcasting
    """
    # Set Torch device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # nvshmem4py requires a cuda.core Device at init time
    dev = Device(local_rank)
    dev.set_current()
    global stream
    stream = dev.create_stream()

    # Initialize torch.distributed process group
    world_size = torch.cuda.device_count()
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )

    # Extract rank, nranks from process group
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank_id == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # This is a NumPy array which is the same shape as a struct nvshmem_uniqueid_t
    data = torch.tensor(uniqueid._data)
    # We use torch.distributed.broadcast to send the UID to all ranks
    dist.broadcast(data, src=0)
    dist.barrier()

    if rank_id != 0:
        uniqueid._data = data.numpy()

    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank_id, nranks=num_ranks, initializer_method="uid")

def torchrun_uid_init_bcast_object():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher

    It uses torch.distributed.broadcast_object_list to broadcast the Python Uniqueid
    """
    # Set Torch device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # nvshmem4py requires a cuda.core Device at init time
    dev = Device(local_rank)
    dev.set_current()
    global stream
    stream = dev.create_stream()

    # Initialize torch.distributed process group
    world_size = torch.cuda.device_count()
    dist.init_process_group(
        backend="cpu:gloo",
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )

    # Extract rank, nranks from process group
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank_id == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    # We use torch.distributed.broadcast_object_list to send the UID to all ranks
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

def mpi_init():
    # This uses the MPI communicator to perform initialization of NVSHMEM
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None,
                      mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    nvshmem.core.finalize()


def emulated_mpi_init():
    # This uses the MPI communicator to perform initialization of NVSHMEM
    # Internally, NVSHMEM4Py performs an MPI4Py broadcast and UID init
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None,
                      mpi_comm=MPI.COMM_WORLD, initializer_method="emulated_mpi")
    nvshmem.core.finalize()

