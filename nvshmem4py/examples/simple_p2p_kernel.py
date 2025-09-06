# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

"""
This file shows a minimal example of using NVSHMEM4Py to run a collective operation on CuPy arrays
"""

import cupy
import nvshmem.core
from cuda.core.experimental import Device, system
from numba import cuda

@cuda.jit
def simple_shift(arr, dst_pe):
    arr[0] = dst_pe

# Initialize NVSHMEM Using an MPI communicator
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()
stream = dev.create_stream()
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

# Helper function to return a CuPy ArrayView backed by NVSHMEM symmetric memory
array = nvshmem.core.array((1,), dtype="int32")

my_pe = nvshmem.core.my_pe()
# A unidirectional ring - always get the neighbor to the right
dst_pe = (my_pe + 1) % nvshmem.core.n_pes()

# This function returns an Array which can be directly load/store'd to over NVLink
# The dst_PE must be in the same NVL domain as the PE calling this function, otherwise it will raise an Exception
dev_dst = nvshmem.core.get_peer_array(b, dst_pe)


block = 1
grid = (size + block - 1) // block
simple_shift[block, grid, 0, 0](array, my_pe)
nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream)
# This should print the neighbor's PE
print(f"From PE {my_pe}, array contains {array}")

nvshmem.core.free_array(arr_src)
nvshmem.core.free_array(arr_dst)
nvshmem.core.finalize()
