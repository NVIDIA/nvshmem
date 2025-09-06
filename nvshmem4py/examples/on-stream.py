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
This file implements `examples/on-stream.cu` in Python
"""
import cupy
from numba import cuda
from cuda.core.experimental import Device, system

import nvshmem.core

from mpi4py import MPI

THRESHOLD = 42
CORRECTION = 7

@cuda.jit
def accumulate(input, partial_sum):
	"""
	Accumulate kernel: Input is a 1-d array and partial_sum is a 1x1 array
	"""
	index = cuda.threadIdx.x
	if index == 0:
		partial_sum[0] = 0
	cuda.syncthreads()
	numba.cuda.atomic.add(partial_sum, 0, input[index])

@cuda.jit
def correct_accumulate(input, partial_sum, full_sum):
	index = cuda.threadIdx.x
	if (full_sum > THRESHOLD):
		input[index] = input[index] - CORRECTION
	if index == 0:
		partial_sum[0] = 0
	cuda.syncthreads()
	numba.cuda.atomic.add(partial_sum, 0, input[index])

# Initialize NVSHMEM Using an MPI communicator
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

mype = nvshmem.core.my_pe()
npes = nvshmem.core.n_pes()
mype_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)

input_nelems = 512
to_all_elems = 1
stream = dev.create_stream()
input = nvshmem.core.array((input_nelems,), dtype="int")
partial_sum = nvshmem.core.array((1,), dtype="int")
full_sum = nvshmem.core.array((1,), dtype="int")

accumulate[1, input_nelems, 0, stream](input, partial_sum)
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, full_sum, partial_sum, "sum", stream=stream)

correct_accumulate[1, input_nelems, 0, stream](input, partial_sum, full_sum)
stream.sync()

print(f"[{mype} of {npes}] Run complete")

nvshmem.core.free_array(input)
nvshmem.core.free_array(partial_sum)
nvshmem.core.free_array(full_sum)
nvshmem.core.finalize()
