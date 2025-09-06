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
This file demonstrates a custom communication kernel using Triton to generate CUDA programs.
The kernel is designed to perform a broadcast operation using NVSHMEM4Py-allocated tensors.

Each participating process (PE) uses NVSHMEM to allocate a shared tensor. The root process
loads values from its local tensor and performs remote multicast stores directly to all peers.
This example illustrates low-level device-side communication using `tl.load` and `tl.store`.
This will work only when all PEs are launched within a single NVLink domain

NOTE: We're using `mc_ptr()` to obtain a multicast-compatible remote pointer, allowing a
single set of stores to broadcast data to all PEs simultaneously via NVLink or SHMEM backend.
"""

import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem
from nvshmem.bindings import mc_ptr  # Gets a device-accessible remote pointer
from mpi4py import MPI
from cuda.core.experimental import Device, system

"""
The functions load_v4_u32 and multimem_st_b64 are adapted from the Triton-Distributed project to show a practical example of custom communication kernels on NVSHMEM symmetric heap memory.
Source: https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/low_latency_allgather.py
"""

@triton.jit
def load_v4_u32(ptr):
    """
        Perform a vectorized load of 4x4B integers
    """
    return tl.inline_asm_elementwise(
        asm="""
        ld.volatile.global.v4.u32 {$0,$1,$2,$3}, [$4];
        """,
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
        )

@triton.jit
def multimem_st_b64(ptr, val0):
    """
        Perform a multicast store of 1x8B integer
    """
    return tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,l"),
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )

@triton.jit
def broadcast_naive_block(src_ptr, nbytes, rank, root_rank, remote_mc_ptr):
    """
    Triton device kernel to perform a naive broadcast from root PE using tl.load and tl.store.

    Only the root PE executes this kernel; all others passively receive data through
    remote multicast writes (via remote_mc_ptr).

    Parameters:
        src_ptr (float32*): Local pointer to the source buffer (only valid on root PE).
        nbytes (int): Total number of bytes to copy.
        rank (int): Current PE's rank.
        root_rank (int): Rank of the broadcasting root PE.
        remote_mc_ptr (float32*): Multicast pointer used to write to all PEs.

    Notes:
        Triton indexing is in elements, not bytes. We perform two 32-bit loads and stores per iteration.
    """
    remote_mc_ptr = tl.cast(remote_mc_ptr, tl.pointer_type(tl.float32))  # Cast remote pointer
    if rank == root_rank:
        thread_idx = tl.program_id(axis=0)         # Unique thread index
        block_dim = tl.num_programs(axis=0)        # Total number of threads (programs)
        src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.float32))
        num_int4 = nbytes // 16                  # Convert total byte count to int4 element count
        for n in range(thread_idx, num_int4, block_dim):
            # Load 4 consecutive 32-bit unsigned integers (4 floats) from source buffer at offset 4*n
            val0, val1, val2, val3 = load_v4_u32(src_ptr + 4 * n)
            # Pack pairs of 32-bit integers into 64-bit values for multicast store:
            # Even though they are floats, we just care about the bytes so it's ok to use integers like this
            # val01 packs val1 (upper 32 bits) and val0 (lower 32 bits)
            val01 = (tl.cast(val1, tl.uint64) << 32) | tl.cast(val0, tl.uint64)
            val23 = (tl.cast(val3, tl.uint64) << 32) | tl.cast(val2, tl.uint64)
            # Calculate base offset in floats for multicast stores:
            # Each iteration handles 4 floats, so offset advances by 4 floats per iteration
            # As noted in the comment, triton operates on indices, not bytes
            base_offset = 4 * n
            # Perform two 64-bit multicast stores to write all 4 floats:
            # First store writes val0 and val1 at remote_mc_ptr[base_offset : base_offset+2]
            multimem_st_b64(remote_mc_ptr + base_offset, val01)
            # Second store writes val2 and val3 at remote_mc_ptr[base_offset+2 : base_offset+4]
            multimem_st_b64(remote_mc_ptr + base_offset + 2, val23)


# ----- Runtime Setup -----

# Select a GPU device for the current MPI rank (one GPU per process)
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()

# Initialize NVSHMEM with MPI-based transport
nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

# Obtain ranks and teams
mype = nvshmem.my_pe()
npes = nvshmem.n_pes()
mype_node = nvshmem.team_my_pe(nvshmem.Teams.TEAM_NODE)

# ----- Tensor Allocation -----

input_nelems = 512
stream = dev.create_stream()
tensor = nvshmem.tensor((input_nelems,), dtype=torch.float32)  # Shared tensor allocated by NVSHMEM

# Initialize root PE's data for broadcast
if mype == 0:
    tensor[:] = 1.0

buf_sz = tensor.numel() * tensor.element_size()  # Size in bytes

# Obtain a remote multicast-compatible pointer (usable only from device code)
remote_mc_ptr = mc_ptr(nvshmem.Teams.TEAM_WORLD, tensor.data_ptr())

print(f"[PE {mype}] Tensor before broadcast:", tensor)

# ----- Launch Broadcast Kernel -----

# Triton launch syntax: kernel_name[grid](args...)
# grid = (input_nelems,) launches one thread per element (or per float pair)
broadcast_naive_block[(input_nelems,)](tensor, buf_sz, mype, 0, remote_mc_ptr)

# Synchronize across PEs and the CUDA stream
nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=stream)
torch.cuda.synchronize()

print(f"[PE {mype}] Tensor after broadcast:", tensor)

# ----- Cleanup -----
nvshmem.free_tensor(tensor)
nvshmem.finalize()
