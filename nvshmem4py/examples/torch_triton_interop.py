"""
This example initializes NVSHMEM4Py with the `torchrun` 
launcher and torch.distributed

It runs a kernel expressed with Triton

Run this program with `torchrun --nproc-per-node <NGPUs> torch_triton_interop.py`
"""

import torch.distributed as dist
import torch
import triton
import triton.language as tl
import nvshmem.core
import os
from cuda.core.experimental import Device, system

###
#  Helper code from https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/pytorch_example.py
#  Used to extract PyTorch Stream into a cuda.core.Stream for NVSHMEM APIs
###

# Create a wrapper class that implements __cuda_stream__
# Example of using https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def torchrun_uid_init():
    """
    Initialize NVSHMEM using UniqueID with `torchrun` as the launcher
    """
    # Set Torch device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # nvshmem4py requires a cuda.core Device at init time
    global dev
    dev = Device(device.index)
    dev.set_current()
    global stream
    # Get PyTorch's current stream
    pt_stream = torch.cuda.current_stream()
    stream = PyTorchStreamWrapper(pt_stream)

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
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    # We use torch.distributed.broadcast_object_list to send the UID to all ranks
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    """
    Addition kernel borrowed from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
    """
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


if __name__ == '__main__':
    torchrun_uid_init()

    """
    Allocate 3 tensors on the NVSHMEM symmetric heap
    We will add tensor1 to tensor2, and store that to tensor_out
    Then, we will use nvshmem.core to sum-reduce all PEs' copies of tensor_out
    """
    n_elements = 867530
    tensor1 = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
    tensor1[:] = nvshmem.core.my_pe() + 1
    tensor2 = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
    tensor2[:] = nvshmem.core.my_pe() + 2
    tensor_out = nvshmem.core.tensor((n_elements,), dtype=torch.float32)

    """
    Launch the vector addition kernel
    """
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # This gets launched on the Torch current stream
    add_kernel[grid](tensor1, tensor2, tensor_out, n_elements, BLOCK_SIZE=1024)
    # If you uncomment this, you need to add torch.cuda.synchronize() first
    # print(f"From {nvshmem.core.my_pe()} intermediate output: {tensor_out}")

    """
    use nvshmem.core to reduce (sum) all the copies of tensor_out
    No need to synchronize, because both operations are on the same Stream
    """
    nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, tensor_out, tensor_out, "sum", stream=stream)
    if nvshmem.core.my_pe() == 0:
        expected_val = 0
        for i in range(nvshmem.core.n_pes()):
            expected_val += (i + 1)
            expected_val += (i + 2)

        expected_tensor = torch.zeros_like(tensor_out)
        expected_tensor[:] = expected_val
        torch.cuda.synchronize()
        torch.testing.assert_close(tensor_out, expected_tensor)
        print(f"Final output: {tensor_out}")

    nvshmem.core.free_tensor(tensor1)
    nvshmem.core.free_tensor(tensor2)
    nvshmem.core.free_tensor(tensor_out)
    nvshmem.core.finalize()
    dist.destroy_process_group()
