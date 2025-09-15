# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

# Based on https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management 

"""
Example: Using NVSHMEM-backed CUDA VMM memory pool with PyTorch

This example demonstrates how to integrate a custom CUDA memory allocator using
NVSHMEM and CUDA Virtual Memory Management (VMM) APIs into PyTorch. By leveraging
NVSHMEM and VMM, device memory allocations can be shared efficiently across multiple
processes or GPUs, which is particularly useful for distributed training and collective
communication libraries such as NVSHMEM.

The example provides:
- A C++ extension source for a custom allocator that uses CUDA VMM APIs to allocate
  and free device memory with NVSHMEM compatibility.
- Guidance on how to register this allocator with PyTorch's memory management system
  via the CUDAPluggableAllocator interface.
- A template for integrating this allocator into distributed PyTorch workflows.

This approach is intended for advanced users who need fine-grained control over
device memory allocation and sharing in multi-GPU or multi-process environments.

References:
- PyTorch CUDA memory management: https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
- CUDA VMM APIs: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
- NVSHMEM: https://developer.nvidia.com/nvshmem
"""

import os

import torch
import torch.distributed as dist
from torch.cuda.memory import CUDAPluggableAllocator
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils import cpp_extension


import nvshmem.core
from cuda.core.experimental import Device

# Allocate memory with CUDA VMM APIs
vmm_allocator_source = """
#include <cuda.h>
extern "C" {

void* nvshmem_vmm_alloc(size_t size, int device, void* stream) {
    void *bufAddr = nullptr;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle userAllocHandle;

    CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));
    CU_CHECK(
        cuMemSetAccess((CUdeviceptr)bufAddr, size, (const CUmemAccessDesc *)&accessDescriptor, 1));
    return bufAddr;


}

void nvshmem_vmm_free(void* ptr, size_t size, int device, void* stream) {
    CUmemGenericAllocationHandle memHandle;
    CU_CHECK(cuMemRetainAllocationHandle(&memHandle, ptr));
    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(memHandle));
}

}
"""
vmm_allocator_libname = "vmm_allocator"
vmm_allocator = torch.utils.cpp_extension.load_inline(
    name=vmm_allocator_libname,
    cpp_sources=vmm_allocator_source,
    with_cuda=True,
    extra_ldflags=["-lnvshmem"],
    verbose=True,
    is_python_module=False,
    build_directory="./",
)

# NOTE: it would be interesting to explore if Torch is willing to support a CUDAPluggableAllocator in pure Python
# Then we could use nvshmem4py buffer() and register_external_tensor() to register the buffer with NVSHMEM without using any C++
allocator = CUDAPluggableAllocator(
    f"./{vmm_allocator_libname}.so", "nvshmem_vmm_alloc", "nvshmem_vmm_free"
).allocator()

# setup distributed
rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
world_size = int(os.getenv("WORLD_SIZE"))
torch.cuda.set_device(local_rank)
# NOTE! Torch doesn't support a NVSHMEM backend, so we use NCCL
# Internally, Torch will fail if you pass non-"NCCL" as the backend to cuda or leave it blank
# Even if you don't use the backend
dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{local_rank}")
dev = Device(local_rank)
default_pg = _get_default_group()
backend = default_pg._get_backend(device)

# Create an empty uniqueid for all ranks
uniqueid = nvshmem.core.get_unique_id(empty=True)
if rank == 0:
    # Rank 0 gets a real uniqueid
    uniqueid = nvshmem.core.get_unique_id()

torch.distributed.broadcast_object_list([uniqueid], src=0)
dist.barrier()

# Initialize NVSHMEM using UID bootstrapped with torch-distributed broadcast
nvshmem.core.init(uid=uniqueid, rank=local_rank, nranks=world_size, initializer_method="uid")

# Create a memory pool
pool = torch.cuda.MemPool(allocator)

# Allocate memory with the memory pool
with torch.cuda.use_mem_pool(pool):
    tensor = torch.arange(1024 * 1024 * 2, device=device)

# Register the buffer with NVSHMEM
registered_tensor = nvshmem.core.register_external_tensor(tensor)

# Allreduce
nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, registered_tensor, registered_tensor, op="sum", stream=dev.create_stream())
torch.cuda.synchronize(device=device)

# Clean up memory
nvshmem.core.unregister_external_tensor(registered_tensor)
del tensor, pool

# Finalize NVSHMEM
nvshmem.core.finalize()



