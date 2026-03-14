# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import nvshmem.core
from mpi4py import MPI
from functools import partial
from typing import Union
import os
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer, Boolean, Int32, Int, Int64, Constexpr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch.nvvm_wrappers import FULL_MASK, WARP_SIZE
import cutlass
from cuda.core import Device, system

from nvshmem.bindings.device.cute import int_p as cute_int_p
from nvshmem.bindings.device.cute import my_pe as cute_my_pe
from nvshmem.bindings.device.cute import n_pes as cute_n_pes


@cute.kernel
def simple_shift_kernel(destTensor: cute.Tensor, ):
    tidx, _, _ = cute.arch.thread_idx()

    mype = cute_my_pe()
    npes = cute_n_pes()
    peer = (mype + 1) % npes

    if tidx == 0:
        cute.printf("mype: %d, peer: %d, npes: %d, value: %d", mype, peer, npes, mype + 1)
        cute.printf("tidx: %d", tidx)
        cute_int_p(destTensor.iterator, mype + 1, peer)


@cute.jit
def simple_shift(destTensor: cute.Tensor, ):
    simple_shift_kernel(destTensor, ).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
    )


def run():
    my_rank = MPI.COMM_WORLD.Get_rank()
    local_rank = my_rank % system.get_num_devices()
    dev = Device(local_rank)
    dev.set_current()
    nvshmem.core.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    my_pe = nvshmem.core.my_pe()
    tensor = nvshmem.core.tensor(8, dtype=torch.int32)

    tensor_dlpack = from_dlpack(tensor).mark_layout_dynamic()
    print(tensor_dlpack)

    nvshmem_device_bc = nvshmem.core.find_device_bitcode_library()

    compilerd_func = cute.compile(
        simple_shift,
        tensor_dlpack,
        options=f" --link-libraries={nvshmem_device_bc}",
    )

    compilerd_func = compilerd_func.to(my_pe)
    cuda_library = compilerd_func.jit_module.cuda_library
    nvshmem_kernel = nvshmem.core.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.core.library_init(nvshmem_kernel)

    # launch the kernel
    compilerd_func(tensor_dlpack)
    torch.cuda.synchronize()
    print(tensor)
    nvshmem.core.free_tensor(tensor)
    nvshmem.core.finalize()


if __name__ == "__main__":
    run()
