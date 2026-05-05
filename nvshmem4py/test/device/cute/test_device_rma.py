# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from cuda.core import Device
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE

import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute

_KERNEL_OBJECTS: list[nvshmem.core.NvshmemKernelObject] = []

rma_dtypes = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": getattr(torch, "uint16", None),
    "uint32": getattr(torch, "uint32", None),
    "uint64": getattr(torch, "uint64", None),
}


def _torch_dtype(dtype_name):
    dtype = _TORCH_DTYPE_MAP.get(dtype_name)
    if dtype is None:
        pytest.skip(f"Torch dtype not supported for CuTe test: {dtype_name}")
    return dtype


def _make_torch_tensor(shape, dtype_name, value):
    tensor = nvshmem.core.tensor(shape, dtype=_torch_dtype(dtype_name))
    tensor.fill_(value)
    Device().sync()
    return tensor


def _cute_from_torch(tensor):
    return from_dlpack(tensor).mark_layout_dynamic()


def _assert_torch_tensor(tensor, value):
    expected = torch.full_like(tensor, value)
    assert torch.equal(tensor, expected)


def _nvshmem_device_bc():
    try:
        nvshmem_device_bc = nvshmem.core.find_device_bitcode_library()
    except Exception as e:
        pytest.skip(f"Failed to locate NVSHMEM device bitcode library: {e}")

    if not os.path.exists(nvshmem_device_bc):
        pytest.skip(f"NVSHMEM device bitcode not found at {nvshmem_device_bc}")

    return nvshmem_device_bc


def _nvshmem_stream():
    dev = Device()
    return dev.create_stream()


def _compile_kernel(kernel, *example_args):
    nvshmem_device_bc = _nvshmem_device_bc()
    compiled = cute.compile(
        kernel,
        *example_args,
        options=f" --link-libraries={nvshmem_device_bc}",
    )
    compiled = compiled.to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    nvshmem_kernel = nvshmem.core.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.core.library_init(nvshmem_kernel)
    _KERNEL_OBJECTS.append(nvshmem_kernel)
    return compiled


def _finalize_kernels():
    while _KERNEL_OBJECTS:
        nvshmem.core.library_finalize(_KERNEL_OBJECTS.pop())


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    buf_src = _make_torch_tensor((4, 4), dtype, nvshmem.core.my_pe() + 1)
    buf_dst = _make_torch_tensor((4, 4), dtype, 0)

    dst_cute = _cute_from_torch(buf_dst)
    src_cute = _cute_from_torch(buf_src)

    @cute.kernel
    def test_put(dst: cute.Tensor, src: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.put(dst, src, pe)

    @cute.jit
    def test_put_launcher(dst: cute.Tensor, src: cute.Tensor, pe: Int32):
        test_put(dst, src, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_put_launcher, dst_cute, src_cute, 0)

    peer = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    compiled(dst_cute, src_cute, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    expected = ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()) + 1
    _assert_torch_tensor(buf_dst, expected)

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_get_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    buf_src = _make_torch_tensor((4, 4), dtype, 0)
    buf_dst = _make_torch_tensor((4, 4), dtype, nvshmem.core.my_pe() + 1)

    dst_cute = _cute_from_torch(buf_src)
    src_cute = _cute_from_torch(buf_dst)

    @cute.kernel
    def test_get(dst: cute.Tensor, src: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.get(dst, src, pe)

    @cute.jit
    def test_get_launcher(dst: cute.Tensor, src: cute.Tensor, pe: Int32):
        test_get(dst, src, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_get_launcher, dst_cute, src_cute, 0)
    compiled(dst_cute, src_cute, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(buf_dst, nvshmem.core.my_pe() + 1)

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    buf_src = _make_torch_tensor((4, 4), dtype, nvshmem.core.my_pe() + 1)
    buf_dst = _make_torch_tensor((4, 4), dtype, 0)
    signal_var = _make_torch_tensor((1, ), "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    dst_cute = _cute_from_torch(buf_dst)
    src_cute = _cute_from_torch(buf_src)
    signal_cute = _cute_from_torch(signal_var)

    @cute.kernel
    def test_put_signal(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_val: Int32,
                        signal_op: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.put_signal(dst, src, signal_var, signal_val, signal_op, pe)

    @cute.jit
    def test_put_signal_launcher(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_val: Int32,
                                 signal_op: Int32, pe: Int32):
        test_put_signal(dst, src, signal_var, signal_val, signal_op, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_put_signal_launcher, dst_cute, src_cute, signal_cute, 0, 0, 0)
    compiled(dst_cute, src_cute, signal_cute, signal_val, signal_op, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(buf_dst, nvshmem.core.my_pe() + 1)

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)
    nvshmem.core.free_tensor(signal_var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_with_wait_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    buf_src = _make_torch_tensor((4, 4), dtype, nvshmem.core.my_pe() + 1)
    buf_dst = _make_torch_tensor((4, 4), dtype, 0)
    signal_var = _make_torch_tensor((1, ), "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    dst_cute = _cute_from_torch(buf_dst)
    src_cute = _cute_from_torch(buf_src)
    signal_cute = _cute_from_torch(signal_var)

    @cute.kernel
    def test_put_signal_with_wait(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_val: Int32,
                                  signal_op: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.put_signal(dst, src, signal_var, signal_val, signal_op, pe)
            nvshmem_cute.signal_wait(signal_var, nvshmem.core.ComparisonType.CMP_GE, signal_val)

    @cute.jit
    def test_put_signal_with_wait_launcher(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor,
                                           signal_val: Int32, signal_op: Int32, pe: Int32):
        test_put_signal_with_wait(dst, src, signal_var, signal_val, signal_op, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_put_signal_with_wait_launcher, dst_cute, src_cute, signal_cute, 0, 0, 0)
    compiled(dst_cute, src_cute, signal_cute, signal_val, signal_op, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    if nvshmem.core.my_pe() == 1:
        _assert_torch_tensor(buf_dst, nvshmem.core.my_pe() + 1)

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)
    nvshmem.core.free_tensor(signal_var)


@pytest.mark.mpi
def test_signal_op_signal_wait(nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    signal_var = _make_torch_tensor((1, ), "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    signal_cute = _cute_from_torch(signal_var)

    @cute.kernel
    def test_signal_op_signal_wait(signal_var: cute.Tensor, signal_val: Int32, signal_op: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.signal_op(signal_var, signal_val, signal_op, pe)
            nvshmem_cute.signal_wait(signal_var, nvshmem.core.ComparisonType.CMP_GE, signal_val)

    @cute.jit
    def test_signal_op_signal_wait_launcher(signal_var: cute.Tensor, signal_val: Int32, signal_op: Int32, pe: Int32):
        test_signal_op_signal_wait(signal_var, signal_val, signal_op, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_signal_op_signal_wait_launcher, signal_cute, 0, 0, 0)
    compiled(signal_cute, signal_val, signal_op, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    nvshmem.core.free_tensor(signal_var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_p(dtype, nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    var = _make_torch_tensor((1, ), dtype, 0)
    val = 1

    var_cute = _cute_from_torch(var)

    @cute.kernel
    def test_p_kernel(var: cute.Tensor, val: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute.p(var, val, pe)

    @cute.jit
    def test_p_launcher(var: cute.Tensor, val: Int32, pe: Int32):
        test_p_kernel(var, val, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_p_launcher, var_cute, 0, 0)
    compiled(var_cute, val, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(var, 1)

    nvshmem.core.free_tensor(var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_g(dtype, nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    var = _make_torch_tensor((1, ), dtype, 1)
    dest = _make_torch_tensor((1, ), dtype, 0)

    var_cute = _cute_from_torch(var)
    dest_cute = _cute_from_torch(dest)

    @cute.kernel
    def test_g_kernel(dest: cute.Tensor, var: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            # TODO: g() triggers a ptxas ICE in current CuTe DSL; use get() to validate RMA read.
            nvshmem_cute.get(dest, var, pe)

    @cute.jit
    def test_g_launcher(dest: cute.Tensor, var: cute.Tensor, pe: Int32):
        test_g_kernel(dest, var, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(test_g_launcher, dest_cute, var_cute, 0)
    compiled(dest_cute, var_cute, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(dest, 1)

    nvshmem.core.free_tensor(dest)
    nvshmem.core.free_tensor(var)
