import os

import pytest
from cuda.core import Device, Stream, system
import numpy as np
import cutlass.cute as cute
import cuda.bindings.driver as cudrv
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
from cutlass.base_dsl.env_manager import detect_gpu_arch

import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute
import nvshmem.core.interop.cute as cute_interop
import nvshmem.core.device.cute.rma as nvshmem_cute_rma

_KERNEL_OBJECTS: list[nvshmem.core.NvshmemKernelObject] = []

rma_dtypes = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

_CUTE_DTYPE_MAP = {
    "float32": cute.Float32,
    "float64": cute.Float64,
    "int8": cute.Int8,
    "int16": cute.Int16,
    "int32": cute.Int32,
    "int64": cute.Int64,
    "uint8": cute.Uint8,
    "uint16": cute.Uint16,
    "uint32": cute.Uint32,
    "uint64": cute.Uint64,
}

_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _cute_dtype(dtype_name):
    dtype = _CUTE_DTYPE_MAP.get(dtype_name)
    if dtype is None:
        pytest.skip(f"CuTe dtype not supported for CuTe test: {dtype_name}")
    return dtype


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


def _fill_cute_tensor(tensor, dtype_name, value):
    np_dtype = _NUMPY_DTYPE_MAP[dtype_name]
    host = np.full(tuple(tensor.shape), np_dtype(value), dtype=np_dtype)
    buf, _, _ = cute_interop.tensor_get_buffer(tensor)
    cudrv.cuMemcpyHtoD(buf.handle, host, host.nbytes)
    dev = Device()
    dev.sync()


def _read_cute_tensor(tensor, dtype_name):
    np_dtype = _NUMPY_DTYPE_MAP[dtype_name]
    host = np.empty(tuple(tensor.shape), dtype=np_dtype)
    buf, _, _ = cute_interop.tensor_get_buffer(tensor)
    cudrv.cuMemcpyDtoH(host, buf.handle, host.nbytes)
    dev = Device()
    dev.sync()
    return host


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf_src = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_src, dtype, nvshmem.core.my_pe() + 1)
    buf_dst = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_dst, dtype, 0)

    dst_cute = buf_dst
    src_cute = buf_src

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
    expected_host = np.full((4, 4), _NUMPY_DTYPE_MAP[dtype](expected), dtype=_NUMPY_DTYPE_MAP[dtype])
    assert (_read_cute_tensor(buf_dst, dtype) == expected_host).all()

    cute_interop.free_tensor(buf_dst)
    cute_interop.free_tensor(buf_src)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_get_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf_src = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_src, dtype, 0)
    buf_dst = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_dst, dtype, nvshmem.core.my_pe() + 1)

    dst_cute = buf_src
    src_cute = buf_dst

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

    expected_host = np.full((4, 4), _NUMPY_DTYPE_MAP[dtype](nvshmem.core.my_pe() + 1), dtype=_NUMPY_DTYPE_MAP[dtype])
    assert (_read_cute_tensor(buf_dst, dtype) == expected_host).all()

    cute_interop.free_tensor(buf_dst)
    cute_interop.free_tensor(buf_src)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf_src = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_src, dtype, nvshmem.core.my_pe() + 1)
    buf_dst = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_dst, dtype, 0)
    signal_var = cute_interop.tensor((1, ), dtype=cute.Int64)
    _fill_cute_tensor(signal_var, "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    dst_cute = buf_dst
    src_cute = buf_src
    signal_cute = signal_var

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

    expected_host = np.full((4, 4), _NUMPY_DTYPE_MAP[dtype](nvshmem.core.my_pe() + 1), dtype=_NUMPY_DTYPE_MAP[dtype])
    assert (_read_cute_tensor(buf_dst, dtype) == expected_host).all()

    cute_interop.free_tensor(buf_dst)
    cute_interop.free_tensor(buf_src)
    cute_interop.free_tensor(signal_var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_with_wait_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf_src = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_src, dtype, nvshmem.core.my_pe() + 1)
    buf_dst = cute_interop.tensor((4, 4), dtype=cute_dtype)
    _fill_cute_tensor(buf_dst, dtype, 0)
    signal_var = cute_interop.tensor((1, ), dtype=cute.Int64)
    _fill_cute_tensor(signal_var, "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    dst_cute = buf_dst
    src_cute = buf_src
    signal_cute = signal_var

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
        expected_host = np.full((4, 4),
                                _NUMPY_DTYPE_MAP[dtype](nvshmem.core.my_pe() + 1),
                                dtype=_NUMPY_DTYPE_MAP[dtype])
        assert (_read_cute_tensor(buf_dst, dtype) == expected_host).all()

    cute_interop.free_tensor(buf_dst)
    cute_interop.free_tensor(buf_src)
    cute_interop.free_tensor(signal_var)


@pytest.mark.mpi
def test_signal_op_signal_wait(nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    signal_var = cute_interop.tensor((1, ), dtype=cute.Int64)
    _fill_cute_tensor(signal_var, "int64", 0)
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    signal_cute = signal_var

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

    cute_interop.free_tensor(signal_var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_p(dtype, nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    var = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(var, dtype, 0)
    val = 1

    var_cute = var

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

    expected_host = np.full((1, ), _NUMPY_DTYPE_MAP[dtype](1), dtype=_NUMPY_DTYPE_MAP[dtype])
    assert (_read_cute_tensor(var, dtype) == expected_host).all()

    cute_interop.free_tensor(var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_g(dtype, nvshmem_init_fini):
    stream = _nvshmem_stream()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    var = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(var, dtype, 1)
    dest = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(dest, dtype, 0)

    var_cute = var
    dest_cute = dest

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

    expected_host = np.full((1, ), _NUMPY_DTYPE_MAP[dtype](1), dtype=_NUMPY_DTYPE_MAP[dtype])
    assert (_read_cute_tensor(dest, dtype) == expected_host).all()

    cute_interop.free_tensor(dest)
    cute_interop.free_tensor(var)
