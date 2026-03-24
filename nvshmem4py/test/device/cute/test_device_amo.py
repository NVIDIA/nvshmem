from cuda.core import Device, Stream, system
import numpy as np
import pytest

import cutlass.cute as cute
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE

import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute
import nvshmem.core.device.cute.amo as nvshmem_cute_amo
import nvshmem.core.interop.cute as cute_interop

from test_device_rma import (
    _compile_kernel,
    _nvshmem_stream,
    _fill_cute_tensor,
    _read_cute_tensor,
    _cute_dtype,
)

amo_std_dtypes = ["int32", "int64", "uint64"]
amo_float_dtypes = ["float32", "float64"]
amo_swap_dtypes = amo_std_dtypes + amo_float_dtypes

_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint64": np.uint64,
}


def _assert_tensor_equals(tensor, dtype, expected):
    host = _read_cute_tensor(tensor, dtype)
    if np.issubdtype(host.dtype, np.floating):
        assert np.allclose(host, expected)
    else:
        assert (host == expected).all()


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_std_dtypes)
def test_atomic_add_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.get_num_devices()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(buf, dtype, 0)

    @cute.kernel
    def kernel_atomic_add(arr: cute.Tensor, val: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute_amo.atomic_add(arr, val, pe)

    @cute.jit
    def kernel_atomic_add_launcher(arr: cute.Tensor, val: Int32, pe: Int32):
        kernel_atomic_add(arr, val, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(kernel_atomic_add_launcher, buf, 0, 0)
    compiled(buf, 5, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_tensor_equals(buf, dtype, _NUMPY_DTYPE_MAP[dtype](5))
    cute_interop.free_tensor(buf)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_std_dtypes)
def test_atomic_fetch_add_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.get_num_devices()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf = cute_interop.tensor((1, ), dtype=cute_dtype)
    out = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(buf, dtype, 0)
    _fill_cute_tensor(out, dtype, 0)

    @cute.kernel
    def kernel_atomic_fetch_add(arr: cute.Tensor, out: cute.Tensor, val: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            result = nvshmem_cute_amo.atomic_fetch_add(arr, val, pe)
            out[0] = result

    @cute.jit
    def kernel_atomic_fetch_add_launcher(arr: cute.Tensor, out: cute.Tensor, val: Int32, pe: Int32):
        kernel_atomic_fetch_add(arr, out, val, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(kernel_atomic_fetch_add_launcher, buf, out, 0, 0)
    compiled(buf, out, 5, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_tensor_equals(buf, dtype, _NUMPY_DTYPE_MAP[dtype](5))
    _assert_tensor_equals(out, dtype, _NUMPY_DTYPE_MAP[dtype](0))

    cute_interop.free_tensor(buf)
    cute_interop.free_tensor(out)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_float_dtypes)
def test_atomic_fetch_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.get_num_devices()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf = cute_interop.tensor((1, ), dtype=cute_dtype)
    out = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(buf, dtype, 4.5)
    _fill_cute_tensor(out, dtype, 0)

    @cute.kernel
    def kernel_atomic_fetch(arr: cute.Tensor, out: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            result = nvshmem_cute_amo.atomic_fetch(arr, pe)
            out[0] = result

    @cute.jit
    def kernel_atomic_fetch_launcher(arr: cute.Tensor, out: cute.Tensor, pe: Int32):
        kernel_atomic_fetch(arr, out, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(kernel_atomic_fetch_launcher, buf, out, 0)
    compiled(buf, out, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_tensor_equals(out, dtype, _NUMPY_DTYPE_MAP[dtype](4.5))

    cute_interop.free_tensor(buf)
    cute_interop.free_tensor(out)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_float_dtypes)
def test_atomic_set_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.get_num_devices()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(buf, dtype, 0)

    @cute.kernel
    def kernel_atomic_set(arr: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute_amo.atomic_set(arr, 7.25, pe)

    @cute.jit
    def kernel_atomic_set_launcher(arr: cute.Tensor, pe: Int32):
        kernel_atomic_set(arr, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(kernel_atomic_set_launcher, buf, 0)
    compiled(buf, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_tensor_equals(buf, dtype, _NUMPY_DTYPE_MAP[dtype](7.25))
    cute_interop.free_tensor(buf)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_swap_dtypes)
def test_atomic_swap_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    local_rank = nvshmem.core.my_pe() % system.get_num_devices()
    dev = Device()
    cute_dtype = _cute_dtype(dtype)
    buf = cute_interop.tensor((1, ), dtype=cute_dtype)
    out = cute_interop.tensor((1, ), dtype=cute_dtype)
    _fill_cute_tensor(buf, dtype, 1.25)
    _fill_cute_tensor(out, dtype, 0)

    @cute.kernel
    def kernel_atomic_swap(arr: cute.Tensor, out: cute.Tensor, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            result = nvshmem_cute_amo.atomic_swap(arr, 3.5, pe)
            out[0] = result

    @cute.jit
    def kernel_atomic_swap_launcher(arr: cute.Tensor, out: cute.Tensor, pe: Int32):
        kernel_atomic_swap(arr, out, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(kernel_atomic_swap_launcher, buf, out, 0)
    compiled(buf, out, nvshmem.core.my_pe())

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_tensor_equals(buf, dtype, _NUMPY_DTYPE_MAP[dtype](3.5))
    if dtype in amo_float_dtypes:
        pytest.xfail("nvshmem_*_atomic_swap returns integer-cast old value for float/double (NVSHMEM_TYPE_SWAP_CAST)")
    _assert_tensor_equals(out, dtype, _NUMPY_DTYPE_MAP[dtype](1.25))

    cute_interop.free_tensor(buf)
    cute_interop.free_tensor(out)
