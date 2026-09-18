# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda.core import Device
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE

import nvshmem.core
from nvshmem.core.interop.cute import cute_compile_helper
import nvshmem.core.device.cute as nvshmem_cute
import nvshmem.bindings.device.cute as nvshmem_cute_bindings

_KERNEL_OBJECTS: list[nvshmem.core.NvshmemKernelObject] = []

rma_dtypes = [
    "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
]

_TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
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


def _nvshmem_stream():
    dev = Device()
    return dev.create_stream()


def _compile_kernel(kernel, *example_args):
    compiled, nvshmem_kernel = cute_compile_helper(kernel, *example_args)
    _KERNEL_OBJECTS.append(nvshmem_kernel)
    return compiled


def _finalize_kernels():
    while _KERNEL_OBJECTS:
        nvshmem.core.library_finalize(_KERNEL_OBJECTS.pop())


@pytest.mark.mpi
def test_direct_externs_link_nvshmem_bitcode(nvshmem_init_fini):
    """Link multiple typed externs without passing ``--link-libraries``."""
    stream = _nvshmem_stream()
    dev = Device()
    int_src = _make_torch_tensor((1, ), "int32", nvshmem.core.my_pe() + 1)
    int_dst = _make_torch_tensor((1, ), "int32", 0)
    float_src = _make_torch_tensor((1, ), "float32", float(nvshmem.core.my_pe() + 1))
    float_dst = _make_torch_tensor((1, ), "float32", 0)

    int_src_cute = _cute_from_torch(int_src)
    int_dst_cute = _cute_from_torch(int_dst)
    float_src_cute = _cute_from_torch(float_src)
    float_dst_cute = _cute_from_torch(float_dst)

    @cute.kernel
    def direct_externs(
        int_dst: cute.Tensor,
        int_src: cute.Tensor,
        float_dst: cute.Tensor,
        float_src: cute.Tensor,
        pe: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            nvshmem_cute_bindings.int32_put(
                int_dst.iterator,
                int_src.iterator,
                cutlass.Uint64(1),
                pe,
            )
            nvshmem_cute_bindings.float_put(
                float_dst.iterator,
                float_src.iterator,
                cutlass.Uint64(1),
                pe,
            )

    @cute.jit
    def direct_externs_launcher(
        int_dst: cute.Tensor,
        int_src: cute.Tensor,
        float_dst: cute.Tensor,
        float_src: cute.Tensor,
        pe: Int32,
    ):
        direct_externs(int_dst, int_src, float_dst, float_src, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    compiled = _compile_kernel(
        direct_externs_launcher,
        int_dst_cute,
        int_src_cute,
        float_dst_cute,
        float_src_cute,
        0,
    )
    peer = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    compiled(int_dst_cute, int_src_cute, float_dst_cute, float_src_cute, peer)

    dev.sync()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    previous_pe = (nvshmem.core.my_pe() - 1) % nvshmem.core.n_pes()
    expected = previous_pe + 1
    _assert_torch_tensor(int_dst, expected)
    _assert_torch_tensor(float_dst, float(expected))

    nvshmem.core.free_tensor(int_src)
    nvshmem.core.free_tensor(int_dst)
    nvshmem.core.free_tensor(float_src)
    nvshmem.core.free_tensor(float_dst)


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
    # All PEs must finish initializing their destination before a peer writes it.
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    compiled(dst_cute, src_cute, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    # Each PE puts its own src into (my_pe + 1), so the value that lands in
    # buf_dst comes from the PE whose successor is us -- not from our own
    # successor. The two coincide only when there are exactly 2 PEs.
    predecessor = (nvshmem.core.my_pe() - 1 + nvshmem.core.n_pes()) % nvshmem.core.n_pes()
    _assert_torch_tensor(buf_dst, predecessor + 1)

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_get_on_tensor(nvshmem_init_fini, dtype):
    stream = _nvshmem_stream()
    dev = Device()
    buf_src = _make_torch_tensor((4, 4), dtype, nvshmem.core.my_pe() + 1)
    buf_dst = _make_torch_tensor((4, 4), dtype, 0)

    dst_cute = _cute_from_torch(buf_dst)
    src_cute = _cute_from_torch(buf_src)

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
    # Read the same predecessor whose value the ring put delivers locally.
    peer = (nvshmem.core.my_pe() - 1) % nvshmem.core.n_pes()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    compiled(dst_cute, src_cute, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(buf_dst, peer + 1)

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
    peer = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    compiled(dst_cute, src_cute, signal_cute, signal_val, signal_op, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    expected = ((nvshmem.core.my_pe() - 1) % nvshmem.core.n_pes()) + 1
    _assert_torch_tensor(buf_dst, expected)
    _assert_torch_tensor(signal_var, signal_val)

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
    peer = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    compiled(dst_cute, src_cute, signal_cute, signal_val, signal_op, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    expected = ((nvshmem.core.my_pe() - 1) % nvshmem.core.n_pes()) + 1
    _assert_torch_tensor(buf_dst, expected)
    _assert_torch_tensor(signal_var, signal_val)

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
    peer = (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    compiled(signal_cute, signal_val, signal_op, peer)

    dev.sync()  # Sync to ensure kernel completes before barrier
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    _assert_torch_tensor(signal_var, signal_val)

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


_SCOPE_WARPS = 2
_SCOPE_CHUNK = 128  # elements handled per warp for warp-scoped calls
_SCOPE_NELEMS = _SCOPE_WARPS * _SCOPE_CHUNK
_SCOPE_OPS = ["put", "put_nbi", "get", "get_nbi", "put_signal", "put_signal_nbi"]
_SCOPES = ["", "_warp", "_block"]
_SCOPE_DTYPES = ["int8", "float32"]


def _make_scope_rma_kernel(op, scope):
    """Build a launcher that issues ``op`` at ``scope`` from a 2-warp CTA.

    Thread scope issues one call from thread 0, warp scope issues one call per
    warp on that warp's slice, and block scope issues one call from the whole CTA.
    """
    rma = getattr(nvshmem_cute, f"{op}{scope}")
    is_signal = op.startswith("put_signal")
    is_nbi = op.endswith("_nbi")

    @cute.jit
    def issue(dst, src, signal_var, signal_op, pe):
        if cutlass.const_expr(is_signal):
            rma(dst, src, signal_var, cutlass.Uint64(1), signal_op, pe)
        else:
            rma(dst, src, pe)

    @cute.kernel
    def scope_rma(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_op: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(scope == "_warp"):
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            issue(cute.local_tile(dst, (_SCOPE_CHUNK, ), (warp, )), cute.local_tile(src, (_SCOPE_CHUNK, ), (warp, )),
                  signal_var, signal_op, pe)
        elif cutlass.const_expr(scope == "_block"):
            issue(dst, src, signal_var, signal_op, pe)
        else:
            if tidx == 0:
                issue(dst, src, signal_var, signal_op, pe)
        if cutlass.const_expr(is_nbi):
            nvshmem_cute.quiet()

    @cute.jit
    def scope_rma_launcher(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_op: Int32, pe: Int32):
        scope_rma(dst, src, signal_var, signal_op, pe).launch(
            grid=[1, 1, 1],
            block=[_SCOPE_WARPS * 32, 1, 1],
        )

    return scope_rma_launcher


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", _SCOPE_DTYPES)
@pytest.mark.parametrize("scope", _SCOPES, ids=["thread", "warp", "block"])
@pytest.mark.parametrize("op", _SCOPE_OPS)
def test_rma_scopes(nvshmem_init_fini, op, scope, dtype):
    """Validate thread-, warp-, and block-scoped RMA wrappers end to end."""
    stream = _nvshmem_stream()
    dev = Device()
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    peer = (my_pe + 1) % n_pes
    previous_pe = (my_pe - 1) % n_pes

    src = _make_torch_tensor((_SCOPE_NELEMS, ), dtype, my_pe + 1)
    dst = _make_torch_tensor((_SCOPE_NELEMS, ), dtype, 0)
    signal_var = _make_torch_tensor((1, ), "int64", 0)
    src_cute = _cute_from_torch(src)
    dst_cute = _cute_from_torch(dst)
    signal_cute = _cute_from_torch(signal_var)

    signal_op = nvshmem.core.SignalOp.SIGNAL_ADD

    compiled = _compile_kernel(_make_scope_rma_kernel(op, scope), dst_cute, src_cute, signal_cute, 0, 0)
    compiled(dst_cute, src_cute, signal_cute, signal_op, peer)

    dev.sync()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    if op.startswith("get"):
        # get pulls the peer's source into the local destination.
        _assert_torch_tensor(dst, peer + 1)
    else:
        # put writes into the next PE, so the local destination holds the previous PE's source.
        _assert_torch_tensor(dst, previous_pe + 1)
    if op.startswith("put_signal"):
        expected_signals = _SCOPE_WARPS if scope == "_warp" else 1
        assert int(signal_var.item()) == expected_signals

    nvshmem.core.free_tensor(src)
    nvshmem.core.free_tensor(dst)
    nvshmem.core.free_tensor(signal_var)


@pytest.mark.mpi
def test_tma_shared_memory_management(nvshmem_init_fini):
    """Compile and execute CuTe tensor wrappers for TMA shared memory."""
    dev = Device()
    if dev.compute_capability < (9, 0):
        pytest.skip("TMA shared-memory APIs require SM90+")

    stream = _nvshmem_stream()
    results = _make_torch_tensor((3, ), "int32", 0)
    results_cute = _cute_from_torch(results)
    minimum_smem = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_MINIMUM)

    @cute.kernel
    def tma_smem_management_kernel(out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        smem_ptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
        smem = cute.make_tensor(smem_ptr, cute.make_layout(minimum_smem // 4))
        if tidx == 0:
            out[0] = nvshmem_cute.ask_smem(nvshmem_cute.SmemAmount.SMEM_RECOMMENDED)
            out[1] = nvshmem_cute.ask_smem(nvshmem_cute.SmemAmount.SMEM_MINIMUM)
            out[2] = nvshmem_cute.ask_smem(nvshmem_cute.SmemAmount.SMEM_BARRIERS_ONLY)

        # Each thread participates in registration and release.
        nvshmem_cute.give_smem(smem)
        cute.arch.sync_threads()
        nvshmem_cute.release_smem()

    @cute.jit
    def tma_smem_management_launcher(out: cute.Tensor):
        tma_smem_management_kernel(out).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
            smem=minimum_smem,
        )

    compiled = _compile_kernel(tma_smem_management_launcher, results_cute)
    compiled(results_cute)
    dev.sync()

    expected = torch.tensor([
        nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_RECOMMENDED),
        minimum_smem,
        nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_BARRIERS_ONLY),
    ],
                            dtype=results.dtype,
                            device=results.device)
    assert torch.equal(results, expected), "CuTe TMA shared-memory wrapper results are incorrect"
    nvshmem.core.free_tensor(results)
