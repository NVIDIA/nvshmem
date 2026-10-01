# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMA-backed RMA through the CuTe DSL wrappers.

NVSHMEM routes put/get through TMA only when ``NVSHMEM_TMA_POLICY`` is
``ENABLE`` (or ``FORCE``) at initialization, the CTA registered shared memory
with ``give_smem``, the peer is P2P reachable, and addresses and sizes are
16-byte aligned.  The policy is read once per process, so these tests live in
their own module and must be launched with the policy already exported.
"""

import os

import pytest
from cuda.core import Device
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute

from test_device_rma import (
    _assert_torch_tensor,
    _compile_kernel,
    _cute_from_torch,
    _make_torch_tensor,
    _nvshmem_stream,
    _torch_dtype,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NVSHMEM_TMA_POLICY", "DISABLE").upper() not in ("ENABLE", "FORCE"),
    reason="TMA RMA tests require NVSHMEM_TMA_POLICY=ENABLE when NVSHMEM initializes",
)

# Four warps satisfy the two-warp minimum of the block-scoped gmem staging path.
_TMA_THREADS = 128
# Large enough to span several staging tiles of the registered shared memory.
_TMA_GMEM_BYTES = 256 * 1024
# Payload that lives in the user-owned part of the registered shared memory.
_TMA_SMEM_PAYLOAD_BYTES = 4096

_TMA_PUT_OPS = ["put", "put_nbi", "put_signal", "put_signal_nbi"]
_TMA_GET_OPS = ["get", "get_nbi"]
_TMA_SCOPES = ["", "_warp", "_block"]
_TMA_SCOPE_IDS = ["thread", "warp", "block"]
_TMA_SPACES = ["gmem", "smem"]
_TMA_DTYPES = ["int8", "float32"]


def _require_tma_device():
    if Device().compute_capability < (9, 0):
        pytest.skip("TMA-backed RMA requires SM90+")


def _nelems(dtype, nbytes):
    return nbytes // _torch_dtype(dtype).itemsize


def _make_tma_rma_kernel(op, scope, space, payload_elems):
    """Build a launcher that issues one TMA-eligible ``op`` at ``scope``.

    NVSHMEM allows only one TMA issuer per CTA at a time, so thread scope
    issues from thread 0, warp scope from warp 0, and block scope from the
    whole CTA.  With ``space == "smem"`` the local operand is a payload placed
    after the NVSHMEM-reserved barrier region of the registered shared memory.
    """
    rma = getattr(nvshmem_cute, f"{op}{scope}")
    is_get = op.startswith("get")
    is_signal = op.startswith("put_signal")
    is_nbi = op.endswith("_nbi")
    minimum_smem = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_MINIMUM)
    payload_offset = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_BARRIERS_ONLY)

    @cute.jit
    def issue(dst, src, signal_var, signal_op, pe):
        if cutlass.const_expr(is_signal):
            rma(dst, src, signal_var, cutlass.Uint64(1), signal_op, pe)
        else:
            rma(dst, src, pe)

    @cute.jit
    def issue_scoped(dst, src, signal_var, signal_op, pe):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(scope == "_block"):
            issue(dst, src, signal_var, signal_op, pe)
        elif cutlass.const_expr(scope == "_warp"):
            if warp == 0:
                issue(dst, src, signal_var, signal_op, pe)
        else:
            if tidx == 0:
                issue(dst, src, signal_var, signal_op, pe)
        if cutlass.const_expr(is_nbi):
            nvshmem_cute.quiet()
        cute.arch.sync_threads()

    @cute.kernel
    def tma_rma(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_op: Int32, pe: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        smem_base = cute.arch.get_dyn_smem(cutlass.Int8, alignment=16)
        smem = cute.make_tensor(smem_base, cute.make_layout(minimum_smem))
        nvshmem_cute.give_smem(smem)

        if cutlass.const_expr(space == "gmem"):
            issue_scoped(dst, src, signal_var, signal_op, pe)
        else:
            payload_ptr = cute.recast_ptr(smem_base + payload_offset, dtype=dst.element_type)
            payload = cute.make_tensor(payload_ptr, cute.make_layout(payload_elems))
            if cutlass.const_expr(is_get):
                # Remote gmem -> local smem, then copy smem out for the host check.
                issue_scoped(payload, src, signal_var, signal_op, pe)
                for i in cutlass.range(tidx, payload_elems, _TMA_THREADS):
                    dst[i] = payload[i]
            else:
                # Stage the source into smem and hand it to the async proxy.
                for i in cutlass.range(tidx, payload_elems, _TMA_THREADS):
                    payload[i] = src[i]
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                issue_scoped(dst, payload, signal_var, signal_op, pe)

        nvshmem_cute.release_smem()

    @cute.jit
    def tma_rma_launcher(dst: cute.Tensor, src: cute.Tensor, signal_var: cute.Tensor, signal_op: Int32, pe: Int32):
        tma_rma(dst, src, signal_var, signal_op, pe).launch(
            grid=[1, 1, 1],
            block=[_TMA_THREADS, 1, 1],
            smem=minimum_smem,
        )

    return tma_rma_launcher


def _run_tma_rma(op, scope, space, dtype):
    _require_tma_device()
    stream = _nvshmem_stream()
    dev = Device()
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    peer = (my_pe + 1) % n_pes
    previous_pe = (my_pe - 1) % n_pes

    nbytes = _TMA_GMEM_BYTES if space == "gmem" else _TMA_SMEM_PAYLOAD_BYTES
    nelems = _nelems(dtype, nbytes)
    src = _make_torch_tensor((nelems, ), dtype, my_pe + 1)
    dst = _make_torch_tensor((nelems, ), dtype, 0)
    signal_var = _make_torch_tensor((1, ), "int64", 0)
    src_cute = _cute_from_torch(src)
    dst_cute = _cute_from_torch(dst)
    signal_cute = _cute_from_torch(signal_var)
    signal_op = nvshmem.core.SignalOp.SIGNAL_ADD

    launcher = _make_tma_rma_kernel(op, scope, space, nelems)
    compiled = _compile_kernel(launcher, dst_cute, src_cute, signal_cute, 0, 0)
    compiled(dst_cute, src_cute, signal_cute, signal_op, peer)

    dev.sync()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    if op.startswith("get"):
        _assert_torch_tensor(dst, peer + 1)
    else:
        _assert_torch_tensor(dst, previous_pe + 1)
    if op.startswith("put_signal"):
        assert int(signal_var.item()) == 1

    nvshmem.core.free_tensor(src)
    nvshmem.core.free_tensor(dst)
    nvshmem.core.free_tensor(signal_var)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", _TMA_DTYPES)
@pytest.mark.parametrize("space", _TMA_SPACES, ids=["from_gmem", "from_smem"])
@pytest.mark.parametrize("scope", _TMA_SCOPES, ids=_TMA_SCOPE_IDS)
@pytest.mark.parametrize("op", _TMA_PUT_OPS)
def test_tma_put(nvshmem_init_fini, op, scope, space, dtype):
    """Put from a global or shared-memory source with TMA enabled."""
    _run_tma_rma(op, scope, space, dtype)


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", _TMA_DTYPES)
@pytest.mark.parametrize("space", _TMA_SPACES, ids=["to_gmem", "to_smem"])
@pytest.mark.parametrize("scope", _TMA_SCOPES, ids=_TMA_SCOPE_IDS)
@pytest.mark.parametrize("op", _TMA_GET_OPS)
def test_tma_get(nvshmem_init_fini, op, scope, space, dtype):
    """Get into a global or shared-memory destination with TMA enabled."""
    _run_tma_rma(op, scope, space, dtype)
