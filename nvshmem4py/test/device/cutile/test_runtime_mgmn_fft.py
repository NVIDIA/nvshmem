# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contracts and opt-in execution for the MathDX MGMN FFT integration test."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

RUNNER = Path(__file__).with_name("runtime_mgmn_fft.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("nvshmem4py_runtime_mgmn_fft", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mgmn_runner_uses_only_the_public_tile_api():
    """Keep the MathDX port on the supported NVSHMEM/CuTile surface."""
    source = RUNNER.read_text(encoding="utf-8")
    compile(source, str(RUNNER), "exec")

    assert "from nvshmem.core.device.tile import" in source
    # CuTile owns NVSHMEM library init/finalize for kernels with NVSHMEM calls.
    for removed_option in ("host_library_callback", "lto=True", "simt_num_warps", "library_init("):
        assert removed_option not in source
    assert "nvshmem_put_block(" in source
    assert "communication_tile" in source
    assert "status_output" not in source
    assert "put_status" not in source
    assert "cuda.tile._cext" not in source
    assert "_spy_on_cuLaunchKernel" not in source
    assert "cuda.bindings import driver" not in source
    assert "import gc" in source
    assert "kernels.clear()" in source
    assert source.index("kernels.clear()") < source.index("nvshmem.core.finalize()")


@pytest.mark.mpi(min_size=2)
@pytest.mark.cutile_runtime
@pytest.mark.cutile_mathdx_mgmn
def test_mathdx_mgmn_fft() -> None:
    """Run one distributed float-256 MathDX Tile/cuFFTDx correctness smoke.

    This test is intentionally opt-in. It needs a matching CuTile/TileIR tuple
    with linkage-provider support, MathDX Tile and nvmath distributed wheels,
    an MPI-backed NVSHMEM runtime, and at least two ranks with one GPU per
    local rank. A CI job should launch this test under ``mpirun -np 2`` (or a
    larger multi-node allocation) with ``NVSHMEM_BOOTSTRAP=MPI`` and a
    sufficiently large symmetric heap, for example::

        NVSHMEM4PY_CUTILE_MGMN_REQUIRED=1 NVSHMEM_BOOTSTRAP=MPI \\
        NVSHMEM_SYMMETRIC_SIZE=8G mpirun -np 2 \\
        python -m pytest --with-mpi -q -s \\
        nvshmem4py/test/device/cutile/test_runtime_mgmn_fft.py
    """
    if os.environ.get("NVSHMEM4PY_CUTILE_MGMN_REQUIRED") != "1":
        pytest.skip("set NVSHMEM4PY_CUTILE_MGMN_REQUIRED=1 in a MathDX/CuTile MPI GPU job")

    precision = os.environ.get("NVSHMEM4PY_CUTILE_MGMN_PRECISION", "float")
    fft_size = os.environ.get("NVSHMEM4PY_CUTILE_MGMN_FFT_SIZE", "256")
    warmup = os.environ.get("NVSHMEM4PY_CUTILE_MGMN_WARMUP", "1")
    cycles = os.environ.get("NVSHMEM4PY_CUTILE_MGMN_CYCLES", "1")
    verdict = _load_runner().main([
        "--precision",
        precision,
        "--fft-size",
        fft_size,
        "--warmup",
        warmup,
        "--cycles",
        cycles,
    ])
    if verdict == "SKIP":
        pytest.skip("the FFT needs more GPU memory than some rank has free")
    assert verdict == "PASS"
