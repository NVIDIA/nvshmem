# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTile source-contract tests and opt-in distributed runtime tests."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cutile_runtime: requires cuda-tile, a CUDA GPU, and multi-PE NVSHMEM",
    )
    config.addinivalue_line(
        "markers",
        "cutile_mathdx_mgmn: requires the MathDX Tile/cuFFTDx linkage provider "
        "and a multi-rank GPU run",
    )


@pytest.fixture(scope="session")
def generated_tilelib(tmp_path_factory):
    """The generated TileLib sources, with the Python modules in ``python``."""
    pytest.importorskip("ast_canopy")
    output_dir = tmp_path_factory.mktemp("generated-tilelib")
    generator = REPO_ROOT / "nvshmem4py" / "build_assets" / "cutile" / "generate_tilelib.py"
    arguments = ["sources", "--output-dir", output_dir, "--python-dir", output_dir / "python"]
    arguments += ["--include-dir", REPO_ROOT / "src" / "include"]
    subprocess.run([sys.executable, generator, *arguments], check=True)
    return output_dir
