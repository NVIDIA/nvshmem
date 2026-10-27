# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for building and packaging the cuTile TileLib."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_checkout_contains_generator_policy_not_generated_tilelib_sources():
    """The 3k-symbol MLIR/CUDA artifacts must exist only in the build tree."""
    generated_source_suffixes = (".mlir", ".mlir.in", ".cu")
    checked_in_artifacts = [
        path for path in (REPO_ROOT / "nvshmem4py" / "build_assets" / "cutile").rglob("*")
        if path.is_file() and path.name.endswith(generated_source_suffixes)
    ]
    assert checked_in_artifacts == []


def test_ungenerated_package_names_the_build_option(tmp_path):
    """A wheel built without the TileLib ships the package without its generated modules."""
    package = tmp_path / "cutile_without_modules"
    package.mkdir()
    shutil.copy(REPO_ROOT / "nvshmem4py" / "nvshmem" / "core" / "device" / "tile" / "__init__.py",
                package / "__init__.py")
    spec = importlib.util.spec_from_file_location(package.name, package / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package.name] = module
    try:
        with pytest.raises(ImportError, match="NVSHMEM4PY_BUILD_CUTILE_BINDINGS=ON"):
            spec.loader.exec_module(module)
    finally:
        del sys.modules[package.name]


def test_wheel_metadata_packages_the_tilelib_and_cutile_extra():
    generator = REPO_ROOT / "nvshmem4py" / "scripts" / "generate_pyproject_toml.py"
    metadata = subprocess.run(
        [sys.executable, str(generator), "13", str(REPO_ROOT / "nvshmem4py")],
        check=True,
        capture_output=True,
        text=True).stdout
    for package in ("nvshmem.bindings.device.tile", "nvshmem.core.device.tile"):
        assert f'    "{package}"' in metadata
    assert 'cutile = [\n    "cuda-tile",\n    "cuda-tile-preview",\n]' in metadata
