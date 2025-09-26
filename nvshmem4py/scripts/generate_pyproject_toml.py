#!/usr/bin/env python3
import sys
import re

content = """
[build-system]
requires = [
    "setuptools>=62",
    "setuptools_scm",
    "wheel",
    "Cython>=0.29.24"
]
build-backend = "setuptools.build_meta"

[project]
name = "nvshmem4py"
dynamic = ["version"]
description = "Python bindings for NVSHMEM"
authors = [
    {name = "NVIDIA Corporation"}
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.6.0",
    "cupy-cuda11x"
]

[tool.setuptools]
packages = [
    "nvshmem",
    "nvshmem.bindings",
    "nvshmem.bindings.device",
    "nvshmem.bindings.device.numba",
    "nvshmem.core",
    "nvshmem.core.device",
    "nvshmem.core.device.numba"
]
include-package-data = true

[tool.setuptools.package-data]
"nvshmem.core" = ["*.py"]  # Match Python package name, not filesystem path
"nvshmem.core.device" = ["*.py"]
"nvshmem.core.device.numba" = ["*.py"]
"nvshmem.bindings" = ["*.py", "*.pxd","*.so"]
"nvshmem.bindings._internal" = ["*.py", "*.pxd", "*.so"]
"nvshmem.bindings.device.numba" = ["*.py", "entry_point.h"]


[tool.setuptools_scm]
write_to = "nvshmem/version.py"
# TODO: setuptools_scm is meant to use a git tag for the version
# Once we have releases, we should move to that system
fallback_version = "0.1.2"
version_scheme = "guess-next-dev"

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-mpi",
    "black",
    "isort",
    # TODO: update before release
    "numba-cuda[cu12]==0.20.0",
]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100
"""

def update_pyproject(cuda_ver, nvshmem4py_path):
    req_file = f"{nvshmem4py_path}/requirements_cuda{cuda_ver}.txt"

    # Read requirements file
    with open(req_file) as f:
        deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    dep_block = "dependencies = [\n"
    for dep in deps:
        dep_block += f'    "{dep}",\n'
    dep_block = dep_block.rstrip(",\n") + "\n]\n"

    # Replace dependencies block or append if missing
    pattern = re.compile(r"dependencies\s*=\s*\[.*?\]", re.DOTALL)
    new_content, count = pattern.subn(dep_block, content)

    if count == 0:
        new_content = content + "\n" + dep_block

    print(new_content)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <CUDA_VERSION> <NVSHMEM4Py_PATH>")
        sys.exit(1)
    cuda_version = sys.argv[1]
    nvshmem4py_path = sys.argv[2]
    update_pyproject(cuda_version, nvshmem4py_path)
