# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the examples and runtime scripts on two PEs; each checks its own results.

Opt in with ``NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED=1`` on a node with two GPUs.
``NVSHMEM4PY_CUTILE_MPIRUN`` replaces the launcher, ``mpirun -np 2``.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2] / "examples" / "cutile"
SCRIPTS = [*sorted(EXAMPLES.glob("0*.py")), HERE / "runtime_tile_rma.py", HERE / "runtime_pointer_views.py"]


@pytest.mark.cutile_runtime
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_runs_on_two_pes(script):
    if os.environ.get("NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED") != "1":
        pytest.skip("set NVSHMEM4PY_CUTILE_RUNTIME_REQUIRED=1 on a node with two GPUs")
    launcher = shlex.split(os.environ.get("NVSHMEM4PY_CUTILE_MPIRUN", "mpirun -np 2"))
    subprocess.run([*launcher, sys.executable, str(script)], cwd=script.parent, check=True)
