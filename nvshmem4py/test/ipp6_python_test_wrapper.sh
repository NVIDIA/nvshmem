#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stage the large binary package and Python install work off the shared CI
# filesystem before delegating to the normal nvshmem4py runner.  This wrapper
# runs inside the Slurm allocation so /tmp is local to the GPU node.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 CI_PROJECT_DIR TEST_SUITE" >&2
    exit 2
fi

CI_PROJECT_HOME="$1"
TEST_SUITE="$2"
# The Python-only build publishes these separately from libnvshmem*.tgz.
WHEEL_DIST_DIR="$CI_PROJECT_HOME/nvshmem_pkg/build/dist"

shopt -s nullglob
packages=("$CI_PROJECT_HOME"/libnvshmem*.tgz)
if [ "${#packages[@]}" -ne 1 ]; then
    echo "Expected exactly one binary package under $CI_PROJECT_HOME" >&2
    exit 1
fi

if ! compgen -G "$WHEEL_DIST_DIR/nvshmem4py_cu*.whl" > /dev/null; then
    echo "Expected nvshmem4py wheel artifacts under $WHEEL_DIST_DIR" >&2
    ls -la "$WHEEL_DIST_DIR" 2>/dev/null || true
    exit 1
fi

if [ "${NVSHMEM4PY_LOCAL_SCRATCH:-0}" = "1" ]; then
    scratch_base="${SLURM_TMPDIR:-/tmp}"
    test_workdir="$(mktemp -d "${scratch_base%/}/nvshmem4py-${SLURM_JOB_ID:-job}-XXXXXX")"
    trap 'rm -rf -- "$test_workdir"' EXIT
    package_dir="$test_workdir/nvshmem_pkg"
    local_wheel_dir="$test_workdir/wheels"
    export VENV_DIR="$test_workdir/venv"
    export PIP_CACHE_DIR="$test_workdir/pip-cache"
    mkdir -p "$package_dir" "$local_wheel_dir" "$PIP_CACHE_DIR"
    cp -f -- "$WHEEL_DIST_DIR"/nvshmem4py_cu*.whl "$local_wheel_dir"/
    export NVSHMEM4PY_DIST_DIR="$local_wheel_dir"
    echo "Using node-local scratch: $test_workdir"
    df -h "$scratch_base"
else
    # Multi-node jobs need the environment at an identical shared path for
    # remote mpirun ranks.  They retain the existing shared-package behavior.
    package_dir="$CI_PROJECT_HOME/nvshmem_pkg"
    export VENV_DIR="${VENV_DIR:-$CI_PROJECT_HOME/nvshmem4py_test_venv}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/nvshmem4py-pip-cache-${SLURM_JOB_ID:-job}}"
    export NVSHMEM4PY_DIST_DIR="$WHEEL_DIST_DIR"
    mkdir -p "$package_dir" "$PIP_CACHE_DIR"
fi

tar -xf "${packages[0]}" -C "$package_dir" --strip-components=1

export NVSHMEM_HOME="$package_dir"
export NVSHMEM_PREFIX="$package_dir"
export NVSHMEM4PY_PROJECT_DIR="$CI_PROJECT_HOME"

# Do not exec: the EXIT trap must remove the per-allocation local scratch.
bash "$CI_PROJECT_HOME/nvshmem4py/test/run_tests.sh" "$TEST_SUITE"
