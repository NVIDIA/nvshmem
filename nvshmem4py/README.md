# NVSHMEM4Py

NVSHMEM4Py exposes NVSHMEM host, device, and on-stream interfaces to Python, with interoperability
for CUDA Python, Numba, PyTorch, CuPy, and CuTe DSL.

## Quick Start

Follow the [NVSHMEM4Py installation guide][install-guide] to install the package matching the CUDA
major version. This example requires two supported GPUs on one Linux node and `mpi4py` built for the
active MPI installation.

From the repository root:

```bash
mpirun -np 2 python nvshmem4py/examples/hello.py
```

The output includes one line from each PE, in either order:

```text
Hello from PE 0 of 2
Hello from PE 1 of 2
```

The [example](examples/hello.py) selects one GPU per local MPI rank and initializes NVSHMEM from
`MPI_COMM_WORLD`. See the main [NVSHMEM quick start](../README.md#quick-start) for platform
requirements and the [launching guide][launching-guide] for other launchers.

## Building Wheels

Wheel targets are part of the NVSHMEM CMake build. See the
[NVSHMEM4Py CMake configuration](CMakeLists.txt) for supported options and defaults. For example,
to build a Python 3.12 wheel for CUDA 12:

```bash
cmake -S . -B build \
  -DNVSHMEM4PY_BUILD_ALL_WHEELS=OFF \
  -DNVSHMEM4PY_PYTHON_VERSIONS=3.12 \
  -DNVSHMEM4PY_CUDA_VERSIONS=12
cmake --build build --target build_nvshmem4py_wheel_cu12_3.12
```

The wheel is written to `build/dist/` and can be installed as described in the
[NVSHMEM4Py installation guide][install-guide].

## Examples

The [examples](examples/) cover host, device, on-stream, and framework-interoperability workflows.
See the [main NVSHMEM README](../README.md) for additional documentation and project information.

[install-guide]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem4py-install-proc.html
[launching-guide]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#launching-nvshmem-programs
