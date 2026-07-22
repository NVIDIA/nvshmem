# NVSHMEM

GPU-initiated communication for scalable NVIDIA GPU clusters.

## Introduction

NVSHMEM is an OpenSHMEM-based parallel programming interface that provides a partitioned global
address space across NVIDIA GPUs. Applications use symmetric memory for one-sided transfers,
atomics, signaling, synchronization, and collective operations. NVSHMEM provides host, CUDA kernel,
and CUDA stream interfaces, allowing CPU or GPU threads to initiate communication.

[![license](https://img.shields.io/badge/License-Apache_2.0-blue)](License.txt)
[![docs](https://img.shields.io/badge/Documentation-latest-green)](https://docs.nvidia.com/nvshmem/api/)
[![contributions](https://img.shields.io/badge/Contributions-Accepted-green)](CONTRIBUTING.md)

## Quick Start

### Prerequisites

- A Linux system, NVIDIA driver, and CUDA Toolkit supported by the current
  [NVSHMEM release][release-notes].
- Two NVIDIA data center GPUs.
- A compatible process launcher, such as Open MPI, Hydra, or Slurm.

### Install NVSHMEM

Download a package from the [NVSHMEM product page][product-page] and follow the
[installation guide][install-guide]. Set `NVSHMEM_PREFIX` to the installation directory:

```bash
export NVSHMEM_PREFIX=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### Verify the Installation

Launch two NVSHMEM processing elements (PEs), one per GPU, using a process launcher available on
your system:

```bash
# Open MPI
NVSHMEM_BOOTSTRAP=MPI \
mpirun -np 2 \
  "$NVSHMEM_PREFIX/bin/perftest/device/pt-to-pt/shmem_put_bw" \
  --min_size 8 --max_size 1M --use_smem 0

# Hydra (PMI-1)
NVSHMEM_BOOTSTRAP_PMI=PMI \
mpiexec.hydra -n 2 -ppn 2 \
  "$NVSHMEM_PREFIX/bin/perftest/device/pt-to-pt/shmem_put_bw" \
  --min_size 8 --max_size 1M --use_smem 0

# Slurm (PMI-2)
NVSHMEM_BOOTSTRAP_PMI=PMI-2 \
srun --mpi=pmi2 --nodes=1 --ntasks=2 --gpus-per-task=1 \
  "$NVSHMEM_PREFIX/bin/perftest/device/pt-to-pt/shmem_put_bw" \
  --min_size 8 --max_size 1M --use_smem 0
```

A successful run reports both GPUs, prints put bandwidth for a range of message sizes, and exits
with status zero. If initialization fails, rerun with `NVSHMEM_DEBUG=INFO` and consult the
[launching guide][launching-guide]. For further help, use the channels in
[Support and Community](#support-and-community).

## Build from Source

See the [installation guide][install-guide] for source prerequisites and build options. A standard
CMake build is:

```bash
git clone https://github.com/NVIDIA/nvshmem.git
cd nvshmem

export NVSHMEM_PREFIX="$PWD/install"

cmake -S . -B build \
  -DNVSHMEM_PREFIX="$NVSHMEM_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$NVSHMEM_PREFIX"
cmake --build build --parallel
cmake --install build
```

The development branch may be newer than the latest published release and its documentation. For
a release-aligned build, check out the corresponding branch or tag before running CMake.

## Use NVSHMEM

Installed packages provide imported CMake targets for the host and device libraries:

```cmake
find_package(NVSHMEM REQUIRED CONFIG)

target_link_libraries(my_target PRIVATE
  nvshmem::nvshmem_host
  nvshmem::nvshmem_device
)
```

Applications that use only host-side or stream-ordered APIs can link only the host target. See the
[Using NVSHMEM guide](https://docs.nvidia.com/nvshmem/api/using.html) for the programming model,
application lifecycle, compilation, and launch requirements. The
[put-block example](examples/put-block.cu) provides a complete CUDA program.

Python users should continue with the [NVSHMEM4Py quick start](nvshmem4py/README.md#quick-start).

## Documentation

- [API reference](https://docs.nvidia.com/nvshmem/api/) and source [examples](examples/)
- [Installation guide][install-guide] and [launching guide][launching-guide]
- [Release notes][release-notes] for supported platforms, compatibility, and known issues
- [Environment variables](https://docs.nvidia.com/nvshmem/api/gen/env.html) and
  [best practices][best-practices]

## Support and Community

- Report reproducible bugs, request features, or ask questions through
  [GitHub Issues](https://github.com/NVIDIA/nvshmem/issues).
- Contact the maintainers at [nvshmem@nvidia.com](mailto:nvshmem@nvidia.com).

## Contributing

See the [Contributing Guide](CONTRIBUTING.md) before opening a pull request. Commits require a
Developer Certificate of Origin (DCO) sign-off (`git commit -s`).

## License

NVSHMEM is licensed under the [Apache License 2.0](License.txt).

[best-practices]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/index.html
[install-guide]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html
[launching-guide]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#launching-nvshmem-programs
[product-page]: https://developer.nvidia.com/nvshmem
[release-notes]: https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html
