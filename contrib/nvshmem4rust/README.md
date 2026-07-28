# nvshmem4rust

`nvshmem4rust` generates Rust bindings for NVSHMEM's public host and device
APIs. It provides:

- Raw host FFI declarations linked against `libnvshmem_host`
- Device declarations that CUDA-Oxide kernels can call
- A small `nvshmem` host runtime crate with initialization, symmetric-memory,
  and CUDA-module registration helpers
- CUDA-Oxide smoke and performance test programs

This is an experimental, contributor-maintained project. It is not part of the
NVSHMEM core release quality commitment.

## Prerequisites

Generating the bindings requires:

- CMake 3.21 or newer
- Python 3 with `venv` support
- An NVSHMEM source checkout or installed NVSHMEM tree with public headers
- A CUDA Toolkit installation
- A [CUDA-Oxide](https://github.com/NVlabs/cuda-oxide) checkout for device
  bindings and the generated runtime crate's `cuda-core` dependency
- Network access to install Numbast and its Python dependencies from PyPI

Building the generated runtime crate additionally requires:

- Rust and Cargo with Edition 2024 support
- A built NVSHMEM host library

The optional tests also require the CUDA-Oxide `cargo-oxide` executable and an
NVSHMEM device LTOIR file or LTOIR fat binary.

See [ThirdPartyNotices.txt](ThirdPartyNotices.txt) for dependency and license
information.

## Generate the bindings

From the NVSHMEM repository root, configure `nvshmem4rust` as an independent
CMake project:

```bash
cmake -S contrib/nvshmem4rust -B build/nvshmem4rust \
  -DNVSHMEM_SOURCE_DIR="$PWD" \
  -DNVSHMEM_HOME="$PWD/install" \
  -DNVSHMEM_CUDA_OXIDE_ROOT=/path/to/cuda-oxide \
  -DNVSHMEM_CUDA_HOME=/path/to/cuda

cmake --build build/nvshmem4rust --target build_bindings_rust
```

`NVSHMEM_HOME` supplies installed public headers. If NVSHMEM is not installed,
omit it or set `NVSHMEM_INCLUDE_DIR="$PWD/src/include"` explicitly.

By default, generated files are written under
`build/nvshmem4rust/generated/`:

- `nvshmem_device_cuda_oxide.rs` — CUDA-Oxide device declarations
- `nvshmem_host.rs` — raw host declarations
- `nvshmem_host_api.rs` — prefix-stripped host API aliases and safe wrappers
- `nvshmem_host_runtime/` — a build-tree Cargo package named `nvshmem`

Set `NVSHMEM_RUST_BINDINGS_OUTPUT_DIR` to override the output directory.

### Generate raw host bindings only

For raw host FFI without the CUDA-Oxide device bindings or runtime crate, omit
`NVSHMEM_CUDA_OXIDE_ROOT` and configure with
`NVSHMEM_BUILD_RUST_HOST_ONLY=ON`:

```bash
cmake -S contrib/nvshmem4rust -B build/nvshmem4rust-host \
  -DNVSHMEM_SOURCE_DIR="$PWD" \
  -DNVSHMEM_HOME="$PWD/install" \
  -DNVSHMEM_CUDA_HOME=/path/to/cuda \
  -DNVSHMEM_BUILD_RUST_HOST_ONLY=ON

cmake --build build/nvshmem4rust-host --target build_bindings_rust
```

This generates only `nvshmem_host.rs` and `nvshmem_host_api.rs`. A CUDA Toolkit
is still required because the NVSHMEM host headers include CUDA types.

## Use the generated bindings

CUDA-Oxide device code includes `nvshmem_device_cuda_oxide.rs` and links the
resulting kernel LTOIR with NVSHMEM's device LTOIR through `nvJitLink`. Host
programs can depend on the generated `nvshmem_host_runtime` crate, or use the
raw declarations directly.

The runtime crate exposes exact C ABI names under `nvshmem::sys` and
prefix-stripped aliases and safe wrappers such as `nvshmem::my_pe()` and
`nvshmem::barrier_all()`. CUDA-Oxide users should keep the guard returned by
`unsafe { runtime.register_module(&module) }` alive while kernels call NVSHMEM;
it finalizes the module registration before the runtime or module can drop.
Call `registration.finalize()?` after synchronizing when the finalizer status
must be reported.

At Cargo build time, set `NVSHMEM_HOST_LIB_DIR` to the directory containing
`libnvshmem_host.so`. At runtime, the dynamic loader must be able to resolve
NVSHMEM and its dependencies through `LD_LIBRARY_PATH`, an rpath, or a system
installation. `NVSHMEM_HOST_LIB_PATH` can additionally select the exact library
that the runtime reopens with global symbol visibility.

## Build and run the tests

Configure the test targets against an existing NVSHMEM build:

```bash
cmake -S contrib/nvshmem4rust -B build/nvshmem4rust \
  -DNVSHMEM_SOURCE_DIR="$PWD" \
  -DNVSHMEM_HOME="$PWD/install" \
  -DNVSHMEM_BUILD_DIR="$PWD/build" \
  -DNVSHMEM_BUILD_RUST_DEVICE_TESTS=ON \
  -DNVSHMEM_CUDA_OXIDE_ROOT=/path/to/cuda-oxide \
  -DNVSHMEM_CARGO_OXIDE_EXECUTABLE=/path/to/cargo-oxide \
  -DNVSHMEM_CUDA_HOME=/path/to/cuda \
  -DNVSHMEM_RUST_TEST_ARCH=sm_90

cmake --build build/nvshmem4rust --target test_bindings_rust_cuda_oxide
cmake --build build/nvshmem4rust --target test_bindings_rust_cuda_oxide_perf
```

The smoke target runs directly in single-PE unique-ID mode. Use a supported
NVSHMEM launcher and set `NVSHMEM_RUST_INIT=bootstrap` when running the
generated test executable with multiple PEs.

If the normal NVSHMEM build layout is unavailable, set
`NVSHMEM_HOST_LIB_DIR` and `NVSHMEM_RUST_TEST_DEVICE_LTOIR` explicitly. The
performance target compiles the executable; launch it with the appropriate
NVSHMEM bootstrap or MPI launcher. See the
[smoke-test](tests/cuda_oxide_smoke/README.md) and
[performance-test](tests/cuda_oxide_perf/README.md) documentation for runtime
settings.

## Layout

- `generator/` contains the Numbast configuration and Rust emitter.
- `runtime/` contains the template for the generated host runtime crate.
- `tests/` contains CUDA-Oxide smoke and performance programs plus their
  shared support crate.
- `cmake/` contains the standalone generation and test targets.

## Known limitations

- The generated API is intentionally low-level and does not yet follow Rust API
  stability or semantic-versioning guarantees.
- CUDA-Oxide is experimental, and its local crate layout is part of the current
  test integration. The generated runtime Cargo manifest is therefore a
  build-tree artifact tied to the configured CUDA-Oxide checkout.
- Cross-version compatibility is not supported. The bindings, host library,
  and device LTOIR must be generated or built from the same NVSHMEM revision.

## Maintainers

- `benjaming@nvidia.com`

## License

This contribution is licensed under the repository's
[Apache 2.0 license](../../License.txt).
