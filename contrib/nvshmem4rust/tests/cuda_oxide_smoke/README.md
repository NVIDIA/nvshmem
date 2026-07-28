# CUDA-Oxide NVSHMEM Smoke Tests

This CUDA-Oxide test crate exercises the generated NVSHMEM Rust host and device
bindings with raw symmetric pointers.

The kernels cover:

- `nvshmem_my_pe` and `nvshmem_n_pes`
- `nvshmem_ptr` for the local PE
- `nvshmem_int_p` and `nvshmem_int_g`
- `nvshmem_int_put` and `nvshmem_int_get`
- `nvshmemx_signal_op` and `nvshmem_signal_wait_until`
- `nvshmem_int_atomic_fetch_add`
- Ring-style remote put/get with more than one PE
- Generated host bindings for stream RMA, put-with-signal, collectives, and
  flush APIs

## Build

From the NVSHMEM repository root:

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
```

The target configures a build-tree Cargo crate, emits CUDA-Oxide NVVM IR,
links it with `NVSHMEM_RUST_TEST_DEVICE_LTOIR`, initializes NVSHMEM through the
public host API, registers the linked CUDA module, and launches the kernels.
The direct CMake target selects the single-PE `uid` initialization path so it
does not require PMI launcher state.

Each generated test workspace starts from the selected CUDA-Oxide checkout's
`Cargo.lock` and completes its local path dependencies offline. This keeps its
dependency resolution aligned with that checkout.

If the normal NVSHMEM build layout is unavailable, provide these paths
explicitly:

```bash
-DNVSHMEM_HOST_LIB_DIR=/path/to/nvshmem/lib
-DNVSHMEM_RUST_TEST_DEVICE_LTOIR=/path/to/libnvshmem_device.ltoir.fatbin
```

## Runtime settings

`NVSHMEM_RUST_INIT=bootstrap|uid` selects the initialization method. The
default, `bootstrap`, lets NVSHMEM choose a bootstrap from the launch
environment, including a supported MPI launcher. `uid` is a single-PE
unique-ID path. Programs that own an initialized MPI communicator can instead
use `InitMethod::MpiComm` directly.

Device selection defaults to the launcher-local rank filtered by
`CUDA_OXIDE_TARGET`. Set `NVSHMEM_RUST_CUDA_DEVICE` to force a CUDA device
ordinal for a rank.

After the build target produces `nvshmem_cuda_oxide_smoke.cubin`, direct MPI
launches can set `NVSHMEM_RUST_REUSE_CUBIN=1` to reuse the linked cubin instead
of rebuilding it in every rank. Without it, each direct-run process compiles
and links its cubin in memory, so ranks do not write shared `.ltoir` or
`.cubin` artifacts:

```bash
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_HOST_LIB_PATH="$PWD/install/lib/libnvshmem_host.so"
export NVSHMEM_RUST_INIT=bootstrap
export NVSHMEM_RUST_REUSE_CUBIN=1
mpirun --bind-to none -np 2 \
  "$PWD/build/nvshmem4rust/generated/cuda_oxide_smoke/target/release/nvshmem_cuda_oxide_smoke"
```
