# NVSHMEM CUDA-Oxide Rust Performance Tests

This crate ports selected NVSHMEM performance tests to CUDA-Oxide Rust. It uses
the generated host wrappers, includes the generated device bindings in its
kernels, and links Rust LTOIR with NVSHMEM device LTOIR through `nvJitLink`.

The initial coverage includes:

- Host stream APIs: `putmem_nbi_on_stream` and `int_sum_reduce_on_stream`
- Device APIs: `int_put_nbi`, `int_get_nbi`, `int_p`, and `int_g`

## Build

Configure the standalone contrib project as described in the
[project README](../../README.md), with
`NVSHMEM_BUILD_RUST_DEVICE_TESTS=ON`, then build:

```bash
cmake --build build/nvshmem4rust \
  --target test_bindings_rust_cuda_oxide_perf
```

The target is compile-only: it emits NVVM IR through CUDA-Oxide, links a cubin
with NVSHMEM device LTOIR, and stops before launching PEs.

## Run

Launch the resulting executable under MPI or another NVSHMEM bootstrap
launcher:

```bash
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_HOST_LIB_PATH="$PWD/install/lib/libnvshmem_host.so"
export NVSHMEM_RUST_INIT=bootstrap
export NVSHMEM_RUST_REUSE_CUBIN=1
mpirun --bind-to none -np 2 \
  "$PWD/build/nvshmem4rust/generated/cuda_oxide_perf/target/release/nvshmem_cuda_oxide_perf"
```

`NVSHMEM_RUST_REUSE_CUBIN=1` reads the cubin produced by the compile-only
build target. Without it, each direct-run process compiles and links its cubin
in memory, so ranks do not write shared `.ltoir` or `.cubin` artifacts.

Runtime options:

- `NVSHMEM_RUST_PERF_MIN_SIZE`, default `4`
- `NVSHMEM_RUST_PERF_MAX_SIZE`, default `1048576`
- `NVSHMEM_RUST_PERF_STEP`, default `2`
- `NVSHMEM_RUST_PERF_ITERS`, default `100`
- `NVSHMEM_RUST_PERF_WARMUP`, default `10`
- `NVSHMEM_RUST_CUDA_DEVICE`, optional CUDA device ordinal override for the
  calling rank

For a short validation run, use
`NVSHMEM_RUST_PERF_MAX_SIZE=1024 NVSHMEM_RUST_PERF_ITERS=10`.
