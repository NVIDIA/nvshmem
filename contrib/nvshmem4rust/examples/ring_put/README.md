# CUDA-Oxide NVSHMEM Ring-Put Example

A minimal Rust program in which a CUDA-Oxide kernel calls
`nvshmem_int_put` to write into the next PE's symmetric buffer. Each PE fills
`src` with `100 + my_pe`, launches the kernel, and checks that `dst` holds the
values written by its left neighbour.

The kernel and the host steps are in [`src/main.rs`](src/main.rs). The
host-to-symmetric copies and the raw kernel launch are in
[`src/helpers.rs`](src/helpers.rs).

## Build

Configure `nvshmem4rust` with `NVSHMEM_BUILD_RUST_DEVICE_TESTS=ON`, as described
in the [nvshmem4rust README](../../README.md#build-and-run-the-tests), then
build the example:

```bash
cmake --build build/nvshmem4rust --target example_rust_ring_put
```

The target emits the kernel's NVVM IR with `cargo oxide`, links it with
`NVSHMEM_RUST_TEST_DEVICE_LTOIR` into `nvshmem_ring_put.cubin`, and builds the
host executable without running it.

## Run

Launch one process per GPU with a supported NVSHMEM bootstrap. For example,
with Slurm PMI-2 on two nodes:

```bash
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_HOST_LIB_PATH="$PWD/install/lib/libnvshmem_host.so"
export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI-2
export NVSHMEM_RUST_REUSE_CUBIN=1
srun -N2 --ntasks-per-node=1 --mpi=pmi2 \
  "$PWD/build/nvshmem4rust/generated/ring_put/target/release/nvshmem_ring_put"
```

Each PE prints a line such as:

```text
PE 0/2: received 1024 x 101 from PE 1 OK
```

`NVSHMEM_RUST_INIT` and `NVSHMEM_RUST_CUDA_DEVICE` work as described in the
[smoke-test README](../../tests/cuda_oxide_smoke/README.md#runtime-settings).
