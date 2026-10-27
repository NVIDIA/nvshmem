# NCCL AllReduce from Rust

A companion to the [ring-put example](../ring_put/README.md): the same "talk
between GPUs from Rust" idea, using NCCL's host API instead of NVSHMEM. It uses
the [`cudarc`](https://crates.io/crates/cudarc) crate's NCCL bindings, so it
does not depend on the generated nvshmem4rust bindings and is not built by the
nvshmem4rust CMake project.

Each rank fills a buffer with `rank + 1`, enqueues `ncclAllReduce` (sum) on a
CUDA stream, and checks that every element equals `1 + 2 + ... + world`.

## Build

```bash
cargo build --release
```

`cudarc` loads `libcuda` and `libnccl.so.2` at run time; put NCCL's `lib`
directory on `LD_LIBRARY_PATH`.

## Run

Rank and world size come from Slurm (`SLURM_PROCID`, `SLURM_NTASKS`). Rank 0
writes the NCCL unique ID to `NCCL_ID_DIR/nccl-id.$SLURM_JOB_ID`, so
`NCCL_ID_DIR` must be on a filesystem that every node can see:

```bash
export LD_LIBRARY_PATH=/path/to/nccl/lib:${LD_LIBRARY_PATH:-}
export NCCL_ID_DIR=/shared/scratch
srun -N2 --ntasks-per-node=1 target/release/nccl_allreduce
```

Each rank prints a line such as:

```text
[node01] rank 0/2: sum = 3 (expected 3) OK
```
