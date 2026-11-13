# cuTile NVSHMEM examples

Self-checking cuTile kernels that call NVSHMEM through `nvshmem.core.device.tile`.

| Example | Shows |
| --- | --- |
| `01_rma_put_and_quiet.py` | A blocking and a non-blocking put to the next PE, completed by a device `quiet` |
| `02_jacobi_halo_exchange.py` | A Jacobi solve whose halo rows are put as Tiles, checked against a single-GPU solve |
| `03_put_block_bandwidth.py` | `put_block` bandwidth from Arrays and from Tiles, then `sum_reduce_block` |
| `04_allgather_matmul.py` | An all-gather with `put_signal_nbi_block` feeding a matmul that waits on `signal_wait_until` |

## Running

Build NVSHMEM with CUDA 13 and the cuTile TileLib:
`-DNVSHMEM4PY_BUILD_CUTILE_BINDINGS=ON`, with `NVSHMEM4PY_CUTILE_TILEIRAS` and
`NVSHMEM4PY_CUTILE_TRANSLATOR` naming the `tileiras` and `cuda-tile-translate`
of a CUDA 13.5 toolkit. Install cuTile and its preview package with the
`cutile` extra, `pip install "nvshmem4py-cu13[cutile]"`, and run each example
on at least two MPI ranks:

```bash
mpirun -np 2 python 01_rma_put_and_quiet.py
```

`03_put_block_bandwidth.py --help` lists its options.

## Writing kernels

- Function names carry the NVSHMEM scope (`put_block`, `get_warp`) but not a
  type: the dtype of the data operands, `dst` and `src`, picks the overload.
- Counts come from the operands. A put sends its whole source and a get fills
  its whole destination; slice an Array to send part of it. Arrays must be
  contiguous.
- At block scope, a local operand may be a `ct.Array` or a shared-memory
  `ct.Tile`. Tile operands need peers that NVSHMEM reaches over P2P.
- A call with a local destination returns it: `tile = get_block(tile, src, pe)`.
- Scalars may be Python numbers, scalar Tiles, or the exported enums
  `SignalOp`, `ComparisonType`, and `Teams`.
- Kernels need no NVSHMEM-specific setup: cuTile initializes NVSHMEM for each
  kernel library it loads.
