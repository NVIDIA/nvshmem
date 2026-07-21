# NVSHMEM Rust Host Runtime

This `nvshmem` crate packages reusable host-side NVSHMEM runtime glue. It
includes the generated `libnvshmem_host` Rust FFI bindings under `sys` and
provides small wrappers for the pieces every program otherwise has to rewrite:

- `NvshmemRuntime` initializes the process-global NVSHMEM host state through
  the public hostlib API with an explicit `InitMethod`.
- `SymmetricBuffer<T>::new(&runtime, ...)` allocates and frees raw symmetric
  NVSHMEM memory while retaining the runtime until the allocation is freed.
- `sys` exposes the generated raw host FFI with exact `nvshmem_*` and
  `nvshmemx_*` names. `bindings` remains as a compatibility alias for `sys`.
- The crate root re-exports generated prefix-stripped aliases and safe wrappers
  such as `my_pe`, `barrier_all`, and `int_put_on_stream`.
- `NvshmemRuntime::register_module(&module)` returns a guard that keeps the
  runtime and CUDA-Oxide module valid through NVSHMEM finalization.

This crate may depend on CUDA-linked NVSHMEM libraries or CUDA Rust crates
internally when NVSHMEM needs them, but it should not expose CUDA utility APIs
as part of its public surface. It does not select CUDA devices, create CUDA
contexts, own CUDA modules, or copy memory. CUDA-Oxide users should include the
generated `nvshmem_device_cuda_oxide.rs` file from device code and link their
Rust LTOIR with NVSHMEM's shipped device LTOIR or LTOIR fatbin via `nvJitLink`.

Users must register each loaded CUDA-Oxide module before launching kernels that
use NVSHMEM device state. The registration guard finalizes it before the module
or runtime can be dropped:

```rust
let _registration = unsafe { runtime.register_module(&module) }?;

// launch CUDA-Oxide kernels that call NVSHMEM device functions
```

The raw pointer wrappers are still available as
`nvshmem::api::raw_cumodule_init(...)` and
`nvshmem::api::raw_cumodule_finalize(...)`, with exact C names under `sys`.

`NvshmemRuntime::init` returns another handle to an already initialized
process-global runtime. NVSHMEM finalizes after the last runtime handle and
all symmetric buffers tied to it have been dropped.

MPI initialization requires a caller-provided pointer to an initialized
`MPI_Comm`. Construct `InitMethod::MpiComm` with
`unsafe { MpiComm::from_raw(...) }`; the caller is responsible for using the
same MPI implementation as NVSHMEM and keeping the communicator valid while
the runtime is live.

Cargo builds need to find the host library. Set `NVSHMEM_HOST_LIB_DIR` to the
directory containing `libnvshmem_host.so`. The generated `Cargo.toml` points
`cuda-core` at the `NVSHMEM_CUDA_OXIDE_ROOT` configured by CMake. At runtime,
the dynamic loader must resolve NVSHMEM and its dependencies through
`LD_LIBRARY_PATH`, an rpath, or a system installation.
`NVSHMEM_HOST_LIB_PATH` may additionally select the exact host-library path
that the crate reopens with `RTLD_GLOBAL`.
