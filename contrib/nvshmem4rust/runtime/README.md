# NVSHMEM Rust Host Runtime

This `nvshmem` crate packages reusable host-side NVSHMEM runtime glue. It
includes the generated `libnvshmem_host` Rust FFI bindings under `sys` and
provides small wrappers for the pieces every program otherwise has to rewrite:

- `NvshmemRuntime` initializes NVSHMEM through the public hostlib API with an
  explicit `InitMethod`.
- `SymmetricBuffer<T>` allocates and frees raw symmetric NVSHMEM memory.
- `sys` exposes the generated raw host FFI with exact `nvshmem_*` and
  `nvshmemx_*` names. `bindings` remains as a compatibility alias for `sys`.
- The crate root re-exports generated prefix-stripped wrappers such as
  `my_pe`, `barrier_all`, and `int_put_on_stream`.
- `cumodule_init(&module)` and `cumodule_finalize(&module)` register
  CUDA-Oxide modules without callers having to extract or cast raw `CUmodule`
  handles.

This crate may depend on CUDA-linked NVSHMEM libraries or CUDA Rust crates
internally when NVSHMEM needs them, but it should not expose CUDA utility APIs
as part of its public surface. It does not select CUDA devices, create CUDA
contexts, own CUDA modules, or copy memory. CUDA-Oxide users should include the
generated `nvshmem_device_cuda_oxide.rs` file from device code and link their
Rust LTOIR with NVSHMEM's shipped device LTOIR or LTOIR fatbin via `nvJitLink`.

Users are also responsible for registering each loaded CUDA-Oxide module before
launching kernels that use NVSHMEM device state, and finalizing it before the
module is dropped:

```rust
let status = unsafe { nvshmem::cumodule_init(&module) };
assert_eq!(status, 0);

// launch CUDA-Oxide kernels that call NVSHMEM device functions

let status = unsafe { nvshmem::cumodule_finalize(&module) };
assert_eq!(status, 0);
```

The raw pointer wrappers are still available as
`nvshmem::api::raw_cumodule_init(...)` and
`nvshmem::api::raw_cumodule_finalize(...)`, with exact C names under `sys`.

Cargo builds need to find the host library. Set `NVSHMEM_HOST_LIB_DIR` to the
directory containing `libnvshmem_host.so`. The generated `Cargo.toml` points
`cuda-core` at the `NVSHMEM_CUDA_OXIDE_ROOT` configured by CMake. At runtime,
the dynamic loader must resolve NVSHMEM and its dependencies through
`LD_LIBRARY_PATH`, an rpath, or a system installation.
`NVSHMEM_HOST_LIB_PATH` may additionally select the exact host-library path
that the crate reopens with `RTLD_GLOBAL`.
