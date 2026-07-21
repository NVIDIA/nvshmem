/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::missing_safety_doc)]

use cuda_core::{CudaContext, CudaModule, DriverError, IntoResult, LaunchConfig, memory, sys};
use cuda_device::{kernel, thread};
use libnvvm_sys::{LibNvvm, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use nvshmem::{InitMethod, Runtime, SymmetricBuffer};
use std::env;
use std::error::Error;
use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod nvshmem_device {
    include!(env!("NVSHMEM_RUST_BINDINGS"));
}

#[kernel]
pub fn nvshmem_query_smoke(out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            *out.add(0) = nvshmem_device::nvshmem_my_pe();
            *out.add(1) = nvshmem_device::nvshmem_n_pes();
        }
    }
}

#[kernel]
pub fn nvshmem_ptr_smoke(buf: *mut i32, out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            let peer_ptr =
                nvshmem_device::nvshmem_ptr(buf.cast_const().cast::<core::ffi::c_void>(), pe)
                    .cast::<i32>();
            *out = if peer_ptr == buf { 1 } else { 0 };
        }
    }
}

#[kernel]
pub fn nvshmem_p_g_smoke(buf: *mut i32, out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            let value = 17 + pe;
            nvshmem_device::nvshmem_int_p(buf, value, pe);
            nvshmem_device::nvshmem_quiet();
            *out = nvshmem_device::nvshmem_int_g(buf.cast_const(), pe);
        }
    }
}

#[kernel]
pub fn nvshmem_put_get_smoke(dst: *mut i32, src: *mut i32, tmp: *mut i32, out: *mut i32, len: u64) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            nvshmem_device::nvshmem_int_put(dst, src.cast_const(), len, pe);
            nvshmem_device::nvshmem_quiet();
            nvshmem_device::nvshmem_int_get(tmp, dst.cast_const(), len, pe);

            let mut ok = 1;
            let mut idx = 0usize;
            while idx < len as usize {
                if *dst.add(idx) != *src.add(idx) || *tmp.add(idx) != *src.add(idx) {
                    ok = 0;
                }
                idx += 1;
            }
            *out = ok;
        }
    }
}

#[kernel]
pub fn nvshmem_signal_wait_smoke(signal: *mut u64, out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            nvshmem_device::nvshmemx_signal_op(signal, 1, nvshmem_device::NVSHMEM_SIGNAL_SET, pe);
            let observed = nvshmem_device::nvshmem_signal_wait_until(
                signal,
                nvshmem_device::NVSHMEM_CMP_EQ,
                1,
            );
            *out = if observed == 1 { 1 } else { 0 };
        }
    }
}

#[kernel]
pub fn nvshmem_atomic_fetch_add_smoke(buf: *mut i32, out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            let old = nvshmem_device::nvshmem_int_atomic_fetch_add(buf, 5, pe);
            nvshmem_device::nvshmem_quiet();
            let now = nvshmem_device::nvshmem_int_g(buf.cast_const(), pe);
            *out.add(0) = old;
            *out.add(1) = now;
        }
    }
}

#[kernel]
pub fn nvshmem_ring_put_smoke(dst: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            let npes = nvshmem_device::nvshmem_n_pes();
            let peer = (pe + 1) % npes;
            nvshmem_device::nvshmem_int_p(dst, pe, peer);
            nvshmem_device::nvshmem_quiet();
        }
    }
}

#[kernel]
pub fn nvshmem_ring_get_smoke(src: *mut i32, out: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        unsafe {
            let pe = nvshmem_device::nvshmem_my_pe();
            let npes = nvshmem_device::nvshmem_n_pes();
            let peer = (pe + 1) % npes;
            *out = nvshmem_device::nvshmem_int_g(src.cast_const(), peer);
        }
    }
}

fn assert_host_stream_bindings_are_linkable() {
    let _ = nvshmem::int_put_on_stream as unsafe fn(*mut i32, *const i32, u64, i32, *mut c_void);
    let _ = nvshmem::int_put_signal_on_stream
        as unsafe fn(*mut i32, *const i32, u64, *mut u64, u64, i32, i32, *mut c_void);
    let _ = nvshmem::int_sum_reduce_on_stream
        as unsafe fn(i32, *mut i32, *const i32, u64, *mut c_void) -> i32;
    let _ = nvshmem::alltoallmem_on_stream
        as unsafe fn(i32, *mut c_void, *const c_void, u64, *mut c_void) -> i32;
    let _ = nvshmem::flush_on_stream as unsafe fn(*mut c_void);
}

fn init_method_from_env() -> Result<InitMethod, Box<dyn Error>> {
    match env::var("NVSHMEM_RUST_INIT")
        .unwrap_or_else(|_| "bootstrap".to_string())
        .as_str()
    {
        "bootstrap" | "env" => Ok(InitMethod::BootstrapEnv),
        "uid" => Ok(InitMethod::single_pe_uid()?),
        other => Err(format!(
            "unsupported NVSHMEM_RUST_INIT={other}; expected bootstrap, env, or uid"
        )
        .into()),
    }
}

fn target_compute_capability(arch: &str) -> Option<(i32, i32)> {
    let digits = arch.strip_prefix("sm_")?;
    if digits.len() < 2 {
        return None;
    }
    let split = digits.len() - 1;
    let major = digits[..split].parse().ok()?;
    let minor = digits[split..].parse().ok()?;
    Some((major, minor))
}

fn copy_to_symmetric<T>(
    ctx: &CudaContext,
    buffer: &SymmetricBuffer<T>,
    values: &[T],
) -> std::result::Result<(), DriverError>
where
    T: Copy,
{
    assert_eq!(values.len(), buffer.len());
    ctx.bind_to_thread()?;
    unsafe {
        memory::memcpy_htod_sync(
            buffer.as_mut_ptr() as sys::CUdeviceptr,
            values.as_ptr(),
            values.len() * size_of::<T>(),
        )
    }
}

fn copy_from_symmetric<T>(
    ctx: &CudaContext,
    buffer: &SymmetricBuffer<T>,
) -> std::result::Result<Vec<T>, DriverError>
where
    T: Copy + Default,
{
    ctx.bind_to_thread()?;
    let mut values = vec![T::default(); buffer.len()];
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buffer.as_mut_ptr() as sys::CUdeviceptr,
            buffer.len() * size_of::<T>(),
            ptr::null_mut(),
        )?;
    }
    ctx.synchronize()?;
    Ok(values)
}

// CUDA-Oxide exposes compute capability from an existing context, but this
// smoke test needs to choose a matching device before creating one. Keep this
// CUDA-specific helper out of the NVSHMEM runtime until it can move into a CUDA
// host utility crate.
fn select_device_ordinal(target_cc: Option<(i32, i32)>) -> Result<usize, Box<dyn Error>> {
    unsafe {
        cuda_core::init(0)?;
    }
    if let Some(ordinal) = forced_device_ordinal()? {
        return Ok(ordinal);
    }

    let mut count = 0;
    unsafe {
        sys::cuDeviceGetCount(&mut count).result()?;
    }
    if count <= 0 {
        return Err("no CUDA devices found".into());
    }

    let local_rank = local_rank_from_env().unwrap_or(0);

    let mut candidates = Vec::new();
    for ordinal in 0..count {
        let cc = device_compute_capability(ordinal)?;
        if target_cc.map_or(true, |target| target == cc) {
            candidates.push(ordinal as usize);
        }
    }

    if candidates.is_empty() {
        Ok(local_rank % count as usize)
    } else {
        Ok(candidates[local_rank % candidates.len()])
    }
}

fn forced_device_ordinal() -> Result<Option<usize>, Box<dyn Error>> {
    match env::var("NVSHMEM_RUST_CUDA_DEVICE") {
        Ok(value) => Ok(Some(value.parse::<usize>().map_err(|err| {
            format!("invalid NVSHMEM_RUST_CUDA_DEVICE={value}: {err}")
        })?)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("invalid NVSHMEM_RUST_CUDA_DEVICE: {err}").into()),
    }
}

fn local_rank_from_env() -> Option<usize> {
    [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    ]
    .iter()
    .find_map(|name| env::var(name).ok())
    .and_then(|value| value.parse::<usize>().ok())
}

fn device_compute_capability(ordinal: i32) -> std::result::Result<(i32, i32), DriverError> {
    let mut device = MaybeUninit::uninit();
    unsafe {
        sys::cuDeviceGet(device.as_mut_ptr(), ordinal).result()?;
    }
    let device = unsafe { device.assume_init() };

    let mut major = MaybeUninit::uninit();
    let mut minor = MaybeUninit::uninit();
    unsafe {
        sys::cuDeviceGetAttribute(
            major.as_mut_ptr(),
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
        .result()?;
        sys::cuDeviceGetAttribute(
            minor.as_mut_ptr(),
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
        .result()?;
        Ok((major.assume_init(), minor.assume_init()))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    assert_host_stream_bindings_are_linkable();

    let arch = env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_90".to_string());
    let cubin = build_cubin(&arch)?;
    if env::var_os("NVSHMEM_RUST_COMPILE_ONLY").is_some() {
        println!("NVSHMEM CUDA-Oxide compile-only smoke passed for {arch}");
        return Ok(());
    }

    let target_cc = target_compute_capability(&arch);
    let device_ordinal = select_device_ordinal(target_cc)?;
    let cc = device_compute_capability(device_ordinal as i32)?;
    eprintln!(
        "NVSHMEM CUDA-Oxide smoke: selected CUDA device {device_ordinal} with compute capability {}.{}",
        cc.0, cc.1
    );
    let ctx = CudaContext::new(device_ordinal)?;
    eprintln!("NVSHMEM CUDA-Oxide smoke: loading linked {arch} cubin");
    let module = ctx.load_module_from_image(&cubin)?;
    let init_method = init_method_from_env()?;
    eprintln!("NVSHMEM CUDA-Oxide smoke: initializing hostlib");
    let runtime = Runtime::init(init_method, device_ordinal as i32)?;
    eprintln!("NVSHMEM CUDA-Oxide smoke: hostlib initialized");
    eprintln!("NVSHMEM CUDA-Oxide smoke: registering CUDA module");
    let _module_registration = unsafe { runtime.register_module(&module) }?;
    eprintln!("NVSHMEM CUDA-Oxide smoke: CUDA module registered");

    let pe = nvshmem::my_pe();
    let npes = nvshmem::n_pes();
    println!("NVSHMEM CUDA-Oxide smoke test: PE {pe}/{npes} on device {device_ordinal}");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    run_query_test(&runtime, &ctx, &module, cfg, pe, npes)?;
    run_ptr_test(&runtime, &ctx, &module, cfg)?;
    run_p_g_test(&runtime, &ctx, &module, cfg, pe)?;
    run_put_get_test(&runtime, &ctx, &module, cfg)?;
    run_signal_wait_test(&runtime, &ctx, &module, cfg)?;
    run_atomic_fetch_add_test(&runtime, &ctx, &module, cfg)?;
    run_ring_put_test(&runtime, &ctx, &module, cfg, pe, npes)?;
    run_ring_get_test(&runtime, &ctx, &module, cfg, pe, npes)?;

    nvshmem::barrier_all();
    ctx.synchronize()?;
    println!("NVSHMEM CUDA-Oxide smoke tests passed on PE {pe}");
    Ok(())
}

fn build_cubin(arch: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let module_name = env!("CARGO_PKG_NAME");
    let cubin_path = manifest_dir.join(format!("{module_name}.cubin"));

    if env::var_os("NVSHMEM_RUST_REUSE_CUBIN").is_some() {
        let cubin = std::fs::read(&cubin_path).map_err(|err| -> Box<dyn Error> {
            format!(
                "failed to read existing cubin at {}: {err}",
                cubin_path.display()
            )
            .into()
        })?;
        return Ok(cubin);
    }

    let ll_path = manifest_dir.join(format!("{module_name}.ll"));
    if !ll_path.exists() {
        return Err(format!(
            "CUDA-Oxide NVVM IR not found at {}. Run with: cargo oxide run --emit-nvvm-ir --arch={arch}",
            ll_path.display()
        )
        .into());
    }

    let write_artifacts = env::var_os("NVSHMEM_RUST_COMPILE_ONLY").is_some()
        || env::var_os("NVSHMEM_RUST_WRITE_ARTIFACTS").is_some();
    let ltoir_path = write_artifacts.then(|| ll_path.with_extension("ltoir"));
    let rust_ltoir = compile_rust_ltoir(&ll_path, arch, ltoir_path.as_deref())?;
    let nvshmem_ltoir = env::var("NVSHMEM_DEVICE_LTOIR")
        .map(PathBuf::from)
        .map_err(|_| {
            "NVSHMEM_DEVICE_LTOIR must point at libnvshmem_device.ltoir.fatbin or a per-arch .ltoir"
        })?;
    let nvshmem_ltoir_bytes = std::fs::read(&nvshmem_ltoir).map_err(|err| -> Box<dyn Error> {
        format!(
            "failed to read NVSHMEM_DEVICE_LTOIR at {}: {err}",
            nvshmem_ltoir.display()
        )
        .into()
    })?;

    let nvj = LibNvJitLink::load()?;
    let arch_opt = format!("-arch={arch}");
    let mut linker = Linker::new(&nvj, &[arch_opt.as_str(), "-lto"])?;
    linker.add(
        InputType::Ltoir,
        &rust_ltoir,
        "nvshmem_cuda_oxide_smoke.ltoir",
    )?;
    linker.add(
        nvshmem_input_type(&nvshmem_ltoir, &nvshmem_ltoir_bytes),
        &nvshmem_ltoir_bytes,
        &nvshmem_ltoir.display().to_string(),
    )?;
    let cubin = linker.finish()?;

    if write_artifacts {
        write_artifact(&cubin_path, &cubin)?;
    }
    Ok(cubin)
}

fn compile_rust_ltoir(
    ll_path: &Path,
    arch: &str,
    output_path: Option<&Path>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let ll_bytes = std::fs::read(ll_path)?;
    let libdevice_path = find_libdevice()?;
    let libdevice = std::fs::read(&libdevice_path)?;

    let nvvm = LibNvvm::load()?;
    let mut program = Program::new(&nvvm)?;
    program.add_module(&libdevice, "libdevice.10.bc")?;
    program.add_module(&ll_bytes, &ll_path.display().to_string())?;

    let compute = if let Some(suffix) = arch.strip_prefix("sm_") {
        format!("compute_{suffix}")
    } else {
        arch.to_string()
    };
    let arch_opt = format!("-arch={compute}");
    let ltoir = program.compile(&[arch_opt.as_str(), "-gen-lto"])?;
    if let Some(output_path) = output_path {
        write_artifact(output_path, &ltoir)?;
    }
    Ok(ltoir)
}

fn write_artifact(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("artifact");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temporary_path =
        path.with_extension(format!("{extension}.{nonce}.{}.tmp", std::process::id()));

    std::fs::write(&temporary_path, bytes)?;
    if let Err(error) = std::fs::rename(&temporary_path, path) {
        let _ = std::fs::remove_file(&temporary_path);
        return Err(format!(
            "failed to atomically publish artifact {}: {error}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn nvshmem_input_type(path: &Path, bytes: &[u8]) -> InputType {
    const LTOIR_MAGIC: [u8; 4] = [0xed, 0x43, 0x4e, 0x7f];
    if bytes.starts_with(&LTOIR_MAGIC) {
        InputType::Ltoir
    } else if path.extension().and_then(|ext| ext.to_str()) == Some("fatbin") {
        InputType::Fatbin
    } else {
        InputType::Ltoir
    }
}

fn find_libdevice() -> Result<PathBuf, Box<dyn Error>> {
    if let Ok(path) = env::var("CUDA_OXIDE_LIBDEVICE") {
        return Ok(PathBuf::from(path));
    }

    let mut roots = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
        if let Ok(root) = env::var(var) {
            roots.push(PathBuf::from(root));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));

    for root in roots {
        let candidate = root.join("nvvm/libdevice/libdevice.10.bc");
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err("could not find libdevice.10.bc; set CUDA_OXIDE_LIBDEVICE or CUDA_HOME".into())
}

fn run_query_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    pe: i32,
    npes: i32,
) -> Result<(), Box<dyn Error>> {
    let out = SymmetricBuffer::<i32>::new(runtime, 2)?;
    copy_to_symmetric(ctx, &out, &[0, 0])?;
    launch_one_arg(module, cfg, "nvshmem_query_smoke", out.as_mut_ptr())?;
    ctx.synchronize()?;
    let values = copy_from_symmetric(ctx, &out)?;
    assert_eq!(values, vec![pe, npes], "device PE query mismatch");
    nvshmem::barrier_all();
    println!("  query: passed");
    Ok(())
}

fn run_ptr_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
) -> Result<(), Box<dyn Error>> {
    let buf = SymmetricBuffer::<i32>::new(runtime, 1)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &out, &[0])?;
    launch_two_args(
        module,
        cfg,
        "nvshmem_ptr_smoke",
        buf.as_mut_ptr(),
        out.as_mut_ptr(),
    )?;
    ctx.synchronize()?;
    assert_eq!(
        copy_from_symmetric(ctx, &out)?[0],
        1,
        "nvshmem_ptr(self) returned a different pointer"
    );
    nvshmem::barrier_all();
    println!("  ptr: passed");
    Ok(())
}

fn run_p_g_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    pe: i32,
) -> Result<(), Box<dyn Error>> {
    let buf = SymmetricBuffer::<i32>::new(runtime, 1)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &buf, &[0])?;
    copy_to_symmetric(ctx, &out, &[0])?;
    launch_two_args(
        module,
        cfg,
        "nvshmem_p_g_smoke",
        buf.as_mut_ptr(),
        out.as_mut_ptr(),
    )?;
    ctx.synchronize()?;
    assert_eq!(
        copy_from_symmetric(ctx, &out)?[0],
        17 + pe,
        "p/g self test mismatch"
    );
    nvshmem::barrier_all();
    println!("  p/g: passed");
    Ok(())
}

fn run_put_get_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
) -> Result<(), Box<dyn Error>> {
    const LEN: usize = 4;
    let dst = SymmetricBuffer::<i32>::new(runtime, LEN)?;
    let src = SymmetricBuffer::<i32>::new(runtime, LEN)?;
    let tmp = SymmetricBuffer::<i32>::new(runtime, LEN)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &src, &[3, 5, 7, 11])?;
    copy_to_symmetric(ctx, &dst, &[0; LEN])?;
    copy_to_symmetric(ctx, &tmp, &[0; LEN])?;
    copy_to_symmetric(ctx, &out, &[0])?;

    let mut dst_arg = dst.as_mut_ptr();
    let mut src_arg = src.as_mut_ptr();
    let mut tmp_arg = tmp.as_mut_ptr();
    let mut out_arg = out.as_mut_ptr();
    let mut len_arg = LEN as u64;
    let mut args = vec![
        &mut dst_arg as *mut *mut i32 as *mut c_void,
        &mut src_arg as *mut *mut i32 as *mut c_void,
        &mut tmp_arg as *mut *mut i32 as *mut c_void,
        &mut out_arg as *mut *mut i32 as *mut c_void,
        &mut len_arg as *mut u64 as *mut c_void,
    ];
    launch_with_args(module, cfg, "nvshmem_put_get_smoke", &mut args)?;
    ctx.synchronize()?;

    assert_eq!(
        copy_from_symmetric(ctx, &out)?[0],
        1,
        "put/get self test mismatch"
    );
    nvshmem::barrier_all();
    println!("  put/get: passed");
    Ok(())
}

fn run_signal_wait_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
) -> Result<(), Box<dyn Error>> {
    let signal = SymmetricBuffer::<u64>::new(runtime, 1)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &signal, &[0])?;
    copy_to_symmetric(ctx, &out, &[0])?;
    launch_two_args(
        module,
        cfg,
        "nvshmem_signal_wait_smoke",
        signal.as_mut_ptr(),
        out.as_mut_ptr(),
    )?;
    ctx.synchronize()?;
    assert_eq!(
        copy_from_symmetric(ctx, &out)?[0],
        1,
        "signal wait self test mismatch"
    );
    nvshmem::barrier_all();
    println!("  signal/wait: passed");
    Ok(())
}

fn run_atomic_fetch_add_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
) -> Result<(), Box<dyn Error>> {
    let buf = SymmetricBuffer::<i32>::new(runtime, 1)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 2)?;
    copy_to_symmetric(ctx, &buf, &[7])?;
    copy_to_symmetric(ctx, &out, &[0, 0])?;
    launch_two_args(
        module,
        cfg,
        "nvshmem_atomic_fetch_add_smoke",
        buf.as_mut_ptr(),
        out.as_mut_ptr(),
    )?;
    ctx.synchronize()?;
    assert_eq!(
        copy_from_symmetric(ctx, &out)?,
        vec![7, 12],
        "atomic fetch-add self test mismatch"
    );
    nvshmem::barrier_all();
    println!("  atomic fetch-add: passed");
    Ok(())
}

fn run_ring_put_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    pe: i32,
    npes: i32,
) -> Result<(), Box<dyn Error>> {
    let dst = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &dst, &[-1])?;
    nvshmem::barrier_all();
    launch_one_arg(module, cfg, "nvshmem_ring_put_smoke", dst.as_mut_ptr())?;
    ctx.synchronize()?;
    nvshmem::barrier_all();
    let expected = (pe + npes - 1) % npes;
    assert_eq!(
        copy_from_symmetric(ctx, &dst)?[0],
        expected,
        "ring put test mismatch"
    );
    nvshmem::barrier_all();
    println!("  ring put: passed");
    Ok(())
}

fn run_ring_get_test(
    runtime: &Runtime,
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    pe: i32,
    npes: i32,
) -> Result<(), Box<dyn Error>> {
    let src = SymmetricBuffer::<i32>::new(runtime, 1)?;
    let out = SymmetricBuffer::<i32>::new(runtime, 1)?;
    copy_to_symmetric(ctx, &src, &[100 + pe])?;
    copy_to_symmetric(ctx, &out, &[0])?;
    nvshmem::barrier_all();
    launch_two_args(
        module,
        cfg,
        "nvshmem_ring_get_smoke",
        src.as_mut_ptr(),
        out.as_mut_ptr(),
    )?;
    ctx.synchronize()?;
    let expected = 100 + ((pe + 1) % npes);
    assert_eq!(
        copy_from_symmetric(ctx, &out)?[0],
        expected,
        "ring get test mismatch"
    );
    nvshmem::barrier_all();
    println!("  ring get: passed");
    Ok(())
}

fn launch_one_arg<T>(
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    name: &str,
    arg: *mut T,
) -> Result<(), Box<dyn Error>> {
    let mut arg0 = arg;
    let mut args = vec![&mut arg0 as *mut *mut T as *mut c_void];
    launch_with_args(module, cfg, name, &mut args)
}

fn launch_two_args<A, B>(
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    name: &str,
    arg0: *mut A,
    arg1: *mut B,
) -> Result<(), Box<dyn Error>> {
    let mut arg0 = arg0;
    let mut arg1 = arg1;
    let mut args = vec![
        &mut arg0 as *mut *mut A as *mut c_void,
        &mut arg1 as *mut *mut B as *mut c_void,
    ];
    launch_with_args(module, cfg, name, &mut args)
}

fn launch_with_args(
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    name: &str,
    args: &mut [*mut c_void],
) -> Result<(), Box<dyn Error>> {
    let function = module.load_function(name)?;
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            ptr::null_mut(),
            args,
        )?;
    }
    Ok(())
}
