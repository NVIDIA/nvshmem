/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::missing_safety_doc)]

use cuda_core::{
    CudaContext, CudaModule, CudaStream, DriverError, IntoResult, LaunchConfig, memory, sys,
};
use cuda_device::{kernel, thread};
use libnvvm_sys::{LibNvvm, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use nvshmem::{InitMethod, Runtime, SymmetricBuffer};
use std::env;
use std::error::Error;
use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GB: f64 = 1_000_000_000.0;
const DEVICE_OP_PUT: i32 = 0;
const DEVICE_OP_GET: i32 = 1;
const DEVICE_OP_P: i32 = 2;
const DEVICE_OP_G: i32 = 3;

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
pub fn nvshmem_device_rma_perf(buf: *mut i32, nelems: u64, peer: i32, iters: u64, op: i32) {
    if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
        return;
    }

    unsafe {
        let mut iter = 0;
        while iter < iters {
            if op == DEVICE_OP_PUT {
                nvshmem_device::nvshmem_int_put_nbi(buf, buf.cast_const(), nelems, peer);
                nvshmem_device::nvshmem_quiet();
            } else if op == DEVICE_OP_GET {
                nvshmem_device::nvshmem_int_get_nbi(buf, buf.cast_const(), nelems, peer);
                nvshmem_device::nvshmem_quiet();
            } else if op == DEVICE_OP_P {
                let mut idx = 0usize;
                while idx < nelems as usize {
                    nvshmem_device::nvshmem_int_p(buf.add(idx), *buf.add(idx), peer);
                    idx += 1;
                }
                nvshmem_device::nvshmem_quiet();
            } else {
                let mut idx = 0usize;
                while idx < nelems as usize {
                    *buf.add(idx) = nvshmem_device::nvshmem_int_g(buf.add(idx).cast_const(), peer);
                    idx += 1;
                }
            }
            iter += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct PerfOptions {
    min_size: usize,
    max_size: usize,
    step: usize,
    iters: u64,
    warmup: u64,
}

impl PerfOptions {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let min_size = read_env_usize("NVSHMEM_RUST_PERF_MIN_SIZE", 4)?;
        let max_size = read_env_usize("NVSHMEM_RUST_PERF_MAX_SIZE", 1 << 20)?;
        let step = read_env_usize("NVSHMEM_RUST_PERF_STEP", 2)?;
        let iters = read_env_u64("NVSHMEM_RUST_PERF_ITERS", 100)?;
        let warmup = read_env_u64("NVSHMEM_RUST_PERF_WARMUP", 10)?;
        if min_size == 0 || max_size < min_size || step < 2 || iters == 0 {
            return Err(
                "invalid perf options: require min_size > 0, max_size >= min_size, step >= 2, iters > 0"
                    .into(),
            );
        }
        Ok(Self {
            min_size,
            max_size,
            step,
            iters,
            warmup,
        })
    }

    fn sizes(self) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut size = self.min_size;
        while size <= self.max_size {
            sizes.push(size);
            match size.checked_mul(self.step) {
                Some(next) if next > size => size = next,
                _ => break,
            }
        }
        sizes
    }
}

fn read_env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    env::var(name)
        .map(|value| value.parse::<usize>().map_err(Into::into))
        .unwrap_or(Ok(default))
}

fn read_env_u64(name: &str, default: u64) -> Result<u64, Box<dyn Error>> {
    env::var(name)
        .map(|value| value.parse::<u64>().map_err(Into::into))
        .unwrap_or(Ok(default))
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

fn main() -> Result<(), Box<dyn Error>> {
    let arch = env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_90".to_string());
    let cubin = build_cubin(&arch)?;
    if env::var_os("NVSHMEM_RUST_COMPILE_ONLY").is_some() {
        println!("NVSHMEM CUDA-Oxide Rust perf compile-only passed for {arch}");
        return Ok(());
    }

    let opts = PerfOptions::from_env()?;
    let target_cc = target_compute_capability(&arch);
    let device_ordinal = select_device_ordinal(target_cc)?;
    let cc = device_compute_capability(device_ordinal as i32)?;
    let local_rank = local_rank_from_env().unwrap_or(0);
    eprintln!(
        "NVSHMEM CUDA-Oxide Rust perf: selected CUDA device {device_ordinal} for local rank {local_rank} with compute capability {}.{}",
        cc.0, cc.1
    );

    let ctx = CudaContext::new(device_ordinal)?;
    let stream = ctx.new_stream()?;
    let module = ctx.load_module_from_image(&cubin)?;
    let runtime = Runtime::init(init_method_from_env()?, device_ordinal as i32)?;
    let _module_registration = unsafe { runtime.register_module(&module) }?;

    let pe = nvshmem::my_pe();
    let npes = nvshmem::n_pes();
    println!("NVSHMEM CUDA-Oxide Rust perf: PE {pe}/{npes} on device {device_ordinal}");
    if pe == 0 {
        println!(
            "# options min_size={} max_size={} step={} iters={} warmup={}",
            opts.min_size, opts.max_size, opts.step, opts.iters, opts.warmup
        );
    }

    run_host_put_perf(&runtime, &stream, opts, pe, npes)?;
    run_host_reduction_perf(&runtime, &stream, opts, pe)?;
    run_device_rma_perf(&runtime, &ctx, &stream, &module, opts, pe, npes)?;

    nvshmem::barrier_all();
    stream.synchronize()?;
    Ok(())
}

fn run_host_put_perf(
    runtime: &Runtime,
    stream: &CudaStream,
    opts: PerfOptions,
    pe: i32,
    npes: i32,
) -> Result<(), Box<dyn Error>> {
    if npes != 2 {
        if pe == 0 {
            eprintln!("Skipping host put perf: requires exactly two PEs");
        }
        return Ok(());
    }

    let src = SymmetricBuffer::<u8>::new(runtime, opts.max_size)?;
    let dst = SymmetricBuffer::<u8>::new(runtime, opts.max_size)?;
    memset_async(stream, &src, 1, opts.max_size)?;
    memset_async(stream, &dst, 0, opts.max_size)?;
    stream.synchronize()?;

    if pe == 0 {
        println!("# rust_host_putmem_nbi_on_stream");
        print_bw_header();
    }

    for size in opts.sizes() {
        nvshmem::barrier_all();
        if pe == 0 {
            let peer = 1;
            issue_host_put(&dst, &src, size, peer, opts.warmup, stream)?;
            stream.synchronize()?;
            let ms = time_stream(stream, || {
                issue_host_put(&dst, &src, size, peer, opts.iters, stream)
            })?;
            let latency_us = (ms as f64 * 1000.0) / opts.iters as f64;
            let bandwidth = bandwidth_gbs(size, opts.iters, ms);
            print_bw_row(size, size, latency_us, bandwidth);
        }
        nvshmem::barrier_all();
    }
    Ok(())
}

fn run_host_reduction_perf(
    runtime: &Runtime,
    stream: &CudaStream,
    opts: PerfOptions,
    pe: i32,
) -> Result<(), Box<dyn Error>> {
    let max_elems = (opts.max_size / size_of::<i32>()).max(1);
    let src = SymmetricBuffer::<i32>::new(runtime, max_elems)?;
    let dst = SymmetricBuffer::<i32>::new(runtime, max_elems)?;
    memset_async(stream, &src, 1, max_elems * size_of::<i32>())?;
    memset_async(stream, &dst, 0, max_elems * size_of::<i32>())?;
    stream.synchronize()?;

    if pe == 0 {
        println!("# rust_host_int_sum_reduce_on_stream");
        print_bw_header();
    }

    for size in opts.sizes() {
        let nelems = (size / size_of::<i32>()).max(1);
        nvshmem::barrier_all();
        issue_host_reduce(&dst, &src, nelems, opts.warmup, stream)?;
        stream.synchronize()?;
        let ms = time_stream(stream, || {
            issue_host_reduce(&dst, &src, nelems, opts.iters, stream)
        })?;
        let latency_us = (ms as f64 * 1000.0) / opts.iters as f64;
        let bytes = nelems * size_of::<i32>();
        let bandwidth = bandwidth_gbs(bytes, opts.iters, ms);
        if pe == 0 {
            print_bw_row(bytes, nelems, latency_us, bandwidth);
        }
        nvshmem::barrier_all();
    }
    Ok(())
}

fn run_device_rma_perf(
    runtime: &Runtime,
    ctx: &CudaContext,
    stream: &CudaStream,
    module: &Arc<CudaModule>,
    opts: PerfOptions,
    pe: i32,
    npes: i32,
) -> Result<(), Box<dyn Error>> {
    if npes != 2 {
        if pe == 0 {
            eprintln!("Skipping device RMA perf: requires exactly two PEs");
        }
        return Ok(());
    }

    let max_elems = (opts.max_size / size_of::<i32>()).max(1);
    let buf = SymmetricBuffer::<i32>::new(runtime, max_elems)?;
    memset_async(stream, &buf, 7, max_elems * size_of::<i32>())?;
    stream.synchronize()?;

    let ops = [
        ("rust_device_put", DEVICE_OP_PUT),
        ("rust_device_get", DEVICE_OP_GET),
        ("rust_device_p", DEVICE_OP_P),
        ("rust_device_g", DEVICE_OP_G),
    ];
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let function = module.load_function("nvshmem_device_rma_perf")?;

    for (name, op) in ops {
        if pe == 0 {
            println!("# {name}_bw_lat");
            print_bw_header();
        }
        for size in opts.sizes() {
            let nelems = (size / size_of::<i32>()).max(1);
            nvshmem::barrier_all();
            if pe == 0 {
                let mut buf_arg = buf.as_mut_ptr();
                let mut nelems_arg = nelems as u64;
                let mut peer_arg = 1;
                let mut iters_arg = opts.warmup;
                let mut op_arg = op;
                let mut args = [
                    &mut buf_arg as *mut *mut i32 as *mut c_void,
                    &mut nelems_arg as *mut u64 as *mut c_void,
                    &mut peer_arg as *mut i32 as *mut c_void,
                    &mut iters_arg as *mut u64 as *mut c_void,
                    &mut op_arg as *mut i32 as *mut c_void,
                ];
                unsafe {
                    cuda_core::launch_kernel(
                        function.cu_function(),
                        cfg.grid_dim,
                        cfg.block_dim,
                        cfg.shared_mem_bytes,
                        stream.cu_stream(),
                        &mut args,
                    )?;
                }
                stream.synchronize()?;
                iters_arg = opts.iters;
                let ms = time_stream(stream, || {
                    unsafe {
                        cuda_core::launch_kernel(
                            function.cu_function(),
                            cfg.grid_dim,
                            cfg.block_dim,
                            cfg.shared_mem_bytes,
                            stream.cu_stream(),
                            &mut args,
                        )?;
                    }
                    Ok(())
                })?;
                let bytes = nelems * size_of::<i32>();
                let latency_us = (ms as f64 * 1000.0) / opts.iters as f64;
                let bandwidth = bandwidth_gbs(bytes, opts.iters, ms);
                print_bw_row(bytes, nelems, latency_us, bandwidth);
            }
            nvshmem::barrier_all();
            ctx.synchronize()?;
        }
    }
    Ok(())
}

fn issue_host_put(
    dst: &SymmetricBuffer<u8>,
    src: &SymmetricBuffer<u8>,
    size: usize,
    peer: i32,
    iters: u64,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let cstrm = stream.cu_stream().cast::<c_void>();
    for _ in 0..iters {
        unsafe {
            nvshmem::putmem_nbi_on_stream(
                dst.as_mut_ptr().cast::<c_void>(),
                src.as_mut_ptr().cast::<c_void>(),
                size as u64,
                peer,
                cstrm,
            );
        }
    }
    unsafe {
        nvshmem::quiet_on_stream(cstrm);
    }
    Ok(())
}

fn issue_host_reduce(
    dst: &SymmetricBuffer<i32>,
    src: &SymmetricBuffer<i32>,
    nelems: usize,
    iters: u64,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let cstrm = stream.cu_stream().cast::<c_void>();
    for _ in 0..iters {
        let status = unsafe {
            nvshmem::int_sum_reduce_on_stream(
                nvshmem::sys::NVSHMEM_TEAM_WORLD,
                dst.as_mut_ptr(),
                src.as_mut_ptr().cast_const(),
                nelems as u64,
                cstrm,
            )
        };
        check_nvshmem_status(status, "int_sum_reduce_on_stream")?;
    }
    stream.synchronize()?;
    Ok(())
}

fn time_stream<F>(stream: &CudaStream, f: F) -> Result<f32, Box<dyn Error>>
where
    F: FnOnce() -> Result<(), Box<dyn Error>>,
{
    let start = stream.record_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    f()?;
    let stop = stream.record_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    stop.synchronize()?;
    Ok(start.elapsed_ms(&stop)?)
}

fn memset_async<T>(
    stream: &CudaStream,
    buffer: &SymmetricBuffer<T>,
    value: u8,
    bytes: usize,
) -> Result<(), DriverError> {
    unsafe {
        memory::memset_d8_async(
            buffer.as_mut_ptr() as sys::CUdeviceptr,
            value,
            bytes,
            stream.cu_stream(),
        )
    }
}

fn bandwidth_gbs(size: usize, iters: u64, ms: f32) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    (size as f64 * iters as f64) / ((ms as f64 / 1000.0) * GB)
}

fn print_bw_header() {
    println!(
        "{:<12}{:<12}{:<18}{:<15}",
        "size(B)", "count", "latency(us)", "algbw(GB/s)"
    );
}

fn print_bw_row(size: usize, count: usize, latency_us: f64, bandwidth: f64) {
    println!(
        "{:<12}{:<12}{:<18.3}{:<15.3}",
        size, count, latency_us, bandwidth
    );
}

fn check_nvshmem_status(status: i32, name: &str) -> Result<(), Box<dyn Error>> {
    if status != 0 {
        return Err(format!("{name} failed with status {status}").into());
    }
    Ok(())
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
        "nvshmem_cuda_oxide_perf.ltoir",
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
