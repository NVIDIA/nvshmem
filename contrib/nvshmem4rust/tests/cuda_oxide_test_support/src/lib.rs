/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{DriverError, IntoResult, sys};
use libnvvm_sys::{LibNvvm, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use nvshmem::InitMethod;
use std::env;
use std::error::Error;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};

pub fn init_method_from_env() -> Result<InitMethod, Box<dyn Error>> {
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

pub fn target_compute_capability(arch: &str) -> Option<(i32, i32)> {
    let digits = arch.strip_prefix("sm_")?;
    if digits.len() < 2 {
        return None;
    }
    let split = digits.len() - 1;
    let major = digits[..split].parse().ok()?;
    let minor = digits[split..].parse().ok()?;
    Some((major, minor))
}

pub fn select_device_ordinal(target_cc: Option<(i32, i32)>) -> Result<usize, Box<dyn Error>> {
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

pub fn local_rank_from_env() -> Option<usize> {
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

pub fn device_compute_capability(ordinal: i32) -> Result<(i32, i32), DriverError> {
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

pub fn build_cubin(
    manifest_dir: &str,
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(manifest_dir);
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
    let rust_ltoir_name = format!("{module_name}.ltoir");
    linker.add(InputType::Ltoir, &rust_ltoir, &rust_ltoir_name)?;
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

fn forced_device_ordinal() -> Result<Option<usize>, Box<dyn Error>> {
    match env::var("NVSHMEM_RUST_CUDA_DEVICE") {
        Ok(value) => Ok(Some(value.parse::<usize>().map_err(|err| {
            format!("invalid NVSHMEM_RUST_CUDA_DEVICE={value}: {err}")
        })?)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("invalid NVSHMEM_RUST_CUDA_DEVICE: {err}").into()),
    }
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
