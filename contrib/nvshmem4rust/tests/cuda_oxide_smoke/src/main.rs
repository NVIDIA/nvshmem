/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::missing_safety_doc)]

use cuda_core::{CudaContext, CudaModule, DriverError, LaunchConfig, memory, sys};
use cuda_device::{kernel, thread};
use nvshmem::{Runtime, SymmetricBuffer};
use nvshmem_cuda_oxide_test_support::{
    build_cubin, device_compute_capability, init_method_from_env, select_device_ordinal,
    target_compute_capability,
};
use std::env;
use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
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
    let _ = nvshmem::int_put_on_stream
        as unsafe extern "C" fn(*mut i32, *const i32, u64, i32, *mut c_void);
    let _ = nvshmem::int_put_signal_on_stream
        as unsafe extern "C" fn(*mut i32, *const i32, u64, *mut u64, u64, i32, i32, *mut c_void);
    let _ = nvshmem::int_sum_reduce_on_stream
        as unsafe extern "C" fn(i32, *mut i32, *const i32, u64, *mut c_void) -> i32;
    let _ = nvshmem::alltoallmem_on_stream
        as unsafe extern "C" fn(i32, *mut c_void, *const c_void, u64, *mut c_void) -> i32;
    let _ = nvshmem::flush_on_stream as unsafe extern "C" fn(*mut c_void);
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

fn main() -> Result<(), Box<dyn Error>> {
    assert_host_stream_bindings_are_linkable();

    let arch = env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_90".to_string());
    let cubin = build_cubin(env!("CARGO_MANIFEST_DIR"), env!("CARGO_PKG_NAME"), &arch)?;
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
    let module_registration = unsafe { runtime.register_module(&module) }?;
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
    module_registration.finalize()?;
    Ok(())
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
