/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Plumbing kept out of main.rs: host<->symmetric copies and a raw kernel launch.

use cuda_core::simt::memory;
use cuda_core::{CudaContext, CudaModule, DriverError, LaunchConfig, sys};
use nvshmem::SymmetricBuffer;
use std::error::Error;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

pub fn copy_to<T: Copy>(
    ctx: &CudaContext,
    buf: &SymmetricBuffer<T>,
    values: &[T],
) -> Result<(), DriverError> {
    assert_eq!(values.len(), buf.len());
    ctx.bind_to_thread()?;
    unsafe {
        memory::memcpy_htod_sync(
            buf.as_mut_ptr() as sys::CUdeviceptr,
            values.as_ptr(),
            values.len() * size_of::<T>(),
        )
    }
}

pub fn copy_from(
    ctx: &CudaContext,
    buf: &SymmetricBuffer<i32>,
) -> Result<Vec<i32>, DriverError> {
    ctx.bind_to_thread()?;
    let mut values = vec![0i32; buf.len()];
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buf.as_mut_ptr() as sys::CUdeviceptr,
            buf.len() * size_of::<i32>(),
            ptr::null_mut(),
        )?;
    }
    ctx.synchronize()?;
    Ok(values)
}

/// Launch `name` from `module`. Each arg is a reference to a kernel parameter.
///
/// # Safety
///
/// `name` must be a kernel in `module` whose parameters match `args` in number,
/// order and type, and every pointer argument must be valid for the accesses the
/// kernel makes until it completes.
pub unsafe fn launch(
    module: &Arc<CudaModule>,
    cfg: LaunchConfig,
    name: &str,
    args: &[&dyn KernelArg],
) -> Result<(), Box<dyn Error>> {
    let function = module.load_function(name)?;
    let mut raw: Vec<*mut c_void> = args.iter().map(|a| a.as_arg_ptr()).collect();
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            ptr::null_mut(),
            &mut raw,
        )?;
    }
    Ok(())
}

pub trait KernelArg {
    fn as_arg_ptr(&self) -> *mut c_void;
}

// Explicit impls only: a blanket `impl<T: Copy>` would also match
// `&dyn KernelArg` itself and hand the kernel the address of the reference.
macro_rules! kernel_arg {
    ($($t:ty),*) => {$(
        impl KernelArg for $t {
            fn as_arg_ptr(&self) -> *mut c_void {
                self as *const $t as *mut c_void
            }
        }
    )*};
}
kernel_arg!(i32, u32, i64, u64, f32, f64);

impl<T> KernelArg for *mut T {
    fn as_arg_ptr(&self) -> *mut c_void {
        self as *const *mut T as *mut c_void
    }
}

impl<T> KernelArg for *const T {
    fn as_arg_ptr(&self) -> *mut c_void {
        self as *const *const T as *mut c_void
    }
}
