/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::missing_safety_doc)]

use cuda_core::CudaModule;
use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, c_void};
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
pub mod sys {
    include!("bindings.rs");
}

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
pub mod bindings {
    pub use super::sys::*;
}

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
pub mod api {
    include!("api.rs");
}

pub use api::*;

/// Register a CUDA-Oxide module with NVSHMEM.
///
/// This borrows CUDA-Oxide's raw `CUmodule` handle only for the duration of the
/// NVSHMEM registration call. CUDA-Oxide keeps ownership of the module.
///
/// # Safety
///
/// The caller must register before launching kernels that call NVSHMEM device
/// functions, finalize before dropping the module when NVSHMEM requires it, and
/// satisfy CUDA's current-context requirements for this module.
pub unsafe fn cumodule_init(module: &CudaModule) -> i32 {
    let raw_module = unsafe { module.cu_module() }.cast::<c_void>();
    unsafe { sys::nvshmemx_cumodule_init(raw_module) }
}

/// Finalize NVSHMEM registration for a CUDA-Oxide module.
///
/// # Safety
///
/// The caller must ensure no kernels that depend on this module's NVSHMEM
/// registration are still running and satisfy CUDA's current-context
/// requirements for this module.
pub unsafe fn cumodule_finalize(module: &CudaModule) -> i32 {
    let raw_module = unsafe { module.cu_module() }.cast::<c_void>();
    unsafe { sys::nvshmemx_cumodule_finalize(raw_module) }
}

#[derive(Clone, Copy)]
pub enum InitMethod {
    BootstrapEnv,
    MpiCommWorld,
    UniqueId {
        uid: sys::nvshmemx_uniqueid_t,
        rank: i32,
        nranks: i32,
    },
}

impl InitMethod {
    pub fn unique_id(uid: sys::nvshmemx_uniqueid_t, rank: i32, nranks: i32) -> Self {
        Self::UniqueId { uid, rank, nranks }
    }

    pub fn single_pe_uid() -> Result<Self> {
        load_nvshmem_host_global()?;
        let mut uid = sys::nvshmemx_uniqueid_t::default();
        let status = unsafe { sys::nvshmemx_get_uniqueid(&mut uid) };
        if status != 0 {
            return Err(format!("nvshmemx_get_uniqueid failed with status {status}").into());
        }
        Ok(Self::unique_id(uid, 0, 1))
    }
}

pub struct NvshmemRuntime {
    _uid: Option<sys::nvshmemx_uniqueid_t>,
}

pub type Runtime = NvshmemRuntime;

impl NvshmemRuntime {
    /// Initialize NVSHMEM host state.
    ///
    /// `cuda_device_id` is passed through to NVSHMEM's public init attribute
    /// for methods that use it. This crate does not select CUDA devices or
    /// create CUDA contexts; CUDA host utilities should stay in CUDA-facing
    /// crates rather than this NVSHMEM API wrapper.
    pub fn init(init_method: InitMethod, cuda_device_id: i32) -> Result<Self> {
        load_nvshmem_host_global()?;

        match init_method {
            InitMethod::BootstrapEnv => Self::init_from_bootstrap_env(),
            InitMethod::MpiCommWorld => Self::init_with_mpi_comm_world(cuda_device_id),
            InitMethod::UniqueId { uid, rank, nranks } => {
                Self::init_with_unique_id(uid, rank, nranks, cuda_device_id)
            }
        }
    }

    fn init_from_bootstrap_env() -> Result<Self> {
        Self::hostlib_init(0, ptr::null_mut(), None)
    }

    fn init_with_mpi_comm_world(cuda_device_id: i32) -> Result<Self> {
        let mut attr = sys::nvshmemx_init_attr_t::default();
        attr.args.cuda_device_id = cuda_device_id;
        Self::hostlib_init(sys::NVSHMEMX_INIT_WITH_MPI_COMM, &mut attr, None)
    }

    fn init_with_unique_id(
        uid: sys::nvshmemx_uniqueid_t,
        rank: i32,
        nranks: i32,
        cuda_device_id: i32,
    ) -> Result<Self> {
        let mut attr = sys::nvshmemx_init_attr_t::default();
        attr.args.cuda_device_id = cuda_device_id;
        let status = unsafe { sys::nvshmemx_set_attr_uniqueid_args(rank, nranks, &uid, &mut attr) };
        if status != 0 {
            return Err(
                format!("nvshmemx_set_attr_uniqueid_args failed with status {status}").into(),
            );
        }

        Self::hostlib_init(sys::NVSHMEMX_INIT_WITH_UNIQUEID, &mut attr, Some(uid))
    }

    fn hostlib_init(
        flags: u32,
        attr: *mut sys::nvshmemx_init_attr_t,
        uid: Option<sys::nvshmemx_uniqueid_t>,
    ) -> Result<Self> {
        let status = unsafe { sys::nvshmemx_hostlib_init_attr(flags, attr) };
        if status != 0 {
            return Err(format!("nvshmemx_hostlib_init_attr failed with status {status}").into());
        }

        Ok(Self { _uid: uid })
    }
}

impl Drop for NvshmemRuntime {
    fn drop(&mut self) {
        unsafe {
            sys::nvshmemx_hostlib_finalize();
        }
    }
}

pub fn load_nvshmem_host_global() -> Result<()> {
    let mut candidates = Vec::new();
    if let Ok(path) = env::var("NVSHMEM_HOST_LIB_PATH") {
        candidates.push(path);
    }
    candidates.push("libnvshmem_host.so.3".to_string());
    candidates.push("libnvshmem_host.so".to_string());

    let mut last_error = None;
    for candidate in candidates {
        match unsafe { Library::open(Some(OsStr::new(&candidate)), RTLD_NOW | RTLD_GLOBAL) } {
            Ok(library) => {
                // Keep the RTLD_GLOBAL handle live for NVSHMEM's dependent symbol lookups.
                std::mem::forget(library);
                return Ok(());
            }
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }
    }

    Err(format!(
        "failed to load libnvshmem_host with RTLD_GLOBAL: {}",
        last_error.unwrap_or_else(|| "unknown libloading error".to_string())
    )
    .into())
}

pub struct SymmetricBuffer<T> {
    pub ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> SymmetricBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or("symmetric allocation size overflow")?;
        let ptr = unsafe { sys::nvshmem_malloc(bytes as u64) }.cast::<T>();
        if ptr.is_null() {
            return Err(format!("nvshmem_malloc({bytes}) returned null").into());
        }
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for SymmetricBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            sys::nvshmem_free(self.ptr.cast::<c_void>());
        }
    }
}
