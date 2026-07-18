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
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

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

pub enum InitMethod {
    BootstrapEnv,
    MpiComm(MpiComm),
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

pub struct MpiComm {
    raw: *mut c_void,
}

impl MpiComm {
    /// Creates an MPI initialization method from a pointer to an `MPI_Comm` object.
    ///
    /// # Safety
    ///
    /// `raw` must point to a valid initialized `MPI_Comm` while the NVSHMEM
    /// runtime is live and must use the same MPI implementation as NVSHMEM.
    pub unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }
}

struct RuntimeInner {
    _uid: Option<Box<sys::nvshmemx_uniqueid_t>>,
}

enum RuntimeState {
    Uninitialized,
    Active(Weak<RuntimeInner>),
    Finalizing,
}

struct RuntimeRegistry {
    state: Mutex<RuntimeState>,
    ready: Condvar,
}

fn runtime_registry() -> &'static RuntimeRegistry {
    static REGISTRY: OnceLock<RuntimeRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| RuntimeRegistry {
        state: Mutex::new(RuntimeState::Uninitialized),
        ready: Condvar::new(),
    })
}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        let registry = runtime_registry();
        let mut state = registry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *state = RuntimeState::Finalizing;
        unsafe {
            sys::nvshmemx_hostlib_finalize();
        }
        *state = RuntimeState::Uninitialized;
        registry.ready.notify_all();
    }
}

#[derive(Clone)]
pub struct NvshmemRuntime {
    inner: Arc<RuntimeInner>,
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
        let registry = runtime_registry();
        let mut state = registry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());

        loop {
            match &*state {
                RuntimeState::Uninitialized => break,
                RuntimeState::Active(inner) => {
                    if let Some(inner) = inner.upgrade() {
                        return Ok(Self { inner });
                    }
                }
                RuntimeState::Finalizing => {}
            }
            state = registry
                .ready
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }

        load_nvshmem_host_global()?;

        let inner = match init_method {
            InitMethod::BootstrapEnv => Self::init_from_bootstrap_env(),
            InitMethod::MpiComm(mpi_comm) => Self::init_with_mpi_comm(mpi_comm, cuda_device_id),
            InitMethod::UniqueId { uid, rank, nranks } => {
                Self::init_with_unique_id(uid, rank, nranks, cuda_device_id)
            }
        }?;
        *state = RuntimeState::Active(Arc::downgrade(&inner));
        Ok(Self { inner })
    }

    fn init_from_bootstrap_env() -> Result<Arc<RuntimeInner>> {
        Self::hostlib_init(0, ptr::null_mut(), None)
    }

    fn init_with_mpi_comm(
        mpi_comm: MpiComm,
        cuda_device_id: i32,
    ) -> Result<Arc<RuntimeInner>> {
        let mut attr = sys::nvshmemx_init_attr_t::default();
        attr.args.cuda_device_id = cuda_device_id;
        let status = unsafe { sys::nvshmemx_set_attr_mpi_comm_args(mpi_comm.raw, &mut attr) };
        if status != 0 {
            return Err(
                format!("nvshmemx_set_attr_mpi_comm_args failed with status {status}").into(),
            );
        }
        Self::hostlib_init(sys::NVSHMEMX_INIT_WITH_MPI_COMM, &mut attr, None)
    }

    fn init_with_unique_id(
        uid: sys::nvshmemx_uniqueid_t,
        rank: i32,
        nranks: i32,
        cuda_device_id: i32,
    ) -> Result<Arc<RuntimeInner>> {
        let uid = Box::new(uid);
        let mut attr = sys::nvshmemx_init_attr_t::default();
        attr.args.cuda_device_id = cuda_device_id;
        let status = unsafe {
            sys::nvshmemx_set_attr_uniqueid_args(rank, nranks, uid.as_ref(), &mut attr)
        };
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
        uid: Option<Box<sys::nvshmemx_uniqueid_t>>,
    ) -> Result<Arc<RuntimeInner>> {
        let status = unsafe { sys::nvshmemx_hostlib_init_attr(flags, attr) };
        if status != 0 {
            return Err(format!("nvshmemx_hostlib_init_attr failed with status {status}").into());
        }

        Ok(Arc::new(RuntimeInner { _uid: uid }))
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
    _runtime: Arc<RuntimeInner>,
    _marker: PhantomData<T>,
}

impl<T> SymmetricBuffer<T> {
    pub fn new(runtime: &NvshmemRuntime, len: usize) -> Result<Self> {
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
            _runtime: Arc::clone(&runtime.inner),
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
