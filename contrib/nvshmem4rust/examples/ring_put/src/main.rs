/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal nvshmem4rust example: a CUDA-Oxide kernel puts its PE's data into
//! the next PE's symmetric buffer, then every PE checks what it received.

mod helpers;

use cuda_core::{CudaContext, LaunchConfig};
use cuda_device::{kernel, thread};
use helpers::{copy_from, copy_to, launch};
use nvshmem::{Runtime, SymmetricBuffer};
use nvshmem_cuda_oxide_test_support::{
    build_cubin, init_method_from_env, select_device_ordinal, target_compute_capability,
};
use std::{env, error::Error};

// Rust declarations for NVSHMEM's device API, generated from nvshmem.h.
#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
mod nvshmem_device {
    include!(env!("NVSHMEM_RUST_BINDINGS"));
}
use nvshmem_device::{nvshmem_int_put, nvshmem_my_pe, nvshmem_n_pes, nvshmem_quiet};

// This function runs on the GPU. One thread writes `n` ints into the
// next PE's `dst` buffer.
#[kernel]
pub fn ring_put(dst: *mut i32, src: *const i32, n: u64) {
    if thread::threadIdx_x() == 0 {
        unsafe {
            let peer = (nvshmem_my_pe() + 1) % nvshmem_n_pes();
            nvshmem_int_put(dst, src, n, peer); // copy into the neighbour's GPU
            nvshmem_quiet(); // wait until the data arrives
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Link the kernel above with NVSHMEM's device library.
    let arch = env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_90".to_string());
    let cubin = build_cubin(env!("CARGO_MANIFEST_DIR"), env!("CARGO_PKG_NAME"), &arch)?;
    if env::var_os("NVSHMEM_RUST_COMPILE_ONLY").is_some() {
        println!("linked {}.cubin for {arch}", env!("CARGO_PKG_NAME"));
        return Ok(());
    }
    let device = select_device_ordinal(target_compute_capability(&arch))?;
    let ctx = CudaContext::new(device)?;
    let module = ctx.load_module_from_image(&cubin)?;

    // 2. Start NVSHMEM (connects the processes) and register our kernels.
    let runtime = Runtime::init(init_method_from_env()?, device as i32)?;
    let _reg = unsafe { runtime.register_module(&module) }?;
    let (me, npes) = (nvshmem::my_pe(), nvshmem::n_pes());

    // 3. Symmetric buffers: same size on every PE.
    let n = 1024;
    let src = SymmetricBuffer::<i32>::new(&runtime, n)?;
    let dst = SymmetricBuffer::<i32>::new(&runtime, n)?;
    copy_to(&ctx, &src, &vec![100 + me; n])?;
    copy_to(&ctx, &dst, &vec![-1; n])?;
    nvshmem::barrier_all();

    // 4. Launch the kernel: each PE pushes its data to (me + 1) % npes.
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: the arguments match `ring_put`'s (dst, src, n) signature, and both
    // symmetric buffers hold `n` ints and outlive the synchronize below.
    unsafe {
        launch(&module, cfg, "ring_put",
               &[&dst.as_mut_ptr(), &src.as_mut_ptr(), &(n as u64)])?;
    }
    ctx.synchronize()?;
    nvshmem::barrier_all(); // everyone's puts are done

    // 5. Check what our left neighbour wrote into our buffer.
    let got = copy_from(&ctx, &dst)?;
    let from = (me + npes - 1) % npes;
    assert!(got.iter().all(|&x| x == 100 + from));
    println!("PE {me}/{npes}: received {n} x {} from PE {from} OK", got[0]);
    Ok(())
}
