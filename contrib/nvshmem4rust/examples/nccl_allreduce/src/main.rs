/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal NCCL-from-Rust example (cudarc): every rank contributes a buffer of
//! rank + 1, and an allreduce leaves the sum on every GPU.

use cudarc::driver::CudaContext;
use cudarc::nccl::{Comm, Id, ReduceOp};
use std::{env, error::Error, fs, thread, time::Duration};

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Who am I? The launcher (srun) tells every process its rank.
    let rank: usize = env::var("SLURM_PROCID")?.parse()?;
    let world: usize = env::var("SLURM_NTASKS")?.parse()?;
    let host = env::var("SLURMD_NODENAME").unwrap_or_default();

    // 2. One GPU per process, and a CUDA stream to queue work on.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    // 3. Rank 0 makes a unique ID, everyone joins the same communicator.
    let id = share_unique_id(rank)?;
    let comm = Comm::from_rank(stream.clone(), rank, world, id)
        .expect("ncclCommInitRank");

    // 4. Each rank contributes a buffer full of (rank + 1).
    let n = 1 << 20;
    let send = stream.clone_htod(&vec![(rank + 1) as f32; n])?;
    let mut recv = stream.alloc_zeros::<f32>(n)?;

    // 5. Enqueue the allreduce on the stream. This returns immediately;
    //    the GPU does the work, the CPU is free.
    comm.all_reduce(&send, &mut recv, &ReduceOp::Sum)
        .expect("ncclAllReduce");

    // 6. Copy back (stream-ordered after the allreduce) and check.
    let out = stream.clone_dtoh(&recv)?;
    let expected = (world * (world + 1) / 2) as f32;
    assert!(out.iter().all(|&x| x == expected));
    println!("[{host}] rank {rank}/{world}: sum = {} (expected {expected}) OK", out[0]);
    Ok(())
}

/// The NCCL unique ID is 128 opaque bytes. Pass it through a file in
/// NCCL_ID_DIR (default: the current directory), which all ranks must share.
fn share_unique_id(rank: usize) -> Result<Id, Box<dyn Error>> {
    let dir = env::var("NCCL_ID_DIR").unwrap_or_else(|_| ".".to_string());
    let path = format!("{dir}/nccl-id.{}", env::var("SLURM_JOB_ID")?);
    if rank == 0 {
        let id = Id::new().expect("ncclGetUniqueId");
        let bytes: Vec<u8> = id.internal().iter().map(|&c| c as u8).collect();
        fs::write(format!("{path}.tmp"), bytes)?;
        fs::rename(format!("{path}.tmp"), &path)?;
        return Ok(id);
    }
    let bytes = loop {
        match fs::read(&path) {
            Ok(b) if b.len() == 128 => break b,
            _ => thread::sleep(Duration::from_millis(50)),
        }
    };
    Ok(Id::uninit(std::array::from_fn(|i| bytes[i] as _)))
}
