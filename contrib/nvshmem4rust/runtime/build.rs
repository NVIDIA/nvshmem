/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=NVSHMEM_HOST_LIB_DIR");
    println!("cargo:rustc-link-lib=dylib=nvshmem_host");

    if let Ok(lib_dir) = env::var("NVSHMEM_HOST_LIB_DIR") {
        if !lib_dir.is_empty() {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
    }
}
