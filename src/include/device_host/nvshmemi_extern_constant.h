/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Common EXTERN_CONSTANT macro definition for device __constant__ symbols.
 *
 * Selects the appropriate linkage qualifier based on the active compiler:
 *   - RDC / NVRTC:    extern __constant__  (resolved by device linker, nvJitLink/cuLinkAddData for
 *                     NVRTC)
 *   - Clang CUDA:     extern __constant__ (e.g. when compiling as LTOIR)
 *   - Clang bitcode:  extern address_space(4) (LLVM address_space(4) maps to __constant__)
 *   - Non-RDC nvcc:   __constant__  (per-TU copy, implicit internal linkage as each module gets
 *                     its own symbol definition)
 *
 * Each consumer includes this header, declares its variable, then #undefs the
 * macro to prevent leaking.  Callers that need special handling (e.g. the
 * NVRTC+Numba path in nvshmemi_common_device_defines.cuh) may #define
 * EXTERN_CONSTANT before including this header to skip the default logic.
 */
#ifndef EXTERN_CONSTANT

#if defined(__CUDACC_RDC__) || defined(__CUDACC_RTC__)
#define EXTERN_CONSTANT extern __constant__
#elif defined(__clang__)
#ifdef __CUDACC__
// Clang CUDA mode: use __constant__ only (avoid address_space to fix LLVM21)
#define EXTERN_CONSTANT extern __constant__
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only
#define EXTERN_CONSTANT extern __attribute__((address_space(4)))
#endif
#elif defined(__CUDACC__)
// Non-RDC nvcc: per-TU __constant__ copy (internal linkage is implicit).
// Only functional with nvshmemx_cumodule_init / nvshmemx_culibrary_init.
#define EXTERN_CONSTANT __constant__
#endif

#endif /* EXTERN_CONSTANT */
