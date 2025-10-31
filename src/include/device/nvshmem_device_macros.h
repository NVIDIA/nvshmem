#ifndef _NVSHMEM_DEVICE_MACROS_H_
#define _NVSHMEM_DEVICE_MACROS_H_

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep

/*
 * These macros represent various inlining requirements based on configuration rules.
 * All functions are force inlined in the bitcode library.
 * Macro Key:
 * NVSHMEMI_DEVICE_INLINE - inlined based on NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_INLINE - inlined regardless of NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE - like above, but uses NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE
 */

#if !defined __clang_llvm_bitcode_lib__
#define NVSHMEMI_DEVICE_ALWAYS_INLINE inline
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __forceinline__
#define NVSHMEM_ALWAYS_STATIC static
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_STATIC static
#define NVSHMEMI_DEVICE_INLINE inline
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#define NVSHMEMI_STATIC static
#endif
#else
/* clang llvm ir compilation mangles names of functions marked NVSHMEMI_STATIC
 * even if they are behind extern c guards. */
#define NVSHMEMI_STATIC
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_DEVICE_INLINE __attribute__((always_inline))
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#endif
#define NVSHMEMI_DEVICE_ALWAYS_INLINE __attribute__((always_inline))
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __attribute__((always_inline))
#endif

#endif#ifndef _NVSHMEM_DEVICE_MACROS_H_
#define _NVSHMEM_DEVICE_MACROS_H_

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep

/*
 * These macros represent various inlining requirements based on configuration rules.
 * All functions are force inlined in the bitcode library.
 * Macro Key:
 * NVSHMEMI_DEVICE_INLINE - inlined based on NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_INLINE - inlined regardless of NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE - like above, but uses NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE
 */

#if !defined __clang_llvm_bitcode_lib__
#define NVSHMEMI_DEVICE_ALWAYS_INLINE inline
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __forceinline__
#define NVSHMEM_ALWAYS_STATIC static
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_STATIC static
#define NVSHMEMI_DEVICE_INLINE inline
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#define NVSHMEMI_STATIC static
#endif
#else
/* clang llvm ir compilation mangles names of functions marked NVSHMEMI_STATIC
 * even if they are behind extern c guards. */
#define NVSHMEMI_STATIC
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_DEVICE_INLINE __attribute__((always_inline))
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#endif
#define NVSHMEMI_DEVICE_ALWAYS_INLINE __attribute__((always_inline))
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __attribute__((always_inline))
#endif
#endif

#ifdef __NVSHMEM_NUMBA_SUPPORT__
// IBGDA support requires host library definitions, NVRTC does not compile them.
// TODO: @wangm to look into ibgda code and see what we should to do support compiling
// the library with nvrtc.
#undef NVSHMEM_IBGDA_SUPPORT
#endif

#if defined __NVSHMEM_NUMBA_SUPPORT__
#undef NVSHMEMI_DEVICE_INLINE
#undef NVSHMEMI_DEVICE_ALWAYS_INLINE
#undef NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE
// See
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__noinline__#inline-hint
// This will not eliminate the symbol from LTOIR, but also marks the signature
// with llvm attribute always_inline, so when the linker sees this and use with another function
// it will aggressively inline the function into its call site.
#define NVSHMEMI_DEVICE_INLINE __inline_hint__
#define NVSHMEMI_DEVICE_ALWAYS_INLINE __inline_hint__
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __inline_hint__
#undef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#endif
