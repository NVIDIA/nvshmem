# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common Clang/bitcode toolchain setup for NVSHMEM.
#
# Preconditions:
#   - CUDAToolkit has been found (CUDAToolkit_VERSION_MAJOR/MINOR are set)
#   - NVSHMEM_CLANG_DIR, NVSHMEM_CLANG_CXX_FLAGS_EXTRA may be set by the caller
#
# After inclusion the following variables are available:
#   NVSHMEM_CLANG_EXECUTABLE        – path to the clang binary
#   NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST – list of extra flags for bitcode builds
#   _NVSHMEM_CLANG_BITCODE_WORKAROUND_FLAGS – internal workarounds for all bitcode builds
#   _NVSHMEM_CLANG_CUBIN_WORKAROUND_FLAGS – internal workarounds for test/perftest frontends

include_guard(GLOBAL)

# --- Find Clang ----------------------------------------------------------------
if(NOT Clang_FOUND)
  if(NVSHMEM_CLANG_DIR)
    find_package(Clang CONFIG PATHS ${NVSHMEM_CLANG_DIR} NO_DEFAULT_PATH)
  else()
    find_package(Clang CONFIG)
  endif()
  if(NOT Clang_FOUND)
    message(WARNING "Clang/LLVM not found. Disabling bitcode library build. "
            "Set NVSHMEM_CLANG_DIR or install Clang to enable.")
    set(NVSHMEM_BUILD_BITCODE_LIBRARY OFF CACHE BOOL "" FORCE)
    return()
  endif()
endif()

# --- Clang executable ----------------------------------------------------------
if(NOT NVSHMEM_CLANG_EXECUTABLE)
  set(NVSHMEM_CLANG_EXECUTABLE
      "${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
      CACHE PATH "Clang executable used for bitcode library, tests, and perftests")
endif()
message(STATUS "NVSHMEM_CLANG_EXECUTABLE: ${NVSHMEM_CLANG_EXECUTABLE}")

foreach(_tool opt llc llvm-link llvm-dis llvm-as)
  string(TOUPPER "${_tool}" _TOOL_UC)
  string(REPLACE "-" "_" _TOOL_UC "${_TOOL_UC}")
  find_program(NVSHMEM_${_TOOL_UC}_EXECUTABLE ${_tool}
               HINTS "${LLVM_TOOLS_BINARY_DIR}"
               DOC "Path to ${_tool} used for bitcode/cubin builds")
  set(_NVSHMEM_${_TOOL_UC}_CANDIDATE "")
  if(NVSHMEM_${_TOOL_UC}_EXECUTABLE)
    set(_NVSHMEM_${_TOOL_UC}_CANDIDATE "${NVSHMEM_${_TOOL_UC}_EXECUTABLE}")
  elseif(TARGET llvm::${_tool} AND NOT "${_tool}" STREQUAL "llvm-link")
    get_target_property(_NVSHMEM_${_TOOL_UC}_CANDIDATE
                        llvm::${_tool} IMPORTED_LOCATION)
  endif()
  set(NVSHMEM_${_TOOL_UC}_AVAILABLE FALSE)
  if(_NVSHMEM_${_TOOL_UC}_CANDIDATE)
    execute_process(
      COMMAND "${_NVSHMEM_${_TOOL_UC}_CANDIDATE}" --version
      RESULT_VARIABLE _NVSHMEM_${_TOOL_UC}_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)
    if(_NVSHMEM_${_TOOL_UC}_RESULT EQUAL 0)
      if(NOT TARGET llvm::${_tool})
        add_executable(llvm::${_tool} IMPORTED GLOBAL)
      endif()
      set_target_properties(llvm::${_tool} PROPERTIES
        IMPORTED_LOCATION "${_NVSHMEM_${_TOOL_UC}_CANDIDATE}")
      set(NVSHMEM_${_TOOL_UC}_AVAILABLE TRUE)
    else()
      message(WARNING
              "LLVM tool ${_tool} is not runnable: "
              "${_NVSHMEM_${_TOOL_UC}_CANDIDATE}")
    endif()
  endif()
endforeach()

# --- Extra CXX flags (string -> list) ------------------------------------------
if(NOT DEFINED NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST)
  if(NVSHMEM_CLANG_CXX_FLAGS_EXTRA)
    separate_arguments(NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST NATIVE_COMMAND
                       "${NVSHMEM_CLANG_CXX_FLAGS_EXTRA}")
  else()
    set(NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST "")
  endif()
endif()

# --- Version checks ------------------------------------------------------------
# CUDA 13+ requires LLVM/Clang 22+.
# LLVM_VERSION_MAJOR is set by find_package(Clang CONFIG) -> find_package(LLVM).
# TODO: reenable when LLVM 22 is supported by all dependencies.
#if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13 AND LLVM_VERSION_MAJOR LESS 22)
#  message(FATAL_ERROR
#    "CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} requires "
#    "Clang >= 22 for bitcode/cubin builds, but found LLVM "
#    "${LLVM_PACKAGE_VERSION} (${NVSHMEM_CLANG_EXECUTABLE}).")
#endif()

# --- Workarounds ---------------------------------------------------------------
set(_NVSHMEM_CLANG_BITCODE_WORKAROUND_FLAGS "")
# LLVM 23+ defines _NV_RSQRT_SPECIFIER in its CUDA runtime wrapper. Keep the
# CUDA 13.2+ header workaround for older Clang versions.
# TODO: Remove when older Clang versions are no longer supported.
if(LLVM_VERSION_MAJOR LESS 23 AND
   (CUDAToolkit_VERSION_MAJOR GREATER 13 OR
    (CUDAToolkit_VERSION_MAJOR EQUAL 13 AND NOT CUDAToolkit_VERSION_MINOR LESS 2)))
  list(APPEND _NVSHMEM_CLANG_BITCODE_WORKAROUND_FLAGS "-D_NV_RSQRT_SPECIFIER=")
  message(STATUS
    "Adding -D_NV_RSQRT_SPECIFIER= workaround "
    "(CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}, "
    "LLVM ${LLVM_VERSION_MAJOR})")
endif()

# LLVM 23+ can runtime-unroll calls to external device APIs before linking,
# duplicating their large implementations during inlining. Defer heuristic
# runtime unrolling until the device library is linked, while keeping
# constant-trip-count unrolling and later optimization enabled.
set(_NVSHMEM_CLANG_CUBIN_WORKAROUND_FLAGS "${_NVSHMEM_CLANG_BITCODE_WORKAROUND_FLAGS}")
if(LLVM_VERSION_MAJOR GREATER_EQUAL 23)
  list(APPEND _NVSHMEM_CLANG_CUBIN_WORKAROUND_FLAGS -mllvm -unroll-runtime=false)
endif()
