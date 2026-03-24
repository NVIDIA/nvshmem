# Common Clang/bitcode toolchain setup for NVSHMEM.
#
# Preconditions:
#   - CUDAToolkit has been found (CUDAToolkit_VERSION_MAJOR/MINOR are set)
#   - NVSHMEM_CLANG_DIR, NVSHMEM_CLANG_CXX_FLAGS_EXTRA may be set by the caller
#
# After inclusion the following variables are available:
#   NVSHMEM_CLANG_EXECUTABLE        – path to the clang binary
#   NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST – list of extra flags for bitcode builds

include_guard(GLOBAL)

# --- Find Clang ----------------------------------------------------------------
if(NOT Clang_FOUND)
  if(NVSHMEM_CLANG_DIR)
    find_package(Clang CONFIG PATHS ${NVSHMEM_CLANG_DIR} NO_DEFAULT_PATH REQUIRED)
  else()
    find_package(Clang CONFIG REQUIRED)
  endif()
endif()

# --- Clang executable ----------------------------------------------------------
if(NOT NVSHMEM_CLANG_EXECUTABLE)
  set(NVSHMEM_CLANG_EXECUTABLE
      "${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
      CACHE PATH "Clang executable used for bitcode library, tests, and perftests")
endif()
message(STATUS "NVSHMEM_CLANG_EXECUTABLE: ${NVSHMEM_CLANG_EXECUTABLE}")

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
# CUDA 13.2+ headers need _NV_RSQRT_SPECIFIER defined when using Clang (workaround for
# crt/math_functions.hpp vs __clang_cuda_runtime_wrapper.h). The noexcept specifier has
# no effect on bitcode generation, so defining it as empty is safe.
# TODO: Remove this workaround if LLVM fixed it upstream.
if(CUDAToolkit_VERSION_MAJOR GREATER 13 OR
   (CUDAToolkit_VERSION_MAJOR EQUAL 13 AND NOT CUDAToolkit_VERSION_MINOR LESS 2))
  list(APPEND NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST "-D_NV_RSQRT_SPECIFIER=")
  message(STATUS
    "Adding -D_NV_RSQRT_SPECIFIER= workaround "
    "(CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}, "
    "LLVM ${LLVM_VERSION_MAJOR})")
endif()
