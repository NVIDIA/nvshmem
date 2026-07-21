# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select the default C++ / CUDA standard used to build NVSHMEM. The default
# is C++17. Users may override via the NVSHMEM_CXX_STANDARD environment
# variable or -DCMAKE_CXX_STANDARD=XX on the CMake command line. The C++17
# minimum is enforced per target via target_compile_features(cxx_std_17) in
# nvshmem_library_set_base_config().

if(DEFINED ENV{NVSHMEM_CXX_STANDARD})
  set(NVSHMEM_CXX_STANDARD_DEFAULT $ENV{NVSHMEM_CXX_STANDARD})
else()
  set(NVSHMEM_CXX_STANDARD_DEFAULT 17)
endif()
set(CMAKE_CXX_STANDARD ${NVSHMEM_CXX_STANDARD_DEFAULT} CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED On)
set(CMAKE_CUDA_EXTENSIONS Off)

# Map source and build roots so __FILE__ in runtime diagnostics is relative.
macro(nvshmem_apply_file_prefix_maps)
  set(NVSHMEM_FILE_PREFIX_MAP_FLAGS
    "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_BINARY_DIR}=."
  )
  list(REMOVE_DUPLICATES NVSHMEM_FILE_PREFIX_MAP_FLAGS)

  set(NVSHMEM_CUDA_FILE_PREFIX_MAP_FLAGS)
  foreach(_nvshmem_file_prefix_map_flag IN LISTS NVSHMEM_FILE_PREFIX_MAP_FLAGS)
    string(APPEND CMAKE_C_FLAGS " ${_nvshmem_file_prefix_map_flag}")
    string(APPEND CMAKE_CXX_FLAGS " ${_nvshmem_file_prefix_map_flag}")
    if(CMAKE_CUDA_COMPILER_LOADED)
      list(APPEND NVSHMEM_CUDA_FILE_PREFIX_MAP_FLAGS "-Xcompiler=${_nvshmem_file_prefix_map_flag}")
      string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=${_nvshmem_file_prefix_map_flag}")
    endif()
  endforeach()
  unset(_nvshmem_file_prefix_map_flag)
endmacro()
