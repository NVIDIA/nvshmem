# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
