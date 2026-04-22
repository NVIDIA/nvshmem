# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common CUDA toolkit binary discovery for NVSHMEM custom commands.
#
# Preconditions:
#   - CUDAToolkit has been found (or is discoverable via find_package)
#
# After inclusion, imported executable targets may be available:
#   cuda::nvcc, cuda::nvlink, cuda::ptxas, cuda::fatbinary, cuda::cuobjdump

include_guard(GLOBAL)

if(NOT CUDAToolkit_FOUND)
  find_package(CUDAToolkit REQUIRED)
endif()

foreach(_tool nvcc nvlink ptxas fatbinary cuobjdump)
  string(TOUPPER "${_tool}" _TOOL_UC)
  if(NVSHMEM_${_TOOL_UC}_EXECUTABLE)
    get_filename_component(_TOOL_DIR "${NVSHMEM_${_TOOL_UC}_EXECUTABLE}" DIRECTORY)
    get_filename_component(_TOOL_DIR "${_TOOL_DIR}" REALPATH)
    get_filename_component(_CUDA_TOOLKIT_BIN_DIR "${CUDAToolkit_BIN_DIR}" REALPATH)
    if(NOT _TOOL_DIR STREQUAL _CUDA_TOOLKIT_BIN_DIR)
      unset(NVSHMEM_${_TOOL_UC}_EXECUTABLE CACHE)
      unset(NVSHMEM_${_TOOL_UC}_EXECUTABLE)
    endif()
  endif()
  find_program(NVSHMEM_${_TOOL_UC}_EXECUTABLE ${_tool}
               PATHS "${CUDAToolkit_BIN_DIR}"
               NO_DEFAULT_PATH
               DOC "Path to ${_tool} from the selected CUDA Toolkit")
  if(NVSHMEM_${_TOOL_UC}_EXECUTABLE AND NOT TARGET cuda::${_tool})
    add_executable(cuda::${_tool} IMPORTED GLOBAL)
    set_target_properties(cuda::${_tool} PROPERTIES
      IMPORTED_LOCATION "${NVSHMEM_${_TOOL_UC}_EXECUTABLE}")
  endif()
endforeach()
