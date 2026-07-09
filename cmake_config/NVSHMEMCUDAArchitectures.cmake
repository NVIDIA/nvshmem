# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(nvshmem_get_default_cuda_architectures OUT_VAR)
  if(CUDAToolkit_VERSION VERSION_LESS 12.8)
    set(_NVSHMEM_DEFAULT_CUDA_ARCHITECTURES "70-real;80-real;89-real;90")
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.4)
    set(_NVSHMEM_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;100-real;107-real;120")
  elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
    set(_NVSHMEM_DEFAULT_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;100-real;120")
  else()
    set(_NVSHMEM_DEFAULT_CUDA_ARCHITECTURES "70-real;80-real;89-real;90-real;100-real;120")
  endif()

  set(${OUT_VAR} "${_NVSHMEM_DEFAULT_CUDA_ARCHITECTURES}" PARENT_SCOPE)
endfunction()

function(nvshmem_set_default_cuda_architectures)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
      set(_NVSHMEM_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
    else()
      nvshmem_get_default_cuda_architectures(_NVSHMEM_CUDA_ARCHITECTURES)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${_NVSHMEM_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA ARCHITECTURES" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES "${_NVSHMEM_CUDA_ARCHITECTURES}" PARENT_SCOPE)
  endif()
endfunction()
