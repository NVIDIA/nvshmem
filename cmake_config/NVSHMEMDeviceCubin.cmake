# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(nvshmem_set_cubin_architectures)
  set(clang_arch "sm_90")
  if(CUDAToolkit_VERSION_MAJOR EQUAL 12)
    set(ptx_arch "ptx82")
  else()
    set(ptx_arch "ptx86")
  endif()

  # LTOIR must use an architecture produced by NVCC; keep the Clang fallback for CUDA 12.
  set(ltoir_arch "${clang_arch}")
  if(CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 ltoir_arch)
    string(REGEX REPLACE "-(real|virtual)$" "" ltoir_arch "${ltoir_arch}")
    set(ltoir_arch "sm_${ltoir_arch}")
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
      set(clang_arch "${ltoir_arch}")
    endif()
  endif()

  set(NVSHMEM_CLANG_ARCH "${clang_arch}" PARENT_SCOPE)
  set(NVSHMEM_LTOIR_ARCH "${ltoir_arch}" PARENT_SCOPE)
  set(NVSHMEM_PTX_ARCH "${ptx_arch}" PARENT_SCOPE)
endfunction()

function(nvshmem_add_ltoir_cubin)
  set(one_value_arguments
      ARCHITECTURE
      CXX_STANDARD
      LIBRARY
      OUTPUT
      SOURCE
      TARGET
      WORKING_DIRECTORY)
  set(multi_value_arguments DEPENDS INCLUDE_OPTIONS)
  cmake_parse_arguments(PARSE_ARGV 0 LTOIR "" "${one_value_arguments}" "${multi_value_arguments}")

  foreach(required_argument IN LISTS one_value_arguments)
    if(NOT LTOIR_${required_argument})
      message(FATAL_ERROR "nvshmem_add_ltoir_cubin requires ${required_argument}")
    endif()
  endforeach()

  get_filename_component(source_name "${LTOIR_SOURCE}" NAME_WE)
  set(intermediate "${source_name}.ltoir")

  add_custom_command(
    OUTPUT "${LTOIR_OUTPUT}"
    COMMAND cuda::nvcc -std=${LTOIR_CXX_STANDARD} -x cu -arch=${LTOIR_ARCHITECTURE} -dlto
            --ltoir ${LTOIR_INCLUDE_OPTIONS} -DNVSHMEM_HOSTLIB_ONLY "${LTOIR_SOURCE}" -o
            "${intermediate}"
    COMMAND cuda::nvcc -arch=${LTOIR_ARCHITECTURE} -dlto -dlink -o "${LTOIR_OUTPUT}"
            "${intermediate}" "${LTOIR_LIBRARY}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${intermediate}"
    WORKING_DIRECTORY "${LTOIR_WORKING_DIRECTORY}"
    DEPENDS "${LTOIR_SOURCE}" ${LTOIR_DEPENDS}
    COMMAND_EXPAND_LISTS
    VERBATIM)

  add_custom_target(${LTOIR_TARGET} ALL DEPENDS "${LTOIR_OUTPUT}")
endfunction()
