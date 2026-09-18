# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(_nvshmem_make_header_probes
         output_variable target_prefix language extension include_root preamble)
  set(probe_sources)
  foreach(header IN LISTS ARGN)
    file(RELATIVE_PATH relative_header "${include_root}" "${header}")
    string(MAKE_C_IDENTIFIER "${relative_header}" header_identifier)
    set(probe_source
        "${CMAKE_CURRENT_BINARY_DIR}/header_contracts/${target_prefix}/${language}/${header_identifier}.${extension}")
    file(GENERATE OUTPUT "${probe_source}" CONTENT "${preamble}#include <${relative_header}>\n")
    list(APPEND probe_sources "${probe_source}")
  endforeach()
  set(${output_variable} "${probe_sources}" PARENT_SCOPE)
endfunction()

function(nvshmem_add_header_contract_targets target_prefix include_root)
  file(GLOB_RECURSE internal_c_headers LIST_DIRECTORIES false CONFIGURE_DEPENDS
       "${include_root}/non_abi/c/*.h")
  file(GLOB_RECURSE internal_device_headers LIST_DIRECTORIES false CONFIGURE_DEPENDS
       "${include_root}/non_abi/device/*.cuh"
       "${include_root}/non_abi/device/*.h"
       "${include_root}/non_abi/device/*.hpp")
  file(GLOB_RECURSE public_c_headers LIST_DIRECTORIES false CONFIGURE_DEPENDS
       "${include_root}/c_api/*.h")
  file(GLOB_RECURSE public_cpp_headers LIST_DIRECTORIES false CONFIGURE_DEPENDS
       "${include_root}/cpp_api/*.hpp")
  file(GLOB_RECURSE public_device_headers LIST_DIRECTORIES false CONFIGURE_DEPENDS
       "${include_root}/device/*.cuh"
       "${include_root}/device/*.h"
       "${include_root}/device/*.hpp")

  if(NOT internal_c_headers OR NOT internal_device_headers OR NOT public_c_headers OR
     NOT public_cpp_headers OR NOT public_device_headers)
    message(FATAL_ERROR "Header contract directories are incomplete under ${include_root}")
  endif()

  set(c_headers
      ${internal_c_headers}
      ${public_c_headers}
      "${include_root}/nvshmem_host.h")
  set(cxx_headers
      ${c_headers}
      ${public_cpp_headers}
      "${include_root}/nvshmem.h"
      "${include_root}/nvshmemx.h")
  set(cuda_headers
      ${internal_device_headers}
      ${public_device_headers})

  if(NOT NVSHMEM_GPUNETIO_SUPPORT)
    list(FILTER cuda_headers EXCLUDE REGEX "(gdaki|gpunetio)")
  endif()
  if(NOT NVSHMEM_IBGDA_SUPPORT)
    list(FILTER cuda_headers EXCLUDE REGEX "ibgda")
  endif()

  _nvshmem_make_header_probes(
    c_probe_sources "${target_prefix}" c c "${include_root}" "" ${c_headers})
  _nvshmem_make_header_probes(
    cxx_probe_sources "${target_prefix}" cxx cpp "${include_root}" "" ${cxx_headers})
  _nvshmem_make_header_probes(
    cuda_interface_probe_sources "${target_prefix}" cuda cu "${include_root}" "" ${cxx_headers})
  _nvshmem_make_header_probes(
    cuda_device_probe_sources "${target_prefix}" cuda cu "${include_root}"
    "#include <nvshmem.h>\n" ${cuda_headers})
  set(cuda_probe_sources ${cuda_interface_probe_sources} ${cuda_device_probe_sources})

  add_library(${target_prefix}_c_header_contract OBJECT EXCLUDE_FROM_ALL ${c_probe_sources})
  target_compile_features(${target_prefix}_c_header_contract PRIVATE c_std_11)
  target_include_directories(${target_prefix}_c_header_contract PRIVATE "${include_root}")
  target_link_libraries(${target_prefix}_c_header_contract PRIVATE CUDA::cudart)

  add_library(${target_prefix}_cxx_header_contract OBJECT EXCLUDE_FROM_ALL ${cxx_probe_sources})
  target_compile_features(${target_prefix}_cxx_header_contract PRIVATE cxx_std_17)
  target_include_directories(${target_prefix}_cxx_header_contract PRIVATE "${include_root}")
  target_link_libraries(${target_prefix}_cxx_header_contract PRIVATE CUDA::cudart)
  if(TARGET CCCL::CCCL)
    target_link_libraries(${target_prefix}_cxx_header_contract PRIVATE CCCL::CCCL)
  endif()

  add_library(${target_prefix}_cuda_header_contract OBJECT EXCLUDE_FROM_ALL ${cuda_probe_sources})
  target_compile_features(${target_prefix}_cuda_header_contract PRIVATE cuda_std_17)
  target_include_directories(${target_prefix}_cuda_header_contract PRIVATE "${include_root}")
  if(NVSHMEM_GPUNETIO_SUPPORT)
    target_include_directories(${target_prefix}_cuda_header_contract PRIVATE "${GPUNETIO_INCLUDE}")
  endif()
  target_link_libraries(${target_prefix}_cuda_header_contract PRIVATE CUDA::cudart)
  if(TARGET CCCL::CCCL)
    target_link_libraries(${target_prefix}_cuda_header_contract PRIVATE CCCL::CCCL)
  endif()

  if(NVSHMEM_DEVEL)
    add_custom_target(
      ${target_prefix}_header_contracts ALL
      DEPENDS
        ${target_prefix}_c_header_contract
        ${target_prefix}_cxx_header_contract
        ${target_prefix}_cuda_header_contract)
  else()
    add_custom_target(
      ${target_prefix}_header_contracts
      DEPENDS
        ${target_prefix}_c_header_contract
        ${target_prefix}_cxx_header_contract
        ${target_prefix}_cuda_header_contract)
  endif()
endfunction()
