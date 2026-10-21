# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile-only contracts for the transport-policy API. These run as a
# build step instead of ordinary executables: the positive fixture must
# compile, and each negative fixture must fail during CUDA compilation with its
# expected diagnostic, before linking or requiring a GPU. A final PTX check
# confirms which local memory-space spellings classify with __isShared.

foreach(_required_var CUDA_COMPILER CUDA_GENCODE PACKAGE_INCLUDE_DIR TEST_COMMON_INCLUDE_DIR
                      CONTRACT_SOURCE CONTRACT_BINARY_DIR)
  if(NOT DEFINED ${_required_var})
    message(FATAL_ERROR "transport-specific compile contract: ${_required_var} is required")
  endif()
endforeach()

file(MAKE_DIRECTORY "${CONTRACT_BINARY_DIR}")

set(_compiler_command
    "${CMAKE_COMMAND}" -E env
    CPATH=
    CPLUS_INCLUDE_PATH=
    C_INCLUDE_PATH=
    "${CUDA_COMPILER}"
    -forward-unknown-to-host-compiler)
if(DEFINED CUDA_HOST_COMPILER AND NOT "${CUDA_HOST_COMPILER}" STREQUAL "")
  list(APPEND _compiler_command "-ccbin=${CUDA_HOST_COMPILER}")
endif()
list(APPEND _compiler_command
     "-I${PACKAGE_INCLUDE_DIR}"
     "-I${TEST_COMMON_INCLUDE_DIR}")
foreach(_include_dir IN LISTS CONTRACT_INCLUDE_DIRS)
  list(APPEND _compiler_command "-I${_include_dir}")
endforeach()

function(compile_fixture name result_var output_var)
  # Match the test build's architectures instead of nvcc's toolkit-dependent
  # default, which can be older than the device headers support.
  set(_command ${_compiler_command} ${CUDA_GENCODE} ${ARGN})

  set(_object "${CONTRACT_BINARY_DIR}/${name}.o")
  file(REMOVE "${_object}")
  list(APPEND _command -std=c++17 -rdc=true -c "${CONTRACT_SOURCE}" -o "${_object}")
  execute_process(
    COMMAND ${_command}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(${result_var} "${_result}" PARENT_SCOPE)
  set(${output_var} "${_stdout}\n${_stderr}" PARENT_SCOPE)
endfunction()

function(expect_compile_success name)
  compile_fixture(${name} _result _output ${ARGN})
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
            "transport-specific compile contract '${name}' failed to compile:\n${_output}")
  endif()
endfunction()

function(expect_compile_failure name diagnostic)
  compile_fixture(${name} _result _output ${ARGN})
  if(_result EQUAL 0)
    message(FATAL_ERROR
            "transport-specific compile contract '${name}' unexpectedly compiled successfully")
  endif()
  if(NOT _output MATCHES "${diagnostic}")
    message(FATAL_ERROR
            "transport-specific compile contract '${name}' failed for an unexpected reason:\n${_output}")
  endif()
endfunction()

expect_compile_success(legal -DCONTRACT_CASE_legal)

# Without the source opt-in nvshmemx.h, and nvshmem_host.h even with it, must
# leave every transport-policy name undeclared.
set(_names_absent
    "namespace \"nvshmemx\" has no member|name followed by \"::\" must be a class or namespace name|nvshmemx_put\" is undefined")
expect_compile_failure(optin_required "${_names_absent}" -DCONTRACT_CASE_optin_required)
expect_compile_failure(host_header_only "${_names_absent}" -DCONTRACT_CASE_host_header_only)

# A package built without the feature rejects the consumer opt-in.
expect_compile_failure(
  package_unavailable
  "NVSHMEMX_ENABLE_TRANSPORT_POLICY requires an NVSHMEM package built with NVSHMEM_ENABLE_TRANSPORT_POLICY=ON"
  -DCONTRACT_CASE_package_unavailable)

# A pair without a route is rejected with one diagnostic: CFT / Network and
# Pointer / Network have none, and CFT / CE and CFT / SM have none in this
# series.
foreach(_pair invalid_pair pointer_network_pair cft_ce_pair cft_sm_pair)
  expect_compile_failure(
    ${_pair}
    "unsupported addressing/transport-policy pair"
    -DCONTRACT_CASE_${_pair})
endforeach()
# CFT / TMA is block-scope only for every operation: RMA, signals, and
# ordering below block scope are rejected with one diagnostic.
foreach(_case cft_rma_thread cft_rma_warp cft_signal_warp cft_ordering_warp)
  expect_compile_failure(
    ${_case}
    "CFT/TMA is block-scope only"
    -DCONTRACT_CASE_${_case})
endforeach()
# Pointer / TMA gets, and puts from an explicit global source, use CTA-wide
# TMA state and are rejected below block scope.
expect_compile_failure(
  tma_get_below_block
  "Pointer/TMA get and get_nbi are block-scope only"
  -DCONTRACT_CASE_tma_get_below_block)
expect_compile_failure(
  tma_global_put_below_block
  "Pointer/TMA put from a global source is block-scope only"
  -DCONTRACT_CASE_tma_global_put_below_block)
expect_compile_failure(
  remote_put_tag
  "transport-specific put destination must be remote symmetric global memory"
  -DCONTRACT_CASE_remote_put_tag)
expect_compile_failure(
  remote_get_tag
  "transport-specific get source must be remote symmetric global memory"
  -DCONTRACT_CASE_remote_get_tag)
expect_compile_failure(
  source_category_tag
  "no instance of function template \"nvshmemx_put\" matches"
  -DCONTRACT_CASE_source_category_tag)
expect_compile_failure(
  destination_category_tag
  "no instance of function template \"nvshmemx_get\" matches"
  -DCONTRACT_CASE_destination_category_tag)

# Element types are exactly the stable typed RMA types, cv-unqualified.
set(_element_type "transport-specific RMA element type must be a standard NVSHMEM RMA type")
expect_compile_failure(unsupported_type "${_element_type}" -DCONTRACT_CASE_unsupported_type)
expect_compile_failure(cv_qualified_type "${_element_type}" -DCONTRACT_CASE_cv_qualified_type)

# A P2P-only package must not need network transport symbols. The fixture
# instantiates every Auto / Network API at every scope; in the P2P-only
# emulation its device object may reference no transfer, proxy, IBGDA, or
# GPUNetIO symbol. The control build against this package must reference the
# transfer primitives when the package links them from its device library.
get_filename_component(_cuda_bin_dir "${CUDA_COMPILER}" DIRECTORY)
find_program(_cuobjdump cuobjdump HINTS "${_cuda_bin_dir}" NO_DEFAULT_PATH)
if(NOT _cuobjdump)
  find_program(_cuobjdump cuobjdump)
endif()
if(NOT _cuobjdump)
  message(FATAL_ERROR "transport-specific compile contract: cuobjdump not found")
endif()

function(device_symbols name output_var)
  set(_object "${CONTRACT_BINARY_DIR}/${name}.o")
  execute_process(
    COMMAND "${_cuobjdump}" -elf "${_object}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
            "transport-specific compile contract '${name}': cuobjdump failed:\n${_stderr}")
  endif()
  set(${output_var} "${_stdout}" PARENT_SCOPE)
endfunction()

set(_network_symbol_re "nvshmemi_(transfer|proxy|ibgda|gdaki)_[A-Za-z_]+")
expect_compile_success(network_p2p_only -DCONTRACT_CASE_network_p2p_only)
device_symbols(network_p2p_only _p2p_only_symbols)
if(NOT _p2p_only_symbols MATCHES "network_symbols")
  message(FATAL_ERROR
          "transport-specific compile contract 'network_p2p_only': the fixture kernel is "
          "missing from the device object")
endif()
string(REGEX MATCHALL "${_network_symbol_re}" _p2p_only_network "${_p2p_only_symbols}")
if(_p2p_only_network)
  list(REMOVE_DUPLICATES _p2p_only_network)
  message(FATAL_ERROR
          "transport-specific compile contract 'network_p2p_only': a P2P-only Auto / Network "
          "consumer references network transport symbols: ${_p2p_only_network}")
endif()

expect_compile_success(network_symbols -DCONTRACT_CASE_network_symbols)
device_symbols(network_symbols _control_symbols)
if(_control_symbols MATCHES "transport_specific_contract_expects_transfer_symbols" AND
   NOT _control_symbols MATCHES "nvshmemi_transfer_[A-Za-z_]+")
  message(FATAL_ERROR
          "transport-specific compile contract 'network_symbols': the control object has no "
          "nvshmemi_transfer_* reference, so the P2P-only symbol check can no longer observe one")
endif()

# __isShared lowers to the PTX shared-window test. Explicit shared and global
# local operands of Pointer / SM, Pointer / TMA, CFT / TMA, and Auto / Network must
# compile without it, while the generic spelling must still classify.
function(compile_ptx name)
  set(_ptx "${CONTRACT_BINARY_DIR}/${name}.ptx")
  file(REMOVE "${_ptx}")
  execute_process(
    COMMAND ${_compiler_command} ${ARGN} -std=c++17 -rdc=true -arch=sm_90 -ptx
            "${CONTRACT_SOURCE}" -o "${_ptx}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
            "transport-specific compile contract '${name}' failed to compile:\n${_stdout}\n${_stderr}")
  endif()
  file(STRINGS "${_ptx}" _shared_tests REGEX "isspacep\\.shared")
  list(LENGTH _shared_tests _count)
  set(${name}_shared_tests ${_count} PARENT_SCOPE)
endfunction()

foreach(_space shared global generic)
  compile_ptx(memory_space_${_space} -DCONTRACT_CASE_memory_space_ptx -DLOCAL_SPACE=${_space})
  set(_count ${memory_space_${_space}_shared_tests})
  if(_space STREQUAL "generic")
    if(_count EQUAL 0)
      message(FATAL_ERROR
              "transport-specific compile contract 'memory_space_generic' has no __isShared "
              "classification; the PTX check can no longer observe it")
    endif()
  elseif(NOT _count EQUAL 0)
    message(FATAL_ERROR
            "transport-specific compile contract 'memory_space_${_space}': explicit ${_space} "
            "instantiations contain ${_count} isspacep.shared (__isShared) instructions")
  endif()
endforeach()
