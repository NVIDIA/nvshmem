# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile-only contracts for the transport-specific preview. These run as a
# build step instead of ordinary executables: the positive fixture must
# compile, and each negative fixture must fail during CUDA compilation with its
# expected diagnostic, before linking or requiring a GPU.

foreach(_required_var CUDA_COMPILER PACKAGE_INCLUDE_DIR TEST_COMMON_INCLUDE_DIR
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

function(compile_fixture name result_var output_var)
  set(_command ${_compiler_command} ${ARGN})

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
    "namespace \"nvshmemx\" has no member|name followed by \"::\" must be a class or namespace name")
expect_compile_failure(optin_required "${_names_absent}" -DCONTRACT_CASE_optin_required)
expect_compile_failure(host_header_only "${_names_absent}" -DCONTRACT_CASE_host_header_only)

# A package built without the feature rejects the consumer opt-in.
expect_compile_failure(
  package_unavailable
  "NVSHMEMX_ENABLE_TRANSPORT_POLICY requires an NVSHMEM package built with NVSHMEM_ENABLE_TRANSPORT_POLICY=ON"
  -DCONTRACT_CASE_package_unavailable)
