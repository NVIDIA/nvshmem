# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(nvshmem4rust VERSION 0.1.0 LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

get_filename_component(
    NVSHMEM_SOURCE_DIR_DEFAULT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
)
set(NVSHMEM_SOURCE_DIR "${NVSHMEM_SOURCE_DIR_DEFAULT}" CACHE PATH
    "Path to an NVSHMEM source tree")
set(NVSHMEM_HOME "" CACHE PATH "Path to an installed NVSHMEM tree")
set(NVSHMEM_INCLUDE_DIR "" CACHE PATH "Path to NVSHMEM public headers")
set(NVSHMEM_BUILD_DIR "" CACHE PATH
    "Path to an NVSHMEM build tree, required by the CUDA-Oxide tests")
set(NVSHMEM_CUDA_HOME "$ENV{CUDA_HOME}" CACHE PATH "Path to the CUDA Toolkit")
if(NOT NVSHMEM_CUDA_HOME)
    set(NVSHMEM_CUDA_HOME "/usr/local/cuda" CACHE PATH "Path to the CUDA Toolkit" FORCE)
endif()
if(NOT EXISTS "${NVSHMEM_CUDA_HOME}/include/cuda.h")
    message(FATAL_ERROR
        "NVSHMEM_CUDA_HOME must point to a CUDA Toolkit; missing "
        "include/cuda.h under ${NVSHMEM_CUDA_HOME}")
endif()

if(NOT NVSHMEM_INCLUDE_DIR AND NVSHMEM_HOME)
    set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_HOME}/include")
endif()
if(NOT NVSHMEM_INCLUDE_DIR AND
   EXISTS "${NVSHMEM_SOURCE_DIR}/src/include/nvshmem.h")
    set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_SOURCE_DIR}/src/include")
endif()
if(NOT NVSHMEM_INCLUDE_DIR)
    message(FATAL_ERROR
        "Set NVSHMEM_HOME to an NVSHMEM installation or NVSHMEM_INCLUDE_DIR "
        "to a directory containing NVSHMEM's public headers")
endif()
if(NOT EXISTS "${NVSHMEM_INCLUDE_DIR}/nvshmem.h")
    message(FATAL_ERROR
        "NVSHMEM_INCLUDE_DIR must contain NVSHMEM's public headers; "
        "missing nvshmem.h under ${NVSHMEM_INCLUDE_DIR}")
endif()

set(NVSHMEM4RUST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NVSHMEM4RUST_NUMBAST_VERSION "0.9.0" CACHE STRING
    "Numbast version used to generate the Rust bindings")
set(NVSHMEM4RUST_INSTALL_DIR "lib/nvshmem4rust" CACHE STRING
    "Install directory for generated nvshmem4rust files")
option(NVSHMEM_BUILD_RUST_HOST_ONLY
       "Generate raw host bindings without a CUDA-Oxide checkout" OFF)
option(NVSHMEM_BUILD_RUST_DEVICE_TESTS
       "Configure the CUDA-Oxide smoke and performance tests" OFF)
if(NVSHMEM_BUILD_RUST_HOST_ONLY AND NVSHMEM_BUILD_RUST_DEVICE_TESTS)
    message(FATAL_ERROR
        "NVSHMEM_BUILD_RUST_HOST_ONLY cannot be combined with "
        "NVSHMEM_BUILD_RUST_DEVICE_TESTS")
endif()

set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv")
set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")
set(PYTHON_ENV_STAMP "${CMAKE_BINARY_DIR}/externals/python-env.stamp")
add_custom_command(
    OUTPUT "${PYTHON_ENV_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/externals"
    COMMAND "${Python3_EXECUTABLE}" -m venv "${VENV_DIR}"
    COMMAND "${VENV_PYTHON_EXECUTABLE}" -m pip install --upgrade pip
    COMMAND "${CMAKE_COMMAND}" -E touch "${PYTHON_ENV_STAMP}"
    COMMENT "Creating the nvshmem4rust Python environment"
    VERBATIM
)
add_custom_target(setup_py_bindings_env DEPENDS "${PYTHON_ENV_STAMP}")

include(cmake/NVSHMEM4Rust.cmake)
if(NVSHMEM_BUILD_RUST_HOST_ONLY)
    generateRustBindings(HOST_ONLY VERSION "${NVSHMEM4RUST_NUMBAST_VERSION}")
else()
    generateRustBindings(VERSION "${NVSHMEM4RUST_NUMBAST_VERSION}")
endif()
