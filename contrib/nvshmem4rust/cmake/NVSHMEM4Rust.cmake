# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(generateRustBindings)
    cmake_parse_arguments(PARSE_ARGV 0 GENERATERUST "HOST_ONLY" "VERSION" "")
    if(NOT GENERATERUST_VERSION)
        set(GENERATERUST_VERSION "0.9.0")
    endif()

    set(PACKAGE_NAME "numbast_rust")
    set(WORKDIR "${CMAKE_BINARY_DIR}/externals/${PACKAGE_NAME}")
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/externals/output")
    set(GENERATOR_DIR "${NVSHMEM4RUST_SOURCE_DIR}/generator")
    set(RUNTIME_SOURCE_DIR "${NVSHMEM4RUST_SOURCE_DIR}/runtime")
    set(TEST_SOURCE_DIR "${NVSHMEM4RUST_SOURCE_DIR}/tests")
    set(RUST_DEVICE_CONFIG_PATH "${WORKDIR}/config_nvshmem_device.yml")
    set(RUST_HOST_CONFIG_PATH "${WORKDIR}/config_nvshmem_host.yml")
    set(DEVICE_OUTPUT_NAME "nvshmem_device_cuda_oxide.rs")
    set(HOST_OUTPUT_NAME "nvshmem_host.rs")
    set(HOST_API_OUTPUT_NAME "nvshmem_host_api.rs")

    set(NVSHMEM_RUST_BINDINGS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated"
        CACHE PATH "Directory for generated NVSHMEM Rust bindings")
    set(RUST_HOST_BINDINGS_OUTPUT
        "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/${HOST_OUTPUT_NAME}")
    set(RUST_HOST_API_OUTPUT
        "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/${HOST_API_OUTPUT_NAME}")
    if(NOT GENERATERUST_HOST_ONLY)
        set(RUST_DEVICE_BINDINGS_OUTPUT
            "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/${DEVICE_OUTPUT_NAME}")
        set(RUST_RUNTIME_BINARY_DIR
            "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/nvshmem_host_runtime")
        set(RUST_RUNTIME_HOST_BINDINGS_OUTPUT
            "${RUST_RUNTIME_BINARY_DIR}/src/bindings.rs")
        set(RUST_RUNTIME_HOST_API_OUTPUT
            "${RUST_RUNTIME_BINARY_DIR}/src/api.rs")
    endif()

    set(NVSHMEM_CUDA_OXIDE_ROOT "" CACHE PATH
        "Path to a CUDA-Oxide checkout")
    set(NVSHMEM_CARGO_OXIDE_EXECUTABLE "" CACHE FILEPATH
        "Path to the cargo-oxide executable")
    set(NVSHMEM_RUST_TEST_ARCH "sm_90" CACHE STRING
        "CUDA SM target for the CUDA-Oxide tests")
    set(NVSHMEM_RUST_TEST_DEVICE_LTOIR "" CACHE FILEPATH
        "NVSHMEM device LTOIR passed to the CUDA-Oxide tests")
    set(NVSHMEM_HOST_LIB_DIR "" CACHE PATH
        "Directory containing libnvshmem_host for the CUDA-Oxide tests")
    if(NOT GENERATERUST_HOST_ONLY AND
       NOT EXISTS "${NVSHMEM_CUDA_OXIDE_ROOT}/crates/cuda-core")
        message(FATAL_ERROR
            "Set NVSHMEM_CUDA_OXIDE_ROOT to a CUDA-Oxide checkout so the "
            "generated runtime crate has a valid cuda-core dependency")
    endif()

    set(NVSHMEM_BINDING_HEADERS
        "${NVSHMEM_INCLUDE_DIR}/nvshmem_host.h"
        "${NVSHMEM_INCLUDE_DIR}/host/nvshmem_api.h"
        "${NVSHMEM_INCLUDE_DIR}/host/nvshmem_coll_api.h"
        "${NVSHMEM_INCLUDE_DIR}/host/nvshmemx_api.h"
        "${NVSHMEM_INCLUDE_DIR}/host/nvshmemx_coll_api.h")
    if(NOT GENERATERUST_HOST_ONLY)
        list(APPEND NVSHMEM_BINDING_HEADERS
            "${NVSHMEM_INCLUDE_DIR}/nvshmem.h"
            "${NVSHMEM_INCLUDE_DIR}/nvshmemx.h"
            "${NVSHMEM_INCLUDE_DIR}/device/nvshmem_coll_defines.cuh"
            "${NVSHMEM_INCLUDE_DIR}/device/nvshmem_defines.h"
            "${NVSHMEM_INCLUDE_DIR}/device/nvshmemx_coll_defines.cuh"
            "${NVSHMEM_INCLUDE_DIR}/device/nvshmemx_defines.h")
    endif()
    foreach(NVSHMEM_BINDING_HEADER IN LISTS NVSHMEM_BINDING_HEADERS)
        if(NOT EXISTS "${NVSHMEM_BINDING_HEADER}")
            message(FATAL_ERROR
                "NVSHMEM public header required for binding generation is missing: "
                "${NVSHMEM_BINDING_HEADER}")
        endif()
    endforeach()
    file(GLOB_RECURSE NVSHMEM_BINDING_DEPENDENCIES CONFIGURE_DEPENDS
        "${NVSHMEM_INCLUDE_DIR}/*.h"
        "${NVSHMEM_INCLUDE_DIR}/*.hpp"
        "${NVSHMEM_INCLUDE_DIR}/*.cuh")

    set(NUMBAST_CONFIG_VERSION "0.1.0")
    set(ASTCANOPY_CMAKE_INSTALL_PREFIX "${WORKDIR}/ast_canopy/install")
    set(NUMBAST_LD_LIBRARY_PATH
        "LD_LIBRARY_PATH=${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib:${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}")

    file(MAKE_DIRECTORY "${WORKDIR}")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}")
    if(NOT GENERATERUST_HOST_ONLY)
        file(MAKE_DIRECTORY "${RUST_RUNTIME_BINARY_DIR}/src")
    endif()
    set(GENERATOR_SETTINGS "${WORKDIR}/generator-settings.txt")
    file(CONFIGURE OUTPUT "${GENERATOR_SETTINGS}" CONTENT
        "NVSHMEM_INCLUDE_DIR=@NVSHMEM_INCLUDE_DIR@\nNVSHMEM_CUDA_HOME=@NVSHMEM_CUDA_HOME@\nNVSHMEM_RUST_TEST_ARCH=@NVSHMEM_RUST_TEST_ARCH@\n"
        @ONLY)

    if(NOT GENERATERUST_HOST_ONLY)
        configure_file("${RUNTIME_SOURCE_DIR}/Cargo.toml.in"
                       "${RUST_RUNTIME_BINARY_DIR}/Cargo.toml" @ONLY)
        configure_file("${RUNTIME_SOURCE_DIR}/build.rs"
                       "${RUST_RUNTIME_BINARY_DIR}/build.rs" COPYONLY)
        configure_file("${RUNTIME_SOURCE_DIR}/src/lib.rs"
                       "${RUST_RUNTIME_BINARY_DIR}/src/lib.rs" COPYONLY)
        configure_file("${RUNTIME_SOURCE_DIR}/README.md"
                       "${RUST_RUNTIME_BINARY_DIR}/README.md" COPYONLY)
    endif()

    set(NUMBAST_INSTALL_STAMP
        "${OUTPUT_DIR}/numbast-${GENERATERUST_VERSION}.stamp")
    add_custom_command(
        OUTPUT "${NUMBAST_INSTALL_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${WORKDIR}"
        COMMAND "${VENV_PYTHON_EXECUTABLE}" -m pip install
                -r "${NVSHMEM4RUST_SOURCE_DIR}/requirements-build.txt"
        COMMAND "${VENV_PYTHON_EXECUTABLE}" -m pip install
                "numbast==${GENERATERUST_VERSION}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${NUMBAST_INSTALL_STAMP}"
        DEPENDS setup_py_bindings_env
                "${NVSHMEM4RUST_SOURCE_DIR}/requirements-build.txt"
        WORKING_DIRECTORY "${WORKDIR}"
        COMMENT "Installing Numbast ${GENERATERUST_VERSION}"
        VERBATIM
    )
    add_custom_target(pip_install_numbast_rust DEPENDS "${NUMBAST_INSTALL_STAMP}")

    if(GENERATERUST_HOST_ONLY)
        add_custom_command(
            OUTPUT "${RUST_HOST_BINDINGS_OUTPUT}"
                   "${RUST_HOST_API_OUTPUT}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${WORKDIR}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E env "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                    "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_config.py"
                    --nvshmem-include-dir "${NVSHMEM_INCLUDE_DIR}"
                    --gpu-arch "${NVSHMEM_RUST_TEST_ARCH}"
                    --config-version "${NUMBAST_CONFIG_VERSION}"
                    --entry-point-path "${GENERATOR_DIR}/host_entry_point.h"
                    --binding-name "${HOST_OUTPUT_NAME}"
                    --input-path "${GENERATOR_DIR}/templates/config_nvshmem_host.yml.j2"
                    --output-path "${RUST_HOST_CONFIG_PATH}"
            COMMAND "${CMAKE_COMMAND}" -E env
                    "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                    "CUDA_PATH=${NVSHMEM_CUDA_HOME}"
                    "${NUMBAST_LD_LIBRARY_PATH}"
                    "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_bindings.py"
                    --config-path "${RUST_HOST_CONFIG_PATH}"
                    --output-path "${RUST_HOST_BINDINGS_OUTPUT}"
                    --binding-kind host
            COMMAND "${CMAKE_COMMAND}" -E env
                    "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                    "CUDA_PATH=${NVSHMEM_CUDA_HOME}"
                    "${NUMBAST_LD_LIBRARY_PATH}"
                    "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_bindings.py"
                    --config-path "${RUST_HOST_CONFIG_PATH}"
                    --output-path "${RUST_HOST_API_OUTPUT}"
                    --binding-kind host-api
            DEPENDS pip_install_numbast_rust
                    "${GENERATOR_DIR}/host_entry_point.h"
                    "${GENERATOR_DIR}/generate_rust_config.py"
                    "${GENERATOR_DIR}/generate_rust_bindings.py"
                    "${GENERATOR_DIR}/host_api_surface.py"
                    "${GENERATOR_DIR}/templates/config_nvshmem_host.yml.j2"
                    "${GENERATOR_SETTINGS}"
                    ${NVSHMEM_BINDING_DEPENDENCIES}
            WORKING_DIRECTORY "${NVSHMEM4RUST_SOURCE_DIR}"
            COMMENT "Generating NVSHMEM host Rust bindings"
            USES_TERMINAL
            VERBATIM
        )

        add_custom_target(build_bindings_rust ALL
            DEPENDS "${RUST_HOST_BINDINGS_OUTPUT}"
                    "${RUST_HOST_API_OUTPUT}")

        install(FILES "${RUST_HOST_BINDINGS_OUTPUT}"
                      "${RUST_HOST_API_OUTPUT}"
                DESTINATION "${NVSHMEM4RUST_INSTALL_DIR}" OPTIONAL)
        return()
    endif()

    add_custom_command(
        OUTPUT "${RUST_DEVICE_BINDINGS_OUTPUT}"
               "${RUST_HOST_BINDINGS_OUTPUT}"
               "${RUST_HOST_API_OUTPUT}"
               "${RUST_RUNTIME_HOST_BINDINGS_OUTPUT}"
               "${RUST_RUNTIME_HOST_API_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${WORKDIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${RUST_RUNTIME_BINARY_DIR}/src"
        COMMAND "${CMAKE_COMMAND}" -E env "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_config.py"
                --nvshmem-include-dir "${NVSHMEM_INCLUDE_DIR}"
                --gpu-arch "${NVSHMEM_RUST_TEST_ARCH}"
                --config-version "${NUMBAST_CONFIG_VERSION}"
                --entry-point-path "${GENERATOR_DIR}/entry_point.h"
                --binding-name "${DEVICE_OUTPUT_NAME}"
                --input-path "${GENERATOR_DIR}/templates/config_nvshmem.yml.j2"
                --output-path "${RUST_DEVICE_CONFIG_PATH}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "CUDA_PATH=${NVSHMEM_CUDA_HOME}"
                "${NUMBAST_LD_LIBRARY_PATH}"
                "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_bindings.py"
                --config-path "${RUST_DEVICE_CONFIG_PATH}"
                --output-path "${RUST_DEVICE_BINDINGS_OUTPUT}"
                --binding-kind device
        COMMAND "${CMAKE_COMMAND}" -E env "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_config.py"
                --nvshmem-include-dir "${NVSHMEM_INCLUDE_DIR}"
                --gpu-arch "${NVSHMEM_RUST_TEST_ARCH}"
                --config-version "${NUMBAST_CONFIG_VERSION}"
                --entry-point-path "${GENERATOR_DIR}/host_entry_point.h"
                --binding-name "${HOST_OUTPUT_NAME}"
                --input-path "${GENERATOR_DIR}/templates/config_nvshmem_host.yml.j2"
                --output-path "${RUST_HOST_CONFIG_PATH}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "CUDA_PATH=${NVSHMEM_CUDA_HOME}"
                "${NUMBAST_LD_LIBRARY_PATH}"
                "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_bindings.py"
                --config-path "${RUST_HOST_CONFIG_PATH}"
                --output-path "${RUST_HOST_BINDINGS_OUTPUT}"
                --binding-kind host
        COMMAND "${CMAKE_COMMAND}" -E env
                "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "CUDA_PATH=${NVSHMEM_CUDA_HOME}"
                "${NUMBAST_LD_LIBRARY_PATH}"
                "${VENV_PYTHON_EXECUTABLE}" "${GENERATOR_DIR}/generate_rust_bindings.py"
                --config-path "${RUST_HOST_CONFIG_PATH}"
                --output-path "${RUST_HOST_API_OUTPUT}"
                --binding-kind host-api
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${RUST_HOST_BINDINGS_OUTPUT}"
                "${RUST_RUNTIME_HOST_BINDINGS_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${RUST_HOST_API_OUTPUT}"
                "${RUST_RUNTIME_HOST_API_OUTPUT}"
        DEPENDS pip_install_numbast_rust
                "${GENERATOR_DIR}/entry_point.h"
                "${GENERATOR_DIR}/host_entry_point.h"
                "${GENERATOR_DIR}/generate_rust_config.py"
                "${GENERATOR_DIR}/generate_rust_bindings.py"
                "${GENERATOR_DIR}/host_api_surface.py"
                "${GENERATOR_DIR}/templates/config_nvshmem.yml.j2"
                "${GENERATOR_DIR}/templates/config_nvshmem_host.yml.j2"
                "${GENERATOR_SETTINGS}"
                ${NVSHMEM_BINDING_DEPENDENCIES}
                "${RUNTIME_SOURCE_DIR}/Cargo.toml.in"
                "${RUNTIME_SOURCE_DIR}/build.rs"
                "${RUNTIME_SOURCE_DIR}/README.md"
                "${RUNTIME_SOURCE_DIR}/src/lib.rs"
        WORKING_DIRECTORY "${NVSHMEM4RUST_SOURCE_DIR}"
        COMMENT "Generating NVSHMEM host and CUDA-Oxide Rust bindings"
        USES_TERMINAL
        VERBATIM
    )

    add_custom_target(build_bindings_rust ALL
        DEPENDS "${RUST_DEVICE_BINDINGS_OUTPUT}"
                "${RUST_HOST_BINDINGS_OUTPUT}"
                "${RUST_HOST_API_OUTPUT}"
                "${RUST_RUNTIME_HOST_BINDINGS_OUTPUT}"
                "${RUST_RUNTIME_HOST_API_OUTPUT}")

    if(NVSHMEM_BUILD_RUST_DEVICE_TESTS)
        find_program(CARGO_EXECUTABLE cargo REQUIRED)
        if(NOT NVSHMEM_CARGO_OXIDE_EXECUTABLE)
            find_program(CARGO_OXIDE_EXECUTABLE cargo-oxide)
            if(CARGO_OXIDE_EXECUTABLE)
                set(NVSHMEM_CARGO_OXIDE_EXECUTABLE "${CARGO_OXIDE_EXECUTABLE}"
                    CACHE FILEPATH "Path to the cargo-oxide executable" FORCE)
            endif()
        endif()
        if(NOT NVSHMEM_CARGO_OXIDE_EXECUTABLE)
            message(FATAL_ERROR
                "NVSHMEM_BUILD_RUST_DEVICE_TESTS requires cargo-oxide in PATH "
                "or -DNVSHMEM_CARGO_OXIDE_EXECUTABLE=/path/to/cargo-oxide")
        endif()
        if(NOT EXISTS "${NVSHMEM_CARGO_OXIDE_EXECUTABLE}")
            message(FATAL_ERROR
                "NVSHMEM_CARGO_OXIDE_EXECUTABLE does not exist: "
                "${NVSHMEM_CARGO_OXIDE_EXECUTABLE}")
        endif()

        foreach(CUDA_OXIDE_CRATE cuda-device cuda-host libnvvm-sys nvjitlink-sys)
            if(NOT EXISTS "${NVSHMEM_CUDA_OXIDE_ROOT}/crates/${CUDA_OXIDE_CRATE}")
                message(FATAL_ERROR
                    "CUDA-Oxide checkout is missing crates/${CUDA_OXIDE_CRATE}: "
                    "${NVSHMEM_CUDA_OXIDE_ROOT}")
            endif()
        endforeach()

        if(NOT NVSHMEM_HOST_LIB_DIR AND NVSHMEM_BUILD_DIR)
            set(NVSHMEM_HOST_LIB_DIR "${NVSHMEM_BUILD_DIR}/src/lib")
        endif()
        if(NOT NVSHMEM_HOST_LIB_DIR)
            message(FATAL_ERROR
                "NVSHMEM_BUILD_RUST_DEVICE_TESTS requires "
                "-DNVSHMEM_HOST_LIB_DIR=/path/to/nvshmem/build/src/lib")
        endif()
        if(NOT EXISTS "${NVSHMEM_HOST_LIB_DIR}/libnvshmem_host.so")
            message(FATAL_ERROR
                "NVSHMEM_HOST_LIB_DIR does not contain libnvshmem_host.so: "
                "${NVSHMEM_HOST_LIB_DIR}")
        endif()

        if(NOT NVSHMEM_RUST_TEST_DEVICE_LTOIR AND NVSHMEM_BUILD_DIR)
            set(NVSHMEM_RUST_TEST_DEVICE_LTOIR
                "${NVSHMEM_BUILD_DIR}/src/lib/libnvshmem_device.ltoir.fatbin")
        endif()
        if(NOT NVSHMEM_RUST_TEST_DEVICE_LTOIR)
            message(FATAL_ERROR
                "NVSHMEM_BUILD_RUST_DEVICE_TESTS requires "
                "-DNVSHMEM_RUST_TEST_DEVICE_LTOIR=/path/to/libnvshmem_device.ltoir.fatbin")
        endif()
        if(NOT EXISTS "${NVSHMEM_RUST_TEST_DEVICE_LTOIR}")
            message(FATAL_ERROR
                "NVSHMEM_RUST_TEST_DEVICE_LTOIR does not exist: "
                "${NVSHMEM_RUST_TEST_DEVICE_LTOIR}")
        endif()

        set(RUST_TEST_SUPPORT_BINARY_DIR
            "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/cuda_oxide_test_support")
        file(MAKE_DIRECTORY "${RUST_TEST_SUPPORT_BINARY_DIR}/src")
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_test_support/Cargo.toml.in"
                       "${RUST_TEST_SUPPORT_BINARY_DIR}/Cargo.toml" @ONLY)
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_test_support/src/lib.rs"
                       "${RUST_TEST_SUPPORT_BINARY_DIR}/src/lib.rs" COPYONLY)

        set(RUST_TEST_BINARY_DIR
            "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/cuda_oxide_smoke")
        file(MAKE_DIRECTORY "${RUST_TEST_BINARY_DIR}/src")
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_smoke/Cargo.toml.in"
                       "${RUST_TEST_BINARY_DIR}/Cargo.toml" @ONLY)
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_smoke/src/main.rs"
                       "${RUST_TEST_BINARY_DIR}/src/main.rs" COPYONLY)

        set(RUST_PERF_BINARY_DIR
            "${NVSHMEM_RUST_BINDINGS_OUTPUT_DIR}/cuda_oxide_perf")
        file(MAKE_DIRECTORY "${RUST_PERF_BINARY_DIR}/src")
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_perf/Cargo.toml.in"
                       "${RUST_PERF_BINARY_DIR}/Cargo.toml" @ONLY)
        configure_file("${TEST_SOURCE_DIR}/cuda_oxide_perf/src/main.rs"
                       "${RUST_PERF_BINARY_DIR}/src/main.rs" COPYONLY)

        add_custom_target(
            test_bindings_rust_cuda_oxide
            COMMAND "${CMAKE_COMMAND}" -E env
                "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "CUDA_TOOLKIT_PATH=${NVSHMEM_CUDA_HOME}"
                "CUDA_OXIDE_TARGET=${NVSHMEM_RUST_TEST_ARCH}"
                "NVSHMEM_RUST_BINDINGS=${RUST_DEVICE_BINDINGS_OUTPUT}"
                "NVSHMEM_DEVICE_LTOIR=${NVSHMEM_RUST_TEST_DEVICE_LTOIR}"
                "NVSHMEM_HOST_LIB_DIR=${NVSHMEM_HOST_LIB_DIR}"
                "NVSHMEM_RUST_INIT=uid"
                "NVSHMEM_RUST_WRITE_ARTIFACTS=1"
                "LD_LIBRARY_PATH=${NVSHMEM_HOST_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
                "${NVSHMEM_CARGO_OXIDE_EXECUTABLE}" run
                --emit-nvvm-ir --arch=${NVSHMEM_RUST_TEST_ARCH}
            WORKING_DIRECTORY "${RUST_TEST_BINARY_DIR}"
            DEPENDS build_bindings_rust
            COMMENT "Running the CUDA-Oxide Rust NVSHMEM smoke tests"
            USES_TERMINAL
            VERBATIM
        )

        add_custom_target(
            test_bindings_rust_cuda_oxide_perf
            COMMAND "${CMAKE_COMMAND}" -E env
                "CUDA_HOME=${NVSHMEM_CUDA_HOME}"
                "CUDA_TOOLKIT_PATH=${NVSHMEM_CUDA_HOME}"
                "CUDA_OXIDE_TARGET=${NVSHMEM_RUST_TEST_ARCH}"
                "NVSHMEM_RUST_BINDINGS=${RUST_DEVICE_BINDINGS_OUTPUT}"
                "NVSHMEM_DEVICE_LTOIR=${NVSHMEM_RUST_TEST_DEVICE_LTOIR}"
                "NVSHMEM_HOST_LIB_DIR=${NVSHMEM_HOST_LIB_DIR}"
                "NVSHMEM_RUST_COMPILE_ONLY=1"
                "LD_LIBRARY_PATH=${NVSHMEM_HOST_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
                "${NVSHMEM_CARGO_OXIDE_EXECUTABLE}" run
                --emit-nvvm-ir --arch=${NVSHMEM_RUST_TEST_ARCH}
            WORKING_DIRECTORY "${RUST_PERF_BINARY_DIR}"
            DEPENDS build_bindings_rust
            COMMENT "Building the CUDA-Oxide Rust NVSHMEM performance tests"
            USES_TERMINAL
            VERBATIM
        )
    endif()

    install(FILES "${RUST_DEVICE_BINDINGS_OUTPUT}"
                  "${RUST_HOST_BINDINGS_OUTPUT}"
                  "${RUST_HOST_API_OUTPUT}"
            DESTINATION "${NVSHMEM4RUST_INSTALL_DIR}" OPTIONAL)
endfunction()
