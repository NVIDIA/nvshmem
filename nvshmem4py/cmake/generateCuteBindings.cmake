# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/ensureNumbast.cmake")

function(generateCuteBindings)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")
    set(NUMBAST_VERSION "0.9.0")
    nvshmem_ensure_numbast(VERSION "${NUMBAST_VERSION}")

    set(CUTE_TREE_DIR "${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute")
    set(CUTE_WORKDIR "${CMAKE_BINARY_DIR}/externals/numbast_cute")
    set(CUTE_CONFIG_OUTPUT "${CUTE_WORKDIR}/config_nvshmem.yml")
    set(CUTE_BINDINGS_SETTINGS "${CUTE_WORKDIR}/generator-settings.cmake")
    set(CUTE_ASTCANOPY_INSTALL_PREFIX
        "${CMAKE_BINARY_DIR}/externals/numbast/build_assets/ast_canopy/install")
    set(CUTEAST_OUTPUT "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/cute/_cuteast.py")
    set(CUTE_HIGH_LEVEL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/cute")
    set(CUTE_RMA_OUTPUT "${CUTE_HIGH_LEVEL_OUTPUT_DIR}/rma.py")
    set(CUTE_COLLECTIVE_OUTPUT "${CUTE_HIGH_LEVEL_OUTPUT_DIR}/collective.py")
    set(CUTE_AMO_OUTPUT "${CUTE_HIGH_LEVEL_OUTPUT_DIR}/amo.py")
    set(CUTE_NUMBAST_LD_LIBRARY_PATH
        "LD_LIBRARY_PATH=${CUTE_ASTCANOPY_INSTALL_PREFIX}/lib:${CUTE_ASTCANOPY_INSTALL_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}")

    set(CUTE_BINDINGS_OUTPUTS
        "${CUTEAST_OUTPUT}"
        "${CUTE_RMA_OUTPUT}"
        "${CUTE_COLLECTIVE_OUTPUT}"
        "${CUTE_AMO_OUTPUT}"
    )
    set(CUTE_BINDINGS_INPUTS
        "${CUTE_TREE_DIR}/entry_point.h"
        "${CUTE_TREE_DIR}/generate_amo.py"
        "${CUTE_TREE_DIR}/generate_collective.py"
        "${CUTE_TREE_DIR}/generate_cute_bindings.py"
        "${CUTE_TREE_DIR}/generate_cute_config.py"
        "${CUTE_TREE_DIR}/generate_rma.py"
        "${CUTE_TREE_DIR}/templates/config_nvshmem.yml.j2"
        "${CUTE_TREE_DIR}/templates/core/device/cute/amo.py.j2"
        "${CUTE_TREE_DIR}/templates/core/device/cute/collective.py.j2"
        "${CUTE_TREE_DIR}/templates/core/device/cute/rma.py.j2"
        "${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/numbast/numbast_common.py"
        "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/generateCuteBindings.cmake"
        "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/ensureNumbast.cmake"
        "${CMAKE_SOURCE_DIR}/nvshmem4py/requirements_build.txt"
    )
    file(GLOB_RECURSE CUTE_NVSHMEM_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/include/*.h"
        "${CMAKE_SOURCE_DIR}/src/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/include/*.cuh"
    )

    file(MAKE_DIRECTORY "${CUTE_WORKDIR}")
    file(CONFIGURE
        OUTPUT "${CUTE_BINDINGS_SETTINGS}"
        CONTENT "CUDA_HOME=@CUDA_HOME@\nNVSHMEM_SOURCE_DIR=@CMAKE_SOURCE_DIR@\nNUMBAST_CONFIG_VERSION=0.1.0\nNUMBAST_VERSION=@NUMBAST_VERSION@\n"
        @ONLY
    )

    add_custom_command(
        OUTPUT ${CUTE_BINDINGS_OUTPUTS}
        BYPRODUCTS "${CUTE_CONFIG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CUTE_WORKDIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CUTE_HIGH_LEVEL_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env "CUDA_HOME=${CUDA_HOME}"
                ${VENV_PYTHON_EXECUTABLE} "${CUTE_TREE_DIR}/generate_cute_config.py"
                --nvshmem-home ${CMAKE_SOURCE_DIR}
                --config-version 0.1.0
                --entry-point-path "${CUTE_TREE_DIR}/entry_point.h"
                --binding-name nvshmem_cute_device_binding_generated.py
                --input-path "${CUTE_TREE_DIR}/templates/config_nvshmem.yml.j2"
                --output-path "${CUTE_CONFIG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E env "CUDA_HOME=${CUDA_HOME}" "${CUTE_NUMBAST_LD_LIBRARY_PATH}"
                ${VENV_PYTHON_EXECUTABLE} "${CUTE_TREE_DIR}/generate_cute_bindings.py"
                --config-path "${CUTE_CONFIG_OUTPUT}"
                --output-path "${CUTEAST_OUTPUT}"
        COMMAND ${VENV_PYTHON_EXECUTABLE} "${CUTE_TREE_DIR}/generate_rma.py"
                --output-dir "${CUTE_HIGH_LEVEL_OUTPUT_DIR}"
        COMMAND ${VENV_PYTHON_EXECUTABLE} "${CUTE_TREE_DIR}/generate_collective.py"
                --output-dir "${CUTE_HIGH_LEVEL_OUTPUT_DIR}"
        COMMAND ${VENV_PYTHON_EXECUTABLE} "${CUTE_TREE_DIR}/generate_amo.py"
                --output-dir "${CUTE_HIGH_LEVEL_OUTPUT_DIR}"
        DEPENDS pip_install_numbast
                ${CUTE_BINDINGS_INPUTS}
                ${CUTE_NVSHMEM_HEADERS}
                "${CUTE_BINDINGS_SETTINGS}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Generating CuTe bindings"
        USES_TERMINAL
        VERBATIM
    )

    add_custom_target(build_bindings_cute DEPENDS ${CUTE_BINDINGS_OUTPUTS})
    if(TARGET build_bindings_numbast)
        add_dependencies(build_bindings_cute build_bindings_numbast)
    endif()
endfunction()
