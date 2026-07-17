# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

# Options for controlling nvshmem4py wheel builds
option(NVSHMEM4PY_BUILD_ALL_WHEELS "Build all nvshmem4py wheels automatically during normal build" ON)
set(NVSHMEM4PY_PYTHON_VERSIONS "" CACHE STRING "Specific Python versions to build wheels for (e.g., '3.12' or '3.12;3.11'). If empty, all available versions >= 3.9 are used.")
set(NVSHMEM4PY_CUDA_VERSIONS "" CACHE STRING "Specific CUDA versions to build wheels for (e.g., '12' or '12;13'). If empty, defaults to '12;13'.")

# NVSHMEM_FORCE_REBUILD_PYTHON_LIB is defined in NVSHMEMEnv.cmake

# Must run before addCybind/addNumbast/generateCuteBindings so they can recreate targets
if(NVSHMEM_FORCE_REBUILD_PYTHON_LIB AND EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/force-rebuild-python-lib.cmake")
    include(cmake/force-rebuild-python-lib.cmake)
endif()

# Include the custom module
include(cmake/buildWheel.cmake)

# Set up shared Python env for bindings only when sources exist.
if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addCybind.cmake" OR
   EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addNumbast.cmake" OR
   EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addNumbastMlir.cmake" OR
   EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/generateCuteBindings.cmake")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Shared Python env setup for binding generation to avoid
    # concurrent installs and venv races.
    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")
    set(BINDINGS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/externals/output")
    add_custom_target(
        setup_py_bindings_env
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/externals
        COMMAND mkdir -p ${BINDINGS_OUTPUT_DIR}
        COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install --upgrade pip
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/nvshmem4py/requirements_build.txt
        COMMAND touch ${BINDINGS_OUTPUT_DIR}/setup_py_bindings_env.txt
        COMMENT "Setting up shared Python env for bindings"
    )
endif()


# Additional cleanup to clean the build/externals dir
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/externals
    ${CMAKE_BINARY_DIR}/CMakeCache.txt
    ${CMAKE_BINARY_DIR}/CMakeFiles
    ${CMAKE_BINARY_DIR}/Makefile
    ${CMAKE_BINARY_DIR}/cmake_install.cmake
)

# addCybind target is only executed in build_source_pkg instance. In subsequent
# steps, we exclude addCybind from source file. The following step skips
# addCybind target for all subsequent steps.
if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addCybind.cmake" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/_internal/nvshmem.pyx")
    include(cmake/addCybind.cmake)
    # This finds Cybind, clones it, and generates the 1:1 bindings for NVSHMEM
    AddCybind(
        GIT_TAG "main"
        KNOWN_DEST "${CMAKE_SOURCE_DIR}/src/nvshmem4py/bindings/"
    )
endif()

# addNumbast target is only executed in build_source_pkg instance. In subsequent
# steps, we exclude addNumbast from source file. The following step skips
# addNumbast target for all subsequent steps.
# Skip bindings if they are already generated
if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addNumbast.cmake" AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB AND (NVSHMEM_FORCE_REBUILD_PYTHON_LIB OR NOT EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba/_numbast.py"))
    include(cmake/addNumbast.cmake)
    AddNumbast(
        VERSION "0.9.0"
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addNumbastMlir.cmake" AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB AND (NVSHMEM_FORCE_REBUILD_PYTHON_LIB OR NOT EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba_cuda_mlir/_numbast.py"))
    include(cmake/addNumbastMlir.cmake)
    AddNumbastMlir(
        VERSION "0.10.1"
    )
endif()

set(CUTEAST_OUTPUT "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/cute/_cuteast.py")
set(CUTE_BINDINGS_ASSET_DIR "${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute")
set(CUTE_BINDINGS_INPUTS
    "${CUTE_BINDINGS_ASSET_DIR}/entry_point.h"
    "${CUTE_BINDINGS_ASSET_DIR}/generate_amo.py"
    "${CUTE_BINDINGS_ASSET_DIR}/generate_collective.py"
    "${CUTE_BINDINGS_ASSET_DIR}/generate_cute_bindings.py"
    "${CUTE_BINDINGS_ASSET_DIR}/generate_cute_config.py"
    "${CUTE_BINDINGS_ASSET_DIR}/generate_rma.py"
    "${CUTE_BINDINGS_ASSET_DIR}/templates/config_nvshmem.yml.j2"
    "${CUTE_BINDINGS_ASSET_DIR}/templates/core/device/cute/amo.py.j2"
    "${CUTE_BINDINGS_ASSET_DIR}/templates/core/device/cute/collective.py.j2"
    "${CUTE_BINDINGS_ASSET_DIR}/templates/core/device/cute/rma.py.j2"
    "${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/numbast/numbast_common.py"
)
set(CUTE_BINDINGS_GENERATOR_AVAILABLE TRUE)
foreach(CUTE_BINDINGS_INPUT IN LISTS CUTE_BINDINGS_INPUTS)
    if(NOT EXISTS "${CUTE_BINDINGS_INPUT}")
        set(CUTE_BINDINGS_GENERATOR_AVAILABLE FALSE)
    endif()
endforeach()

if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/generateCuteBindings.cmake" AND
   CUTE_BINDINGS_GENERATOR_AVAILABLE AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB)
    if(NVSHMEM_FORCE_REBUILD_PYTHON_LIB AND EXISTS "${CUTEAST_OUTPUT}")
        file(REMOVE "${CUTEAST_OUTPUT}")
        message(STATUS "Force rebuild: removed ${CUTEAST_OUTPUT}")
    endif()

    include(cmake/generateCuteBindings.cmake)
    generateCuteBindings()
endif()

# Avoid race conditions by serializing pip installs.
if(TARGET pip_install_cybind AND TARGET pip_install_numbast)
    add_dependencies(pip_install_numbast pip_install_cybind)
endif()

if(TARGET pip_install_numbast_mlir AND TARGET pip_install_numbast)
    add_dependencies(pip_install_numbast_mlir pip_install_numbast)
endif()

if(TARGET pip_install_numbast_mlir AND TARGET pip_install_cybind)
    add_dependencies(pip_install_numbast_mlir pip_install_cybind)
endif()


# Determine which Python versions to build wheels for
set(PYTHON_LINE_LIST "")

if(NVSHMEM4PY_PYTHON_VERSIONS)
    message(STATUS "Using filtered Python versions: ${NVSHMEM4PY_PYTHON_VERSIONS}")

    foreach(PY_VER IN LISTS NVSHMEM4PY_PYTHON_VERSIONS)
        string(REPLACE "." ";" VER_PARTS "${PY_VER}")
        list(LENGTH VER_PARTS VER_PARTS_LEN)

        if(VER_PARTS_LEN EQUAL 2)
            list(GET VER_PARTS 0 PY_MAJOR)
            list(GET VER_PARTS 1 PY_MINOR)
        else()
            message(FATAL_ERROR "Invalid Python version format: ${PY_VER}. Expected format: 'major.minor' (e.g., '3.12')")
        endif()

        set(OVERRIDE_VAR "NVSHMEM4PY_PYTHON_EXECUTABLE_${PY_MAJOR}_${PY_MINOR}")
        if(DEFINED ${OVERRIDE_VAR})
            set(PY_EXEC "${${OVERRIDE_VAR}}")
            message(STATUS "Using override Python executable for ${PY_VER}: ${PY_EXEC}")
        else()
            find_program(PY_EXEC_${PY_VER}
                NAMES python${PY_VER} python${PY_MAJOR}.${PY_MINOR}
                DOC "Python ${PY_VER} executable"
            )

            if(NOT PY_EXEC_${PY_VER})
                message(WARNING "Python ${PY_VER} not found. Skipping wheel build for this version.")
                continue()
            endif()

            set(PY_EXEC "${PY_EXEC_${PY_VER}}")
        endif()

        if(EXISTS "${PY_EXEC}")
            execute_process(
                COMMAND "${PY_EXEC}" -c "import sys; print('%d.%d' % (sys.version_info[0], sys.version_info[1]))"
                OUTPUT_VARIABLE DETECTED_VER
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            if(NOT DETECTED_VER STREQUAL PY_VER)
                message(WARNING "Python executable ${PY_EXEC} reports version ${DETECTED_VER}, expected ${PY_VER}. Skipping.")
                continue()
            endif()

            list(APPEND PYTHON_LINE_LIST "${PY_VER}|${PY_EXEC}")
            message(STATUS "Found Python ${PY_VER}: ${PY_EXEC}")
        else()
            message(WARNING "Python executable not found: ${PY_EXEC}")
        endif()
    endforeach()
else()
    message(STATUS "Auto-detecting all available Python versions >= 3.9")
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_python_versions.sh"
        OUTPUT_VARIABLE PYTHON_VERSION_LINES
        ERROR_VARIABLE PYTHON_DISCOVER_ERROR
        RESULT_VARIABLE PYTHON_DISCOVER_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT PYTHON_DISCOVER_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to auto-detect Python versions: ${PYTHON_DISCOVER_ERROR}")
    endif()

    # Each line is now "3.10|/usr/bin/python3.10"
    string(REPLACE "\n" ";" PYTHON_LINE_LIST "${PYTHON_VERSION_LINES}")
endif()

# Used to create the master target
set(ALL_WHEEL_TARGETS "")
set(PREV_WHEEL_TARGET "")

if(NVSHMEM4PY_CUDA_VERSIONS)
    set(CUDA_VERSIONS ${NVSHMEM4PY_CUDA_VERSIONS})
    message(STATUS "Using filtered CUDA versions: ${CUDA_VERSIONS}")
else()
    set(CUDA_VERSIONS "12" "13")
    message(STATUS "Using default CUDA versions: ${CUDA_VERSIONS}")
endif()

foreach(LINE IN LISTS PYTHON_LINE_LIST)
    string(REPLACE "|" ";" PYTHON_PAIR "${LINE}")
    list(GET PYTHON_PAIR 0 PY_VER)
    list(GET PYTHON_PAIR 1 PY_EXEC)
    message(STATUS "Found Python3 ${PY_EXEC}")

    foreach(CUDA_VER IN LISTS CUDA_VERSIONS)
        set(WHEEL_TARGET build_nvshmem4py_wheel_cu${CUDA_VER}_${PY_VER})

        BuildWheel(${WHEEL_TARGET} ${PY_VER} ${PY_EXEC} ${CUDA_VER})

        if(PREV_WHEEL_TARGET)
            add_dependencies(${WHEEL_TARGET} ${PREV_WHEEL_TARGET})
        endif()

        # All wheels depend on bindings
        if(TARGET build_bindings_cybind)
            add_dependencies(${WHEEL_TARGET} build_bindings_cybind)
        endif()

        if(TARGET build_bindings_numbast AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB)
            add_dependencies(${WHEEL_TARGET} build_bindings_numbast)
        endif()

        if(TARGET build_bindings_numbast_mlir AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB)
            add_dependencies(${WHEEL_TARGET} build_bindings_numbast_mlir)
        endif()

        if(TARGET build_bindings_cute AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB)
            add_dependencies(${WHEEL_TARGET} build_bindings_cute)
        endif()

        set(PREV_WHEEL_TARGET ${WHEEL_TARGET})
        list(APPEND ALL_WHEEL_TARGETS ${WHEEL_TARGET})
    endforeach()
endforeach()

if(NVSHMEM4PY_BUILD_ALL_WHEELS)
    add_custom_target(build_nvshmem4py_wheels ALL
        COMMENT "Build all wheels for all Python versions"
        DEPENDS ${ALL_WHEEL_TARGETS}
    )
    message(STATUS "nvshmem4py wheels will be built automatically (NVSHMEM4PY_BUILD_ALL_WHEELS=ON)")
else()
    add_custom_target(build_nvshmem4py_wheels
        COMMENT "Build all wheels for all Python versions"
        DEPENDS ${ALL_WHEEL_TARGETS}
    )
    message(STATUS "nvshmem4py wheels will NOT be built automatically. Use 'ninja build_nvshmem4py_wheels' or individual targets to build them. (NVSHMEM4PY_BUILD_ALL_WHEELS=OFF)")
endif()

# Install step to add to tarball
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/dist"
    DESTINATION "lib/python/"
)
