# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(nvshmem_ensure_numbast)
    cmake_parse_arguments(PARSE_ARGV 0 ENSURE_NUMBAST "" "VERSION;CLEAN_TARGET" "")

    if(NOT ENSURE_NUMBAST_VERSION)
        message(FATAL_ERROR "VERSION is required for nvshmem_ensure_numbast")
    endif()

    if(ENSURE_NUMBAST_CLEAN_TARGET AND NOT TARGET "${ENSURE_NUMBAST_CLEAN_TARGET}")
        message(FATAL_ERROR "Unknown Numbast cleanup target: ${ENSURE_NUMBAST_CLEAN_TARGET}")
    endif()

    if(TARGET pip_install_numbast)
        get_property(_numbast_version GLOBAL PROPERTY NVSHMEM_NUMBAST_VERSION)
        if(_numbast_version AND NOT "${_numbast_version}" STREQUAL "${ENSURE_NUMBAST_VERSION}")
            message(FATAL_ERROR
                "Numbast ${_numbast_version} is already requested; cannot also request ${ENSURE_NUMBAST_VERSION}")
        endif()
    else()
        set_property(GLOBAL PROPERTY NVSHMEM_NUMBAST_VERSION "${ENSURE_NUMBAST_VERSION}")

        add_custom_target(
            pip_install_numbast
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/externals
            COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install numbast==${ENSURE_NUMBAST_VERSION}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/externals
            USES_TERMINAL
            DEPENDS setup_py_bindings_env
            COMMENT "Installing Numbast ${ENSURE_NUMBAST_VERSION}"
        )
    endif()

    if(ENSURE_NUMBAST_CLEAN_TARGET)
        add_dependencies(pip_install_numbast "${ENSURE_NUMBAST_CLEAN_TARGET}")
    endif()
endfunction()
