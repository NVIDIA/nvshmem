cmake_minimum_required(VERSION 3.15)

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
        VERSION "0.6.0"
    )
endif()

# Generate CuTe Bindings. Follows the same requirements as the Numba bindings
# Separate build step because users may want one but not the other
if (EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/generateCuteBindings.cmake" AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB AND (NVSHMEM_FORCE_REBUILD_PYTHON_LIB OR NOT EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/cute/_cuteast.py"))
    include(cmake/generateCuteBindings.cmake)
    generateCuteBindings()
endif()

# Avoid race conditions by serializing pip installs.
if(TARGET pip_install_cybind AND TARGET pip_install_numbast)
    add_dependencies(pip_install_numbast pip_install_cybind)
endif()


# Find all the Python versions
# Use bash to find all Python versions >= 3.9
execute_process(
    COMMAND bash -c "${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_python_versions.sh"
    OUTPUT_VARIABLE PYTHON_VERSION_LINES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Each line is now "3.10|/usr/bin/python3.10"
# Convert to list
string(REPLACE "\n" ";" PYTHON_LINE_LIST "${PYTHON_VERSION_LINES}")

# Used to create the master target
set(ALL_WHEEL_TARGETS "")
set(PREV_WHEEL_TARGET "")

foreach(LINE IN LISTS PYTHON_LINE_LIST)
    string(REPLACE "|" ";" PYTHON_PAIR "${LINE}")
    list(GET PYTHON_PAIR 0 PY_VER)
    list(GET PYTHON_PAIR 1 PY_EXEC)
    message(STATUS "Found Python3 ${PY_EXEC}")
    set(CUDA_VERSIONS "12" "13")

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

        if(TARGET build_bindings_cute AND TARGET build_bindings_numbast AND NVSHMEM_BUILD_PYTHON_DEVICE_LIB)
            add_dependencies(${WHEEL_TARGET} build_bindings_cute)
        endif()

        set(PREV_WHEEL_TARGET ${WHEEL_TARGET})
        list(APPEND ALL_WHEEL_TARGETS ${WHEEL_TARGET})
    endforeach()
endforeach()

add_custom_target(build_nvshmem4py_wheels ALL
    COMMENT "Build all wheels for all Python versions"
    DEPENDS ${ALL_WHEEL_TARGETS}
)

# Install step to add to tarball
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/dist"
    DESTINATION "lib/python/"
)
