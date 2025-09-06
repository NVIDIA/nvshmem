cmake_minimum_required(VERSION 3.15)


# Include the custom module
include(cmake/buildWheel.cmake)





# Additional cleanup to clean the build/externals dir
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/externals
    ${CMAKE_BINARY_DIR}/CMakeCache.txt
    ${CMAKE_BINARY_DIR}/CMakeFiles
    ${CMAKE_BINARY_DIR}/Makefile
    ${CMAKE_BINARY_DIR}/cmake_install.cmake
)

if(EXISTS "${CMAKE_SOURCE_DIR}/nvshmem4py/cmake/addCybind.cmake")
    include(cmake/addCybind.cmake)
    # This finds Cybind, clones it, and generates the 1:1 bindings for NVSHMEM
    AddCybind(
        GIT_TAG "main"
        KNOWN_DEST "${CMAKE_SOURCE_DIR}/src/nvshmem4py/bindings/"
    )
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
    set(CUDA_VERSIONS "11" "12" "13")

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
    DIRECTORY "${CMAKE_SOURCE_DIR}/build/dist"
    DESTINATION lib/python/
)
