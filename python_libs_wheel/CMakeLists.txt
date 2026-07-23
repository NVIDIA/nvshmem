cmake_minimum_required(VERSION 3.20)

# Libs-only Python wheel for NVSHMEM C libraries.
#
# This builds a pure-data Python wheel that packages the NVSHMEM shared and
# static libraries produced by the normal build.  When installed, the libraries
# land under <site-packages>/nvidia/nvshmem/lib/, mirroring the layout of the
# externally-published NVSHMEM libs wheels.
#
# Usage:
#   cmake -DNVSHMEM_BUILD_LIBS_WHEEL=ON ...
#   make build_nvshmem_libs_wheel
#
# The resulting wheel is placed in ${CMAKE_BINARY_DIR}/dist/.

include(cmake/buildLibsWheel.cmake)

# Library artifacts are written to <build>/src/lib/ by the src/ subdirectory.
set(NVSHMEM_BUILT_LIB_DIR "${CMAKE_BINARY_DIR}/src/lib")

BuildLibsWheel(
    build_nvshmem_libs_wheel
    "${NVSHMEM_BUILT_LIB_DIR}"
    "${CUDAToolkit_VERSION_MAJOR}"
)

# Include the libs wheel in the default build so `all`/`install` produce it
# whenever NVSHMEM_BUILD_LIBS_WHEEL is enabled.
add_custom_target(build_nvshmem_libs_wheels ALL
    COMMENT "Build NVSHMEM libs-only wheel"
    DEPENDS build_nvshmem_libs_wheel
)

# Include the libs wheel in the install/package output under lib/python/dist/.
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/dist/"
    DESTINATION "lib/python/dist"
    FILES_MATCHING PATTERN "nvidia_nvshmem_cu*.whl"
)
