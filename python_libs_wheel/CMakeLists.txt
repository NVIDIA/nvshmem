cmake_minimum_required(VERSION 3.19)

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
