cmake_minimum_required(VERSION 3.20)

project(
  NVSHMEM
  LANGUAGES CXX C
  VERSION 3.8.0.0
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED On)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_config")
include(NVSHMEMVersioning)
include(NVSHMEMCxxStandard)
include(NVSHMEMFilePrefixMap)

include(NVSHMEMEnv)
include(CMakePackageConfigHelpers)
include(CheckSourceCompiles)
include(ExternalProject)

enable_testing()

if (NVSHMEM_DEBUG)
	message(STATUS "Setting build type to Debug as requested by the environment.")
  set(CMAKE_BUILD_TYPE "debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set(default_build_type "release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set the possible values of build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release"
             "MinSizeRel" "RelWithDebInfo")

## CUDA Selection Start

# Allow users to set the CUDA toolkit through the env. CUDA_HOME is a cache path
# seeded by NVSHMEMEnv from $ENV{CUDA_HOME} or defaulted to /usr/local/cuda.
if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA_HOME: ${CUDA_HOME}")
  set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
endif()

# Save value of CMAKE_CUDA_ARCHITECTURES before enabling the CUDA language.
# enable_language(CUDA) plants nvcc's compiler-default arch into the variable —
# indistinguishable from a user-supplied value after the fact. We latch user intent
# here so src/ can decide whether to propagate to the device sub-project, and the
# fallback below can pick a version-appropriate default when nobody specified one.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

if (NOT DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

# All CUDA prerequisites are now in place: enable the CUDA language.
enable_language(CUDA)

if(NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang") OR
   NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  message(FATAL_ERROR "NVSHMEM only supports compilation with GNU (gcc) or Clang compilers.")
endif()

# Runtime diagnostics use __FILE__, so do not embed absolute source or build paths in artifacts.
nvshmem_apply_file_prefix_maps()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${NVSHMEM_PREFIX}")
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "NVSHMEM requires CUDA 12.0 or later.")
endif()
include(NVSHMEMCUDATools)
include(NVSHMEMCUDAArchitectures)
find_package(CCCL PATHS ${CUDAToolkit_LIBRARY_DIR}/cmake/cccl)

nvshmem_set_default_cuda_architectures()

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

## CUDA Selection End

## Find Packages Start

if(NVSHMEM_UCX_SUPPORT)
  find_package(UCX PATHS ${UCX_HOME} REQUIRED)
endif()

if(NVSHMEM_BUILD_BITCODE_LIBRARY)
  include(NVSHMEMClangSetup)
endif()

if(NVSHMEM_MPI_SUPPORT)
  find_package(MPI REQUIRED)
endif()

# Find the internal nccl.h file first.
if(NVSHMEM_USE_NCCL)
  find_path(
    NCCL_INCLUDE nccl.h
    HINTS ${NCCL_HOME} ${CMAKE_SOURCE_DIR}
    PATH_SUFFIXES include_nccl include lib64
  )
endif()

if(NVSHMEM_USE_GDRCOPY)
  find_path(
    GDRCOPY_INCLUDE gdrapi.h
    PATHS /usr/local/gdrcopy /usr/local/gdrdrv ${CMAKE_SOURCE_DIR}
    HINTS ${CMAKE_SOURCE_DIR} /usr/local/gdrcopy /usr/local/gdrdrv ${GDRCOPY_HOME}
    PATH_SUFFIXES include_gdrcopy include
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  find_library(
    SHMEM_LIB
    NAMES oshmem
    HINTS ${SHMEM_HOME}
    PATH_SUFFIXES lib lib64)
  find_path(SHMEM_INCLUDE NAME shmem.h HINTS ${SHMEM_HOME}
            PATH_SUFFIXES include
  )
  add_library(shmem IMPORTED INTERFACE)
  target_link_libraries(shmem INTERFACE ${SHMEM_LIB})
  target_include_directories(shmem INTERFACE ${SHMEM_INCLUDE})
  if(NVSHMEM_MPI_SUPPORT)
    separate_arguments(SHMEM_C_LINK_FLAGS NATIVE_COMMAND "${MPI_C_LINK_FLAGS}")
    target_link_options(shmem INTERFACE ${SHMEM_C_LINK_FLAGS})
    target_compile_definitions(shmem INTERFACE ${MPI_C_COMPILE_DEFINITIONS})
    target_compile_options(shmem INTERFACE ${MPI_C_COMPILE_OPTIONS})
  endif()
endif()

if(NVSHMEM_GPUNETIO_SUPPORT)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/gpunetio)
  if(NOT DOCA_HOME)
    set(DOCA_HOME ${CMAKE_CURRENT_SOURCE_DIR}/externals/gpunetio)
  endif()
  # Headers are at include/gpunetio/*.h, so -I include/ allows "gpunetio/..." includes.
  set(GPUNETIO_INCLUDE ${DOCA_HOME}/include)
endif()

## Find Packages End

# set these flags to add a RUNPATH entry to the libraries that points at the lib dir in a portable way.
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$\{ORIGIN\}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")

cmake_policy(SET CMP0105 NEW)
set(CMAKE_CUDA_DEVICE_COMPILER_WRAPPER_FLAG "")

# Set CMP0177 for path normalization in install() (CMake 3.31+)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

configure_file(License.txt License.txt COPYONLY)

set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
set(BIN_INSTALL_DIR bin/)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake_config/NVSHMEMConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/NVSHMEMConfig.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/nvshmem
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(
  NVSHMEMVersion.cmake
  VERSION ${NVSHMEM_BASE_VERSION}
  COMPATIBILITY AnyNewerVersion)

add_subdirectory(src)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# If Python3 version is lower than 3.9, we can't build nvshmem4py
# Avoid breaking other stuff by turning off the python lib if this is the case
if(Python3_VERSION VERSION_LESS "3.9")
    message("Disabling Python library build due to insufficient version")
    set(NVSHMEM_BUILD_PYTHON_LIB OFF)
endif()

if (NVSHMEM_BUILD_PYTHON_LIB)
  add_subdirectory(nvshmem4py)
endif()

if (NVSHMEM_BUILD_LIBS_WHEEL)
  add_subdirectory(python_libs_wheel)
endif()
