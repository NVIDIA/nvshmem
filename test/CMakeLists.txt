cmake_minimum_required(VERSION 3.20)

# Set CMP0177 for path normalization in install() (CMake 3.31+)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(NVSHMEM_TEST_TLD ${CMAKE_CURRENT_SOURCE_DIR})

if (DEFINED ENV{NVSHMEM_TEST_INSTALL})
  set(NVSHMEM_TEST_INSTALL_PREFIX $ENV{NVSHMEM_TEST_INSTALL})
else()
  set(NVSHMEM_TEST_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/test_install")
endif()

get_directory_property(SubBuild PARENT_DIRECTORY)

if(NOT SubBuild)
  if (DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME_DEFAULT $ENV{CUDA_HOME})
  else()
    set(CUDA_HOME_DEFAULT "/usr/local/cuda")
  endif()
  set(CUDA_HOME ${CUDA_HOME_DEFAULT} CACHE PATH "path to CUDA installation")

  # Allow users to set the CUDA toolkit through the env.
  if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA_HOME: ${CUDA_HOME}")
    set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
    set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
  endif()

  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES_UNDEFINED 1)
  endif()

  if (NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES_UNDEFINED 1)
  endif()

  project(NVSHMEMTest VERSION 1.0.0
          LANGUAGES CUDA CXX)

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_config")
  include(NVSHMEMCxxStandard)

  if (DEFINED ENV{NVSHMEM_PREFIX})
    set(NVSHMEM_PREFIX_DEFAULT $ENV{NVSHMEM_PREFIX})
  else()
    set(NVSHMEM_PREFIX_DEFAULT "/usr/local/nvshmem")
  endif()

  if (DEFINED ENV{NVSHMEM_MPI_SUPPORT})
    set(NVSHMEM_MPI_SUPPORT_DEFAULT $ENV{NVSHMEM_MPI_SUPPORT})
  else()
    set(NVSHMEM_MPI_SUPPORT_DEFAULT ON)
  endif()

  if (DEFINED ENV{MPI_HOME})
    set(MPI_HOME_DEFAULT $ENV{MPI_HOME})
  else()
    set(MPI_HOME_DEFAULT "/usr/local/ompi")
  endif()

  option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library and tests" $ENV{NVSHMEM_BUILD_BITCODE_LIBRARY})
  set(NVSHMEM_CLANG_DIR $ENV{NVSHMEM_CLANG_DIR} CACHE PATH "path to force cmake to look for clang when compiling the bitcode library.")
  set(NVSHMEM_CLANG_CXX_FLAGS_EXTRA "$ENV{NVSHMEM_CLANG_CXX_FLAGS_EXTRA}" CACHE STRING "Extra C++ flags passed to clang for bitcode/cubin compilation")
  option(NVSHMEM_DEBUG "Toggles NVSHMEM debug compilation settings" $ENV{NVSHMEM_DEBUG})
  option(NVSHMEM_DEVEL "Toggles NVSHMEM devel compilation settings" $ENV{NVSHMEM_DEVEL})
  option(NVSHMEM_MPI_SUPPORT "Enable compilation of the MPI bootstrap and MPI-specific code" ${NVSHMEM_MPI_SUPPORT_DEFAULT})
  option(NVSHMEM_SHMEM_SUPPORT "Enable Compilation of the SHMEM bootstrap and SHMEM specific code" $ENV{NVSHMEM_SHMEM_SUPPORT})
  option(NVSHMEM_VERBOSE "Enable the ptxas verbose compilation option" $ENV{NVSHMEM_VERBOSE})
  set(MPI_HOME ${MPI_HOME_DEFAULT} CACHE PATH "path to MPI installation")
  set(NVSHMEM_PREFIX ${NVSHMEM_PREFIX_DEFAULT} CACHE PATH "path to NVSHMEM install directory.")
  set(SHMEM_HOME ${MPI_HOME} CACHE PATH "path to SHMEM installation")

  find_package(CUDAToolkit)
  if(CUDAToolkit_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "NVSHMEM tests require CUDA 12.0 or later.")
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake_config/NVSHMEMCUDATools.cmake)

  if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
      set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCHITECTURES" FORCE)
    else()
      if(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 8)
        set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      else()
        set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      endif()
    endif()
  endif()

  if(NVSHMEM_BUILD_BITCODE_LIBRARY)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake_config/NVSHMEMClangSetup.cmake)
    if(NOT NVSHMEM_LLVM_LINK_AVAILABLE)
      message(WARNING "llvm-link not found — bitcode cubin test targets will be skipped")
    endif()
  endif()

  include(CheckCompilerFlag)
  check_compiler_flag(CUDA -t4 NVCC_THREADS)

  find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
  add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
  add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

  if(NVSHMEM_MPI_SUPPORT)
    find_package(MPI REQUIRED)
  endif()

  if(NVSHMEM_SHMEM_SUPPORT)
    find_library(
      SHMEM_LIB REQUIRED
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
endif()

set(TEST_CXX_ARG "c++${CMAKE_CXX_STANDARD}")

foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  string(REGEX REPLACE "-(real|virtual)" "" CLEANED_ARCH "${ARCH}")
  set(NVCC_GENCODE "${NVCC_GENCODE}-gencode=arch=compute_${CLEANED_ARCH},code=sm_${CLEANED_ARCH};-gencode=arch=compute_${CLEANED_ARCH},code=compute_${CLEANED_ARCH};")
endforeach()
mark_as_advanced(NVCC_GENCODE CMAKE_CUDA_ARCHITECTURES_UNDEFINED CUDA_ARCHITECTURES_UNDEFINED)

if(CUDAToolkit_VERSION_MAJOR EQUAL 12)
  set(NVSHMEM_CLANG_ARCH "sm_90")
  set(NVSHMEM_PTX_ARCH "ptx82")
else()
  set(NVSHMEM_CLANG_ARCH "sm_90")
  set(NVSHMEM_PTX_ARCH "ptx86")
endif()

if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13 AND CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 NVSHMEM_CLANG_ARCH)
  string(REGEX REPLACE "-(real|virtual)$" "" NVSHMEM_CLANG_ARCH "${NVSHMEM_CLANG_ARCH}")
  set(NVSHMEM_CLANG_ARCH "sm_${NVSHMEM_CLANG_ARCH}")
endif()

if (NOT SubBuild)
  configure_file(${NVSHMEM_TEST_TLD}/../src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in
  ${NVSHMEM_TEST_TLD}/common/include/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
endif()

add_subdirectory(common)

if(NOT DEFINED NVSHMEM_BUILD_BITCODE_CUBIN_TESTS)
  set(NVSHMEM_BUILD_BITCODE_CUBIN_TESTS ON)
endif()

macro(nvshmem_add_test_prefix SOURCE PREFIX LINK_DEVICE)
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_TEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(NAME "${PREFIX}${NAME_}")
  add_executable(${NAME} ${SOURCE})
  set_target_properties(${NAME} PROPERTIES OUTPUT_NAME "${NAME_}" INSTALL_RPATH "$ORIGIN/../../../../lib" BUILD_WITH_INSTALL_RPATH TRUE)
  if (${LINK_DEVICE})
    target_link_libraries(${NAME} nvshmem_test_helper)
  else()
    target_link_libraries(${NAME} nvshmem_test_helper_no_device)
  endif()

  if (NVSHMEM_USE_NCCL)
    target_include_directories(${NAME} PRIVATE ${NCCL_INCLUDE})
  endif()

  set_target_properties(${NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ${LINK_DEVICE}
  )

  target_compile_options(${NAME}
  PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
  $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  $<$<COMPILE_LANGUAGE:CUDA>:-Werror
  all-warnings>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror
  -Wall
  -Wextra
  -Wno-unused-variable
  -Wno-unused-function
  -Wno-unused-parameter
  -Wno-missing-field-initializers>
  )

  install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/${DIR}")
  if (NVSHMEM_TEST_RELEASE_PREFIX)
    install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/${DIR}")
  endif()
endmacro()


macro(nvshmem_add_ltoir_cubin_test_prefix SOURCE PREFIX)
  if(NOT TARGET cuda::nvcc)
    message(WARNING "cuda::nvcc target was not found. Skipping LTOIR cubin test.")
  else()
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_TEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(CUBIN_NAME "${NAME_}_ltoir.cubin")
  set(INCLUDE_DIRECTORIES "-I${NVSHMEM_TEST_TLD}/common" "-I${NVSHMEM_TEST_TLD}/../src/include/" "-I${NVSHMEM_PREFIX}/include")
  if (NOT SubBuild)
    set(LTOIR_LIB_PATH ${NVSHMEM_PREFIX}/lib/libnvshmem_device_${NVSHMEM_CLANG_ARCH}.ltoir)
    # Note: The -DNVSHMEM_HOSTLIB_ONLY macro name is a misnomer: it's used in the testing framework so the unmangled symbols will appear in the LTOIR cubin (just adds "extern c" to all the tests).
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
                      COMMAND cuda::nvcc -std=${TEST_CXX_ARG} -arch=${NVSHMEM_CLANG_ARCH} -dlto --ltoir
                              ${INCLUDE_DIRECTORIES} -DNVSHMEM_HOSTLIB_ONLY ${DIR_} -o ${NAME_}.ltoir
                      COMMAND cuda::nvcc -arch=${NVSHMEM_CLANG_ARCH} -dlto -dlink -o ${CUBIN_NAME} ${NAME_}.ltoir ${LTOIR_LIB_PATH}
                      COMMAND rm ${NAME_}.ltoir
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      DEPENDS nvshmem_test_helper ${LTOIR_LIB_PATH})
  else()
    set(LTOIR_LIB_PATH ${CMAKE_BINARY_DIR}/src/lib/libnvshmem_device_${NVSHMEM_CLANG_ARCH}.ltoir)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
                      COMMAND cuda::nvcc -std=${TEST_CXX_ARG} -arch=${NVSHMEM_CLANG_ARCH} -dlto --ltoir
                              ${INCLUDE_DIRECTORIES} -DNVSHMEM_HOSTLIB_ONLY ${DIR_} -o ${NAME_}.ltoir
                      COMMAND cuda::nvcc -arch=${NVSHMEM_CLANG_ARCH} -dlto -dlink -o ${CUBIN_NAME} ${NAME_}.ltoir ${LTOIR_LIB_PATH}
                      COMMAND rm ${NAME_}.ltoir
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      DEPENDS nvshmem_test_helper libnvshmem_device_ltoir)
  endif()

  add_custom_target(${NAME_}_ltoir_cubin ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME})

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
    DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/${DIR}"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  if (NVSHMEM_TEST_RELEASE_PREFIX)
    install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
            DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/${DIR}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif()
  endif()
endmacro()

function(nvshmem_add_cubin_test_prefix SOURCE PREFIX)
  if(NOT NVSHMEM_BUILD_BITCODE_CUBIN_TESTS)
    return()
  endif()

  if(NOT (NVSHMEM_LLVM_LINK_AVAILABLE AND NVSHMEM_OPT_AVAILABLE AND NVSHMEM_LLC_AVAILABLE
      AND TARGET cuda::ptxas))
    return()
  endif()

  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_TEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(CUBIN_NAME "${NAME_}_bc.cubin")
  set(INCLUDE_DIRECTORIES "-I${NVSHMEM_TEST_TLD}/common" "-I${NVSHMEM_TEST_TLD}/../src/include/" "-I${NVSHMEM_PREFIX}/include"  "-I${CUDA_HOME}/include/cccl")

  # Add debug flags for Debug and RelWithDebInfo builds
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _CUBIN_BUILD_TYPE_UPPER)
  if(_CUBIN_BUILD_TYPE_UPPER STREQUAL "DEBUG" OR _CUBIN_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
    set(CUBIN_DEBUG_FLAGS "-g")
  else()
    set(CUBIN_DEBUG_FLAGS "")
  endif()

  if (NOT SubBuild)
    set(BITCODE_LIB_PATH ${NVSHMEM_PREFIX}/lib/libnvshmem_device_${NVSHMEM_CLANG_ARCH}.bc)
    set(CUBIN_EXTRA_DEPENDS nvshmem_test_helper ${BITCODE_LIB_PATH})
  else()
    set(BITCODE_LIB_PATH ${CMAKE_BINARY_DIR}/src/lib/libnvshmem_device_${NVSHMEM_CLANG_ARCH}.bc)
    set(CUBIN_EXTRA_DEPENDS nvshmem_test_helper libnvshmem_device_bitcode)
  endif()

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
	            COMMAND ${NVSHMEM_CLANG_EXECUTABLE} ${NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST} -c -emit-llvm ${CUBIN_DEBUG_FLAGS} -std=${TEST_CXX_ARG} -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only
                            --cuda-gpu-arch=${NVSHMEM_CLANG_ARCH} ${INCLUDE_DIRECTORIES} -DNVSHMEM_HOSTLIB_ONLY ${DIR_} -o ${NAME_}.bc.unoptimized
                    COMMAND llvm::llvm-link --only-needed ${NAME_}.bc.unoptimized ${BITCODE_LIB_PATH} -o ${NAME_}.bc.unoptimized.linked
                    COMMAND llvm::opt --passes='internalize,inline,mem2reg' ${NAME_}.bc.unoptimized.linked -o ${NAME_}.bc
                    COMMAND llvm::llc -mcpu=${NVSHMEM_CLANG_ARCH} -mattr=${NVSHMEM_PTX_ARCH} ${NAME_}.bc -o ${NAME_}.ptx
                    COMMAND cuda::ptxas ${CUBIN_DEBUG_FLAGS} -arch=${NVSHMEM_CLANG_ARCH} ${NAME_}.ptx -o ${CUBIN_NAME}
                    COMMAND rm ${NAME_}.bc.unoptimized ${NAME_}.bc.unoptimized.linked ${NAME_}.bc ${NAME_}.ptx
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    DEPENDS ${CUBIN_EXTRA_DEPENDS})

  add_custom_target(${NAME_}_cubin ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME})

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
    DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/${DIR}"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  if (NVSHMEM_TEST_RELEASE_PREFIX)
    install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
            DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/${DIR}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif()
endfunction()

macro(nvshmem_add_unit_test_prefix SOURCE PREFIX)
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_TEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(NAME "${PREFIX}${NAME_}")
  add_executable(${NAME} ${SOURCE})
  set_target_properties(${NAME} PROPERTIES OUTPUT_NAME "${NAME_}" INSTALL_RPATH "$ORIGIN/../../../../lib" BUILD_WITH_INSTALL_RPATH TRUE)
  # unit-test should never run on device
  target_link_libraries(${NAME} nvshmem_unit_test_helper)

  set_target_properties(${NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(${NAME}
  PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
  $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  $<$<COMPILE_LANGUAGE:CUDA>:-Werror
  all-warnings>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror
  -Wall
  -Wextra
  -Wno-unused-variable
  -Wno-unused-function
  -Wno-unused-parameter
  -Wno-missing-field-initializers>
  )

  install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/${DIR}")
  if (NVSHMEM_TEST_RELEASE_PREFIX)
    install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/${DIR}")
  endif()
endmacro()

option(INSTALL_GTEST OFF)
find_package(GTest)

if(GTest_FOUND)
include(FetchContent)

FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG "release-1.12.1"
)

FetchContent_MakeAvailable(googletest)


macro(nvshmem_add_gtest_unit_test_prefix SOURCE PREFIX FILE)
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_TEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR_ ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(NAME "${PREFIX}${NAME_}")
  if("${FILE}" STREQUAL "")
    add_executable(${NAME} ${SOURCE})
  else()
    add_executable(${NAME} ${SOURCE} ${FILE})
  endif()
  target_include_directories(${NAME} PRIVATE ${DIR_})
  target_link_libraries(${NAME} nvshmem_gtest_unit_test_helper)

  set_target_properties(${NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(${NAME} BEFORE
  PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
  $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  $<$<COMPILE_LANGUAGE:CUDA>:-Werror
  all-warnings>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror
  -Wall
  -Wextra
  -Wno-unused-variable
  -Wno-unused-function
  -Wno-unused-parameter
  -fprofile-arcs -ftest-coverage
  -Wno-missing-field-initializers
  >
  )
  target_compile_options(${NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-vla-cxx-extension>)

  install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/${DIR}")
endmacro()
endif()

set(FATBIN_DEFS_LIST)
if (NVSHMEM_ENABLE_CFT_HANDLES)
  list(APPEND FATBIN_DEFS_LIST -DCFT_HANDLES_ENABLED)
  if (NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT)
    list(APPEND FATBIN_DEFS_LIST -DPRIORITIZE_LOGICAL_ENDPOINT)
  endif()
endif()

set(KERNEL_NVSHMEM_EXTRA_INCLUDES)
if(SubBuild AND NVSHMEM_GPUNETIO_SUPPORT)
  if(NOT DEFINED GPUNETIO_INCLUDE)
    message(FATAL_ERROR "GPUNETIO_INCLUDE is not defined when NVSHMEM_GPUNETIO_SUPPORT is ON")
  endif()
  list(APPEND KERNEL_NVSHMEM_EXTRA_INCLUDES "-I${GPUNETIO_INCLUDE}")
endif()

if(NOT SubBuild)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin
                   COMMAND ${CMAKE_CUDA_COMPILER}
                   -I${NVSHMEM_PREFIX}/include -I${CUDA_HOME}/include ${KERNEL_NVSHMEM_EXTRA_INCLUDES} ${NVCC_GENCODE} ${FATBIN_DEFS_LIST}
                   ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem.cu
                   -c --fatbin -Xcompiler -fPIC -rdc=true --std=${TEST_CXX_ARG} -o ${CMAKE_CURRENT_BINARY_DIR}/host/init/tmp.fatbin
                   COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_GENCODE} ${CMAKE_CURRENT_BINARY_DIR}/host/init/tmp.fatbin -c --fatbin -dlink -o ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin -L${NVSHMEM_LIB_DIR} -lnvshmem_device
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem.cu
                   DEPENDS nvshmem_test_helper
                   COMMAND_EXPAND_LISTS)
else()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin
                   COMMAND ${CMAKE_CUDA_COMPILER}
                   -I${CMAKE_SOURCE_DIR}/src/include -I${CUDA_HOME}/include ${KERNEL_NVSHMEM_EXTRA_INCLUDES} ${NVCC_GENCODE} ${FATBIN_DEFS_LIST}
                   ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem.cu
                   -c --fatbin -Xcompiler -fPIC -rdc=true --std=${TEST_CXX_ARG} -o ${CMAKE_CURRENT_BINARY_DIR}/host/init/tmp.fatbin
                   COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_GENCODE} ${CMAKE_CURRENT_BINARY_DIR}/host/init/tmp.fatbin -c --fatbin -dlink -o ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin -L${CMAKE_BINARY_DIR}/src/lib -lnvshmem_device
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem.cu
                   DEPENDS nvshmem_test_helper
                   COMMAND_EXPAND_LISTS)
endif()
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin
        DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/init")
if (NVSHMEM_TEST_RELEASE_PREFIX)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/host/init")
endif()
add_custom_target(kernel_nvshmem_dummy ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem.fatbin)

# Non-RDC fatbin: compiled without -rdc=true, with all device functions inlined.
# Single-stage compilation — no device-link step needed.
if(NOT SubBuild)
  set(KERNEL_NVSHMEM_NO_RDC_INCLUDE_DIR "${NVSHMEM_PREFIX}/include")
else()
  set(KERNEL_NVSHMEM_NO_RDC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/include")
endif()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem_no_rdc.fatbin
                   COMMAND ${CMAKE_CUDA_COMPILER}
                   -I${KERNEL_NVSHMEM_NO_RDC_INCLUDE_DIR} -I${CUDA_HOME}/include ${KERNEL_NVSHMEM_EXTRA_INCLUDES} ${NVCC_GENCODE}
                   -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING
                   ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem_no_rdc.cu
                   -c --fatbin -Xcompiler -fPIC --std=${TEST_CXX_ARG}
                   -o ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem_no_rdc.fatbin
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/host/init/kernel_nvshmem_no_rdc.cu
                   COMMAND_EXPAND_LISTS)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem_no_rdc.fatbin
        DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/init")
if (NVSHMEM_TEST_RELEASE_PREFIX)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem_no_rdc.fatbin DESTINATION "${NVSHMEM_TEST_RELEASE_PREFIX}/host/init")
endif()
add_custom_target(kernel_nvshmem_no_rdc_dummy ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/host/init/kernel_nvshmem_no_rdc.fatbin)

add_subdirectory(apps)
add_subdirectory(device)
add_subdirectory(host)
