cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(NVSHMEM_PERFTEST_TLD ${CMAKE_CURRENT_SOURCE_DIR})

if (DEFINED ENV{NVSHMEM_PERFTEST_INSTALL})
  set(NVSHMEM_PERFTEST_INSTALL_PREFIX $ENV{NVSHMEM_PERFTEST_INSTALL})
else()
  set(NVSHMEM_PERFTEST_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/perftest_install")
endif()

get_directory_property(SubBuild PARENT_DIRECTORY)

if(NOT SubBuild)
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

  if (DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME_DEFAULT $ENV{CUDA_HOME})
  else()
    set(CUDA_HOME_DEFAULT "/usr/local/cuda")
  endif()

  option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library and tests" $ENV{NVSHMEM_BUILD_BITCODE_LIBRARY})
  set(NVSHMEM_CLANG_DIR $ENV{NVSHMEM_CLANG_DIR} CACHE PATH "path to force cmake to look for clang when compiling the bitcode library.")
  option(NVSHMEM_DEBUG "Toggles NVSHMEM debug compilation settings" $ENV{NVSHMEM_DEBUG})
  option(NVSHMEM_DEVEL "Toggles NVSHMEM devel compilation settings" $ENV{NVSHMEM_DEVEL})
  option(NVSHMEM_MPI_SUPPORT "Enable compilation of the MPI bootstrap and MPI-specific code" ${NVSHMEM_MPI_SUPPORT_DEFAULT})
  option(NVSHMEM_SHMEM_SUPPORT "Enable Compilation of the SHMEM bootstrap and SHMEM specific code" $ENV{NVSHMEM_SHMEM_SUPPORT})
  option(NVSHMEM_TEST_STATIC_LIB "Force tests to link only against the combined nvshmem.a binary" $ENV{NVSHMEM_TEST_STATIC_LIB})
  option(NVSHMEM_VERBOSE "Enable the ptxas verbose compilation option" $ENV{NVSHMEM_VERBOSE})
  set(CUDA_HOME ${CUDA_HOME_DEFAULT} CACHE PATH "path to CUDA installation")
  set(MPI_HOME ${MPI_HOME_DEFAULT} CACHE PATH "path to MPI installation")
  set(NVSHMEM_PREFIX ${NVSHMEM_PREFIX_DEFAULT} CACHE PATH "path to NVSHMEM install directory.")
  set(SHMEM_HOME ${MPI_HOME} CACHE PATH "path to SHMEM installation")

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

  PROJECT(NVSHMEMPerftest VERSION 1.0.0
          LANGUAGES CUDA CXX)

  find_package(CUDAToolkit)

  #TODO: consolidate cuda architecture detection code in a single file.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
      if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCHITECTURES" FORCE)
      else()
      if(CUDAToolkit_VERSION_MAJOR LESS 11)
        set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR LESS 8)
        set(CMAKE_CUDA_ARCHITECTURES "70-real;80" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 OR (CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 8))
        set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      else()
        set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
      endif()
    endif()
  endif()

  if(NVSHMEM_BUILD_BITCODE_LIBRARY)
    if(NVSHMEM_CLANG_DIR)
      find_package(Clang CONFIG PATHS ${NVSHMEM_CLANG_DIR} NO_DEFAULT_PATH REQUIRED)
    else()
      find_package(Clang CONFIG REQUIRED)
    endif()
  endif()

  include(CheckCompilerFlag)
  check_compiler_flag(CUDA -t4 NVCC_THREADS)

  find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
  add_library(nvshmem ALIAS nvshmem::nvshmem)
  add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
  add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

  if(NVSHMEM_MPI_SUPPORT)
    find_package(MPI REQUIRED)
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
endif()

if(CUDAToolkit_VERSION_MAJOR LESS 13)
  set(PERFTEST_CXX_STANDARD 11)
else()
  set(PERFTEST_CXX_STANDARD 17)
endif()

# Tile-granular API examples need C++ 17
set(PERFTEST_CXX_STANDARD 17)

add_subdirectory(common)

if(CUDAToolkit_VERSION_MAJOR LESS 12)
  set(NVSHMEM_CLANG_ARCH "sm_70")
  set(NVSHMEM_PTX_ARCH "ptx78")
else()
  set(NVSHMEM_CLANG_ARCH "sm_90")
  set(NVSHMEM_PTX_ARCH "ptx82")
endif()

if(CUDAToolkit_VERSION_MAJOR LESS 13)
  set(BITCODE_CXX_STD "c++11")
else()
  set(BITCODE_CXX_STD "c++17")
endif()

if (NOT SubBuild)
  configure_file(${NVSHMEM_PERFTEST_TLD}/../src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in
  ${NVSHMEM_PERFTEST_TLD}/common/include/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
endif()

macro(nvshmem_add_perftest_prefix SOURCE PREFIX)
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_PERFTEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(NAME "${PREFIX}${NAME_}")
  add_executable(${NAME} ${SOURCE})
  set_target_properties(${NAME} PROPERTIES OUTPUT_NAME "${NAME_}" INSTALL_RPATH "$ORIGIN/../../../../lib" BUILD_WITH_INSTALL_RPATH TRUE)
  set_target_properties(${NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD ${PERFTEST_CXX_STANDARD}
  CUDA_STANDARD ${PERFTEST_CXX_STANDARD}
  CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(${NAME}
  PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
  $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  set_target_properties(${NAME} PROPERTIES OUTPUT_NAME "${NAME_}")
  target_link_libraries(${NAME} nvshmem_perftest_helper)
  install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_PERFTEST_INSTALL_PREFIX}/${DIR}")
  if (NVSHMEM_PERFTEST_RELEASE_PREFIX)
    install(TARGETS ${NAME} RUNTIME DESTINATION "${NVSHMEM_PERFTEST_RELEASE_PREFIX}/${DIR}")
  endif()

endmacro()

macro(nvshmem_add_cubin_perftest_prefix SOURCE PREFIX)
  get_filename_component(NAME_ ${SOURCE} NAME_WE)
  get_filename_component(DIR_ ${SOURCE} ABSOLUTE)
  string(REPLACE ${NVSHMEM_PERFTEST_TLD} "" DIR ${DIR_})
  string(REPLACE ${SOURCE} "" DIR ${DIR})
  set(CUBIN_UNLINKED_NAME "${NAME_}_UNLINKED.cubin")
  set(CUBIN_NAME "${NAME_}.cubin")
  set(INCLUDE_DIRECTORIES "-I${NVSHMEM_PERFTEST_TLD}/common" "-I${NVSHMEM_PERFTEST_TLD}/../src/include/" "-I${NVSHMEM_PREFIX}/include" "-I${CUDA_HOME}/include/cccl")
  if (NOT SubBuild)
    set(BITCODE_LIB_PATH ${NVSHMEM_PREFIX}/lib/libnvshmem_device.bc)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
                      COMMAND clang -c -emit-llvm -std=${BITCODE_CXX_STD} -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only
                              --cuda-gpu-arch=${NVSHMEM_CLANG_ARCH} ${INCLUDE_DIRECTORIES} -DNVSHMEM_HOSTLIB_ONLY ${DIR_} -o ${NAME_}.bc.unoptimized
                      COMMAND llvm-link --only-needed ${NAME_}.bc.unoptimized ${BITCODE_LIB_PATH} -o ${NAME_}.bc.unoptimized.linked
                      COMMAND opt -O3 ${NAME_}.bc.unoptimized.linked -o ${NAME_}.bc
                      COMMAND llc -O3 --disable-tail-calls -mcpu=${NVSHMEM_CLANG_ARCH} -mattr=${NVSHMEM_PTX_ARCH} ${NAME_}.bc -o ${NAME_}.ptx
                      COMMAND ${CUDA_HOME}/bin/ptxas -c -arch=${NVSHMEM_CLANG_ARCH} ${NAME_}.ptx -o ${CUBIN_UNLINKED_NAME}
                      COMMAND ${CUDA_HOME}/bin/nvlink -arch=${NVSHMEM_CLANG_ARCH} ${CUBIN_UNLINKED_NAME} -o ${CUBIN_NAME}
                      COMMAND rm ${NAME_}.bc.unoptimized ${NAME_}.bc.unoptimized.linked  ${NAME_}.bc ${NAME_}.ptx ${CUBIN_UNLINKED_NAME}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      DEPENDS nvshmem_perftest_helper ${BITCODE_LIB_PATH})
  else()
    set(BITCODE_LIB_PATH ${CMAKE_BINARY_DIR}/src/lib/libnvshmem_device.bc)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
                      COMMAND clang -c -emit-llvm -std=${BITCODE_CXX_STD} -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only
                              --cuda-gpu-arch=${NVSHMEM_CLANG_ARCH} ${INCLUDE_DIRECTORIES} -DNVSHMEM_HOSTLIB_ONLY ${DIR_} -o ${NAME_}.bc.unoptimized
                      COMMAND llvm-link --only-needed ${NAME_}.bc.unoptimized ${BITCODE_LIB_PATH} -o ${NAME_}.bc.unoptimized.linked
                      COMMAND opt -O3 ${NAME_}.bc.unoptimized.linked -o ${NAME_}.bc
                      COMMAND llc -O3 --disable-tail-calls -mcpu=${NVSHMEM_CLANG_ARCH} -mattr=${NVSHMEM_PTX_ARCH} ${NAME_}.bc -o ${NAME_}.ptx
                      COMMAND ${CUDA_HOME}/bin/ptxas -c -arch=${NVSHMEM_CLANG_ARCH} ${NAME_}.ptx -o ${CUBIN_UNLINKED_NAME}
                      COMMAND ${CUDA_HOME}/bin/nvlink -arch=${NVSHMEM_CLANG_ARCH} ${CUBIN_UNLINKED_NAME} -o ${CUBIN_NAME}
                      COMMAND rm ${NAME_}.bc.unoptimized ${NAME_}.bc.unoptimized.linked  ${NAME_}.bc ${NAME_}.ptx ${CUBIN_UNLINKED_NAME}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      DEPENDS nvshmem_perftest_helper libnvshmem_device_bitcode)
  endif()

  add_custom_target(${NAME_}_cubin ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME})

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
    DESTINATION "${NVSHMEM_PERFTEST_INSTALL_PREFIX}/${DIR}"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  if (NVSHMEM_PERFTEST_RELEASE_PREFIX)
    install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/${CUBIN_NAME}
            DESTINATION "${NVSHMEM_PERFTEST_RELEASE_PREFIX}/${DIR}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  endif()
endmacro()

add_subdirectory(device)
add_subdirectory(host)
