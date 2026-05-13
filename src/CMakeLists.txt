# this requires cmake 3.19
include(CheckCompilerFlag)
include(CheckSourceCompiles)
include(ExternalProject)

check_compiler_flag(CUDA -t4 NVCC_THREADS)

set(BOOTSTRAP_VERSION_MAJOR 3)
set(BOOTSTRAP_VERSION_MINOR 0)
set(BOOTSTRAP_VERSION_PATCH 0)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-vla-cxx-extension)
endif()

# Only forward CMAKE_CUDA_ARCHITECTURES when the user explicitly set it. Otherwise the
# device sub-project picks its own default based on NVSHMEM_DEVICELIB_CUDA_HOME's CTK
# version — which can legitimately differ from the host-side toolkit (same major, older
# minor) for ABI stability across a major release series.
if(NOT CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    list(APPEND my_cmake_cache_args
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    )
endif()

ExternalProject_Add(nvshmem_device_project
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/device
                    CMAKE_ARGS  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DNVSHMEM_PREFIX=${NVSHMEM_PREFIX}
                                -DNVSHMEM_DEBUG=${NVSHMEM_DEBUG}
                                -DNVSHMEM_VERBOSE=${NVSHMEM_VERBOSE}
                                -DNVSHMEM_DEVICELIB_CUDA_HOME=${NVSHMEM_DEVICELIB_CUDA_HOME}
                                -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING}
                                -DNVSHMEM_BUILD_WITH_CUTLASS=${NVSHMEM_BUILD_WITH_CUTLASS}
                                -DNVSHMEM_GPUNETIO_SUPPORT=${NVSHMEM_GPUNETIO_SUPPORT}
                                -DGPUNETIO_INCLUDE=${GPUNETIO_INCLUDE}
                                -DCUTLASS_HOME=${CUTLASS_HOME}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                                -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
                                -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                                -DCMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
                                -DPROJECT_VERSION=${PROJECT_VERSION}
				$ENV{CMAKE_CCACHE_FLAGS}
                    CMAKE_CACHE_ARGS ${my_cmake_cache_args}
                    STEP_TARGETS build install
                    BUILD_ALWAYS TRUE
                    BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a
                    )

add_library(nvshmem_device STATIC IMPORTED)
set_property(TARGET nvshmem_device PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a)

## Start bitcode library configurations

# GPUNetIO include flags for device code custom commands (bitcode, LTO-IR)
if(NVSHMEM_GPUNETIO_SUPPORT)
  set(DEVICE_LIB_GPUNETIO_INCLUDE_FLAGS -I ${GPUNETIO_INCLUDE})
else()
  set(DEVICE_LIB_GPUNETIO_INCLUDE_FLAGS "")
endif()

if(CMAKE_CXX_STANDARD)
  set(BITCODE_CXX_STD "c++${CMAKE_CXX_STANDARD}")
else()
  set(BITCODE_CXX_STD "c++17")
endif()

# Derive device library architectures from CMAKE_CUDA_ARCHITECTURES.
# Strips -real/-virtual suffixes and prepends sm_ to each entry.
set(DEVICE_LIB_ARCHS "")
foreach(ARCH_ENTRY IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" ARCH_NUM "${ARCH_ENTRY}")
  if(NOT "${ARCH_NUM}" STREQUAL "")
    list(APPEND DEVICE_LIB_ARCHS "sm_${ARCH_NUM}")
  endif()
endforeach()
list(REMOVE_DUPLICATES DEVICE_LIB_ARCHS)
message(STATUS "DEVICE_LIB_ARCHS: ${DEVICE_LIB_ARCHS}")

if(NVSHMEM_BUILD_BITCODE_LIBRARY)
  if(NOT NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
    configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in llvm_lib/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
  endif()

  # Filter architectures to only those supported by the discovered Clang.
  # Clang/LLVM 22+ supports SM 100+; older versions max out at SM 90.
  set(BITCODE_LIB_ARCHS "")
  if(LLVM_VERSION_MAJOR GREATER_EQUAL 22)
    set(_CLANG_MAX_SM 100)
  else()
    set(_CLANG_MAX_SM 90)
  endif()
  foreach(ARCH IN LISTS DEVICE_LIB_ARCHS)
    string(REGEX REPLACE "^sm_" "" _SM_NUM "${ARCH}")
    if(_SM_NUM LESS_EQUAL ${_CLANG_MAX_SM})
      list(APPEND BITCODE_LIB_ARCHS "${ARCH}")
    else()
      message(STATUS "Skipping bitcode for ${ARCH} (Clang/LLVM ${LLVM_VERSION_MAJOR} supports up to sm_${_CLANG_MAX_SM})")
    endif()
  endforeach()
  if(NOT BITCODE_LIB_ARCHS)
    message(WARNING "No bitcode architectures supported by Clang ${LLVM_VERSION_MAJOR}; skipping bitcode library")
    set(NVSHMEM_BUILD_BITCODE_LIBRARY OFF)
  endif()

  # Note - today we do optimization to avoid having a too large bitcode file. There are, as of clang 18, some existing
  # limitations around compiling device libraries which make optimizing this library fine. Once these issues are fixed
  # we will have to remove optimizations to enable more flexible builds.

  # Set bitcode build flags based on build type
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _BITCODE_BUILD_TYPE_UPPER)
  if(_BITCODE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    set(BITCODE_OPT_LEVEL "-O0")
  else()
    set(BITCODE_OPT_LEVEL "-O1")
  endif()
  if(_BITCODE_BUILD_TYPE_UPPER STREQUAL "DEBUG" OR _BITCODE_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
    set(BITCODE_DEBUG_FLAGS "-g")
  else()
    set(BITCODE_DEBUG_FLAGS "")
  endif()
  set(BITCODE_CLANG_FLAGS ${BITCODE_DEBUG_FLAGS} ${BITCODE_OPT_LEVEL})

  # Use the clang executable set by the top-level CMakeLists.txt (from find_package(Clang CONFIG)).
  if(NOT NVSHMEM_CLANG_EXECUTABLE OR NOT EXISTS "${NVSHMEM_CLANG_EXECUTABLE}")
    message(FATAL_ERROR "Bitcode library requires NVSHMEM_CLANG_EXECUTABLE (set by top-level find_package(Clang CONFIG)). Got: ${NVSHMEM_CLANG_EXECUTABLE}")
  endif()

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

  # Build per-arch bitcode files and install each one.
  set(BITCODE_ARCH_BC_FILES "")
  foreach(ARCH IN LISTS BITCODE_LIB_ARCHS)
    string(REPLACE "sm_" "" ARCH_NUM ${ARCH})
    set(ARCH_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device_${ARCH}.bc)

    if(_BITCODE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
      set(ARCH_OPT_COMMAND cp libnvshmem_device_${ARCH}.bc.unoptimized libnvshmem_device_${ARCH}.bc)
    else()
      set(ARCH_OPT_COMMAND llvm::opt --passes='internalize,inline,globaldce'
          -internalize-public-api-list='nvshmem_*,nvshmemx_*,nvshmemi_device_lib_version_d,nvshmemi_device_state_d'
          libnvshmem_device_${ARCH}.bc.unoptimized -o libnvshmem_device_${ARCH}.bc)
    endif()

    add_custom_command(OUTPUT ${ARCH_BC_FILE}
                      COMMAND ${NVSHMEM_CLANG_EXECUTABLE} ${NVSHMEM_CLANG_CXX_FLAGS_EXTRA_LIST} -c -emit-llvm ${BITCODE_CLANG_FLAGS} -std=${BITCODE_CXX_STD} -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only
                      --cuda-gpu-arch=${ARCH} -I ${CMAKE_CURRENT_SOURCE_DIR}/include -I ${CUDA_HOME}/include/cccl
                      -I ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib
                      ${DEVICE_LIB_GPUNETIO_INCLUDE_FLAGS}
                      -D__clang_llvm_bitcode_lib__ ${CMAKE_CURRENT_SOURCE_DIR}/device/init/init_device.cu -o libnvshmem_device_${ARCH}.bc.unoptimized
                      COMMAND ${ARCH_OPT_COMMAND}
                      COMMAND llvm::llvm-dis libnvshmem_device_${ARCH}.bc
                      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/bitcode_lib_cleanup.sh ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib/libnvshmem_device_${ARCH}.ll ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib/libnvshmem_device_${ARCH}.ll.new
                      COMMAND llvm::llvm-as libnvshmem_device_${ARCH}.ll.new -o ${ARCH_BC_FILE}
                      COMMAND rm libnvshmem_device_${ARCH}.bc.unoptimized libnvshmem_device_${ARCH}.ll libnvshmem_device_${ARCH}.ll.new libnvshmem_device_${ARCH}.bc
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib
                      DEPENDS device/init/init_device.cu)

    list(APPEND BITCODE_ARCH_BC_FILES ${ARCH_BC_FILE})
  endforeach()

  add_custom_target(libnvshmem_device_bitcode ALL DEPENDS ${BITCODE_ARCH_BC_FILES})

  # Install per-arch bitcode files.
  foreach(ARCH IN LISTS BITCODE_LIB_ARCHS)
    install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device_${ARCH}.bc
            DESTINATION lib/
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
  endforeach()

  # Create a backward-compatible symlink: libnvshmem_device.bc -> libnvshmem_device_sm_90.bc
  # (sm_90 was the previous default architecture for the bitcode library)
  if("sm_90" IN_LIST BITCODE_LIB_ARCHS)
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
      libnvshmem_device_sm_90.bc \
      \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/libnvshmem_device.bc)")
  endif()
endif()

## End bitcode library configurations
## Start ltoir library configurations
#
# LTOIR version compatibility:
#   - CUDA 12.8+: nvcc --ltoir compiles per-arch .ltoir files
#   - CUDA 13.0+: fatbinary --image3=kind=nvvm creates multi-arch fatbins
#                  that nvcc -dlink can consume
#   - CUDA 12.8-12.x: fatbinary --image3=kind=nvvm creates fatbins but
#                  nvcc -dlink cannot extract from them (unresolved symbols).
#                  Fall back to a single sm_90 .ltoir file.
#
if(NVSHMEM_BUILD_LTOIR_LIBRARY)
  # nvcc --ltoir requires CUDA 12.8 or later
  if(CUDAToolkit_VERSION_MAJOR LESS 12 OR
     (CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 8))
    message(WARNING "NVSHMEM_BUILD_LTOIR_LIBRARY requires CUDA 12.8 or later (nvcc --ltoir), but found CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}. Disabling LTOIR library build.")
    set(NVSHMEM_BUILD_LTOIR_LIBRARY OFF CACHE BOOL "Build the nvshmem_device LTOIR library and tests" FORCE)
  else()

  if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    # CUDA 13+: build multi-arch LTOIR fatbin from all target architectures
    set(LTOIR_LIB_ARCHS ${DEVICE_LIB_ARCHS})
  else()
    # CUDA 12.8-12.x: single-arch sm_90 only. Multi-arch fatbins created with
    # fatbinary --image3=kind=nvvm cannot be consumed by nvcc -dlink on CUDA 12
    # (unresolved symbols). Hardcoded to sm_90 to match pre-multi-arch behavior.
    set(LTOIR_LIB_ARCHS "sm_90")
    message(STATUS "CUDA ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}: "
            "LTOIR fatbin extraction requires CUDA 13+. Building single-arch sm_90 LTOIR.")
  endif()

  if(NOT LTOIR_LIB_ARCHS)
    message(WARNING "No architectures for LTOIR library. Disabling LTOIR build.")
    set(NVSHMEM_BUILD_LTOIR_LIBRARY OFF CACHE BOOL "" FORCE)
  endif()

  set(LTOIR_CXX_STD "c++17")

  # ––– paths –––
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

  # ––– compile per-arch LTOIR files –––
  set(LTOIR_ARCH_FILES "")
  set(FATBIN_LTOIR_IMAGE_ARGS "")
  if(NOT TARGET cuda::nvcc)
    message(FATAL_ERROR "nvcc not found in the selected CUDA Toolkit. Cannot build the LTOIR library.")
  endif()
  foreach(ARCH IN LISTS LTOIR_LIB_ARCHS)
    string(REPLACE "sm_" "" ARCH_NUM ${ARCH})
    set(ARCH_LTOIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device_${ARCH}.ltoir)

    add_custom_command(
      OUTPUT ${ARCH_LTOIR_FILE}
      COMMAND cuda::nvcc -std=${LTOIR_CXX_STD} -arch=${ARCH} -dlto --ltoir
              -I ${CMAKE_CURRENT_SOURCE_DIR}/include
              ${DEVICE_LIB_GPUNETIO_INCLUDE_FLAGS}
              -DNVSHMEM_BUILD_LTOIR_LIBRARY
              -o ${ARCH_LTOIR_FILE}
              ${CMAKE_CURRENT_SOURCE_DIR}/device/init/init_device.cu
      DEPENDS device/init/init_device.cu
      VERBATIM
    )

    list(APPEND LTOIR_ARCH_FILES ${ARCH_LTOIR_FILE})
    list(APPEND FATBIN_LTOIR_IMAGE_ARGS "--image3=kind=nvvm,sm=${ARCH_NUM},file=${ARCH_LTOIR_FILE}")
  endforeach()

  if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    # CUDA 13+: combine per-arch LTOIR files into a multi-arch fatbin
    if(NOT TARGET cuda::fatbinary)
      message(FATAL_ERROR "fatbinary not found in the selected CUDA Toolkit. Cannot build the LTOIR fatbin.")
    endif()
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
      COMMAND cuda::fatbinary --64 --create=${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
              ${FATBIN_LTOIR_IMAGE_ARGS}
      DEPENDS ${LTOIR_ARCH_FILES}
      COMMAND_EXPAND_LISTS
    )
  else()
    # CUDA 12.x: copy the single-arch .ltoir as the "fatbin" (no actual bundling)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device_sm_90.ltoir
              ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
      DEPENDS ${LTOIR_ARCH_FILES}
    )
  endif()

  if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13 AND TARGET cuda::cuobjdump)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin.cuobjdump.stamp
      COMMAND cuda::cuobjdump --dump-elf-symbols ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
      COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin.cuobjdump.stamp
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
    )
    set(LTOIR_VALIDATION_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin.cuobjdump.stamp)
  else()
    set(LTOIR_VALIDATION_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin)
  endif()

  add_custom_target(libnvshmem_device_ltoir ALL
                    DEPENDS ${LTOIR_VALIDATION_DEPENDS} ${LTOIR_ARCH_FILES})

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.ltoir.fatbin
          DESTINATION lib/
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  # Also install per-arch LTOIR files (needed by nvcc -dlink for test/perftest)
  foreach(ARCH IN LISTS LTOIR_LIB_ARCHS)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device_${ARCH}.ltoir
            DESTINATION lib/)
  endforeach()
  endif()
  endif()
## End ltoir library configurations



if(NVSHMEM_BUILD_EXAMPLES)
  set(NVSHMEM_EXAMPLES_RELEASE_PREFIX "bin/examples")
  add_subdirectory(${CMAKE_SOURCE_DIR}/examples ${CMAKE_BINARY_DIR}/examples)
endif()
if(NVSHMEM_BUILD_TESTS)
  set(NVSHMEM_PERFTEST_RELEASE_PREFIX "bin/perftest")
  add_subdirectory(${CMAKE_SOURCE_DIR}/perftest ${CMAKE_BINARY_DIR}/perftest)
  if(EXISTS ${CMAKE_SOURCE_DIR}/test)
    if (NVSHMEM_INSTALL_FUNCTIONAL_TESTS)
      set(NVSHMEM_TEST_RELEASE_PREFIX "bin/test")
    endif()
    add_subdirectory(${CMAKE_SOURCE_DIR}/test ${CMAKE_BINARY_DIR}/test)
  endif()
endif()

# Start bootstrap definitions
add_library(
  nvshmem_bootstrap_pmi SHARED
)
set(ACTIVE_BOOTSTRAPS nvshmem_bootstrap_pmi)
target_sources(nvshmem_bootstrap_pmi PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                             modules/bootstrap/pmi/bootstrap_pmi.cpp
                                             modules/bootstrap/pmi/simple-pmi/simple_pmi.cpp
                                             modules/bootstrap/pmi/simple-pmi/simple_pmiutil.cpp)
target_include_directories(nvshmem_bootstrap_pmi
                           PRIVATE include
                           modules/bootstrap/common
                           modules/bootstrap/pmi/simple-pmi
)

add_library(
  nvshmem_bootstrap_pmi2 SHARED
)
set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmi2)
target_sources(nvshmem_bootstrap_pmi2 PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                              modules/bootstrap/pmi/bootstrap_pmi.cpp
                                              modules/bootstrap/pmi/pmi-2/pmi2_api.c
                                              modules/bootstrap/pmi/pmi-2/pmi2_util.c)
target_compile_definitions(nvshmem_bootstrap_pmi2 PRIVATE NVSHMEM_BUILD_PMI2)
target_link_options(nvshmem_bootstrap_pmi2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_bootstrap.sym")
target_include_directories(nvshmem_bootstrap_pmi2
                           PRIVATE include
                           modules/bootstrap/common
                           modules/bootstrap/pmi/pmi-2
)

if(NVSHMEM_PMIX_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmix SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmix)
  target_sources(nvshmem_bootstrap_pmix PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                                modules/bootstrap/pmix/bootstrap_pmix.c)
  find_library(PMIX_lib NAMES pmix HINTS "${PMIX_HOME}/lib")
  target_link_libraries(nvshmem_bootstrap_pmix PRIVATE ${PMIX_lib})
  target_include_directories(nvshmem_bootstrap_pmix
                             PRIVATE include
                             modules/bootstrap/common
                             ${PMIX_HOME}/include

  )
else()
set(NVSHMEM_DEFAULT_PMIX OFF)
endif()

if(NVSHMEM_MPI_SUPPORT)
  add_library(
    nvshmem_bootstrap_mpi SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_mpi)
  target_sources(nvshmem_bootstrap_mpi PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                               modules/bootstrap/mpi/bootstrap_mpi.c)
  target_link_libraries(nvshmem_bootstrap_mpi PRIVATE MPI::MPI_C)
  target_include_directories(nvshmem_bootstrap_mpi
                             PRIVATE include
                             modules/bootstrap/common
                             include/modules/common
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  add_library(
    nvshmem_bootstrap_shmem SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_shmem)
  target_sources(nvshmem_bootstrap_shmem PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                                 modules/bootstrap/shmem/bootstrap_shmem.c)
  target_link_libraries(nvshmem_bootstrap_shmem PRIVATE shmem)
  target_include_directories(nvshmem_bootstrap_shmem
                             PRIVATE include
                             modules/bootstrap/common
                             include/modules/common
  )

  # Detect OpenSHMEM 1.5+ before enabling teams-based bootstrap support.
  # OpenSHMEM 1.4 implementations such as OpenMPI oshmem expose incompatible
  # teams signatures (e.g. shmem_team_sync returns void), so gate the teams path
  # on the standard version macros first and only then verify the byte-level
  # collectives used by the bootstrap are available.
  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${SHMEM_INCLUDE})
  set(CMAKE_REQUIRED_LIBRARIES ${SHMEM_LIB})
  check_c_source_compiles("
    #include <shmem.h>
    #if !defined(SHMEM_MAJOR_VERSION) || !defined(SHMEM_MINOR_VERSION)
    #error OpenSHMEM version macros are not defined
    #endif
    #if (SHMEM_MAJOR_VERSION < 1) || (SHMEM_MAJOR_VERSION == 1 && SHMEM_MINOR_VERSION < 5)
    #error OpenSHMEM version is older than 1.5
    #endif
    int main(void) { return 0; }
  " NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)

  if(NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)
    check_c_source_compiles("
      #include <shmem.h>
      int main(void) {
          shmem_team_t team = SHMEM_TEAM_WORLD;
          /* Verify byte-level teams collectives are declared */
          int (*f1)(shmem_team_t, uint8_t *, const uint8_t *, size_t) = shmem_uint8_fcollect;
          int (*f2)(shmem_team_t, uint8_t *, const uint8_t *, size_t) = shmem_uint8_alltoall;
          (void)team; (void)f1; (void)f2;
          return 0;
      }
    " NVSHMEM_SHMEM_HAS_TEAMS)
  endif()
  if(NVSHMEM_SHMEM_HAS_TEAMS)
    message(STATUS "OpenSHMEM >= 1.5 teams byte-level collectives detected")
    target_compile_definitions(nvshmem_bootstrap_shmem PRIVATE NVSHMEM_SHMEM_HAS_TEAMS)
  elseif(NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)
    message(STATUS "OpenSHMEM >= 1.5 detected, but required teams byte-level collectives were not found")
  else()
    message(STATUS "OpenSHMEM < 1.5 detected; disabling teams-based bootstrap support")
  endif()

  # Detect deprecated OpenSHMEM active-set collectives (shmem_collect32, shmem_alltoall32).
  # These are the pre-1.5 APIs that require manual pSync scratch buffers.
  check_c_source_compiles("
    #include <shmem.h>
    int main(void) {
        long scratch[SHMEM_COLLECT_SYNC_SIZE];
        scratch[0] = SHMEM_SYNC_VALUE;
        shmem_collect32(scratch, scratch, 1, 0, 0, 1, scratch);
        shmem_alltoall32(scratch, scratch, 1, 0, 0, 1, scratch);
        return 0;
    }
  " NVSHMEM_SHMEM_HAS_ACTIVE_SET)
  if(NVSHMEM_SHMEM_HAS_ACTIVE_SET)
    message(STATUS "OpenSHMEM active-set collectives detected (runtime fallback available)")
    target_compile_definitions(nvshmem_bootstrap_shmem PRIVATE NVSHMEM_SHMEM_HAS_ACTIVE_SET)
  elseif(NOT NVSHMEM_SHMEM_HAS_TEAMS)
    message(WARNING "OpenSHMEM implementation supports neither teams-based nor "
            "active-set collectives. Skipping SHMEM bootstrap.")
  endif()
endif()

# Always compile uid bootstrap as no dependency on external sources or libraries
add_library(nvshmem_bootstrap_uid SHARED)
set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_uid)
target_sources(nvshmem_bootstrap_uid PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                             modules/bootstrap/uid/ncclSocket/ncclsocket_socket.cpp
					                                   modules/bootstrap/uid/bootstrap_uid.cpp)
target_include_directories(nvshmem_bootstrap_uid
		                       PRIVATE include
                           include/bootstrap_device_host
                           include/internal/bootstrap_host
                           include/internal/bootstrap_host_transport
                           modules/bootstrap/common
			                     modules/bootstrap/uid/ncclSocket
)


set_target_properties(${ACTIVE_BOOTSTRAPS}
                      PROPERTIES PREFIX ""
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH}
                      SOVERSION ${BOOTSTRAP_VERSION_MAJOR})

foreach(BOOTSTRAP ${ACTIVE_BOOTSTRAPS})
  target_compile_features(${BOOTSTRAP} PUBLIC cxx_std_17)
endforeach()

# End bootstrap definitions

# Start nvshmem lib definitions
add_library(
  nvshmem_host SHARED
)

## Start generic variable configuration
set_target_properties(nvshmem_host
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
		      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_config LIBNAME)
  target_compile_features(${LIBNAME} PUBLIC cxx_std_17)

  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","aarch64">,NVSHMEM_AARCH64,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror
              -Wall
              -Wextra>
    )
  endif()
endmacro()

set(NVSHMEM_PYTHON_SOURCES
    ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/
)

set(NVSHMEM_DEVICE_SOURCES
    device/init/init_device.cu
    device/launch/collective_launch.cpp
)

set(NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT
    host/comm/rma.cu
    host/stream/comm/quiet_on_stream.cu
    host/stream/comm/cuda_interface_sync.cu
    host/stream/coll/alltoall/alltoall.cu
    host/stream/coll/barrier/barrier.cu
    host/stream/coll/broadcast/broadcast.cu
    host/stream/coll/fcollect/fcollect.cu
    host/stream/coll/rdxn/reduce_and.cu
    host/stream/coll/rdxn/reduce_or.cu
    host/stream/coll/rdxn/reduce_xor.cu
    host/stream/coll/rdxn/reduce_min.cu
    host/stream/coll/rdxn/reduce_max.cu
    host/stream/coll/rdxn/reduce_prod.cu
    host/stream/coll/rdxn/reduce_sum.cu
    host/stream/coll/rdxn/reduce_team.cu
    host/stream/coll/reducescatter/reducescatter_and.cu
    host/stream/coll/reducescatter/reducescatter_or.cu
    host/stream/coll/reducescatter/reducescatter_xor.cu
    host/stream/coll/reducescatter/reducescatter_min.cu
    host/stream/coll/reducescatter/reducescatter_max.cu
    host/stream/coll/reducescatter/reducescatter_prod.cu
    host/stream/coll/reducescatter/reducescatter_sum.cu
)

set(NVSHMEM_HOST_SOURCES
    host/bootstrap/bootstrap.cpp
    host/bootstrap/bootstrap_loader.cpp
    host/coll/cpu_coll.cpp
    host/coll/alltoall/alltoall.cpp
    host/coll/alltoall/alltoall_on_stream.cpp
    host/coll/barrier/barrier.cpp
    host/coll/barrier/barrier_on_stream.cpp
    host/coll/broadcast/broadcast.cpp
    host/coll/broadcast/broadcast_on_stream.cpp
    host/coll/fcollect/fcollect.cpp
    host/coll/fcollect/fcollect_on_stream.cpp
    host/coll/rdxn/rdxn.cpp
    host/coll/rdxn/rdxn_on_stream.cpp
    host/coll/reducescatter/reducescatter.cpp
    host/coll/reducescatter/reducescatter_on_stream.cpp
    host/comm/putget.cpp
    host/comm/fence.cpp
    host/comm/quiet.cpp
    host/comm/sync.cpp
    host/comm/amo.cpp
    host/proxy/proxy.cpp
    host/transport/transport.cpp
    host/transport/p2p/p2p.cpp
    host/init/cudawrap.cpp
    host/init/init.cu
    host/init/init_nvtx.cpp
    host/init/nvmlwrap.cpp
    host/init/query_host.cpp
    host/mem/mem.cpp
    host/mem/mem_heap.cpp
    host/mem/mem_transport.cpp
    host/team/team.cu
    host/team/team_internal.cpp
    host/team/team_internal_cuda.cu
    host/team/team_internal_nvls.cpp
    host/topo/topo.cpp
    host/util/debug.cpp
    host/util/env_vars.cpp
    host/util/util.cpp
    host/util/sockets.cpp
    host/util/shared_memory.cpp
)

target_include_directories(
  nvshmem_host
  PRIVATE include
          include/host/env
          device/coll
          host/coll
          host/stream/coll
          host/topo
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

if (NVSHMEM_GPUNETIO_SUPPORT)
  target_include_directories(nvshmem_host PRIVATE ${GPUNETIO_INCLUDE})
  target_include_directories(nvshmem_host INTERFACE
    $<BUILD_INTERFACE:${GPUNETIO_INCLUDE}>
    $<INSTALL_INTERFACE:include/gpunetio>
  )
endif()

if (NVSHMEM_BUILD_WITH_CUTLASS)
    target_compile_definitions(nvshmem_host PUBLIC CUTLASS_ENABLED)
    target_compile_definitions(nvshmem_device INTERFACE CUTLASS_ENABLED)

    if ("${CUTLASS_HOME}" STREQUAL "")
        message(FATAL_ERROR "Please set CUTLASS_HOME to CUTLASS directory.")
    endif()
        set(CUTLASS_INCLUDE_DIR "${CUTLASS_HOME}/include")
        set(CUTLASS_EXAMPLE_INCLUDE_DIR "${CUTLASS_HOME}/examples/common")
        set(CUTLASS_UTIL_INCLUDE_DIR "${CUTLASS_HOME}/tools/util/include")

    message(STATUS "CUTLASS include directory: ${CUTLASS_INCLUDE_DIR}")
    target_include_directories(nvshmem_host PUBLIC
            ${CUTLASS_UTIL_INCLUDE_DIR}
            ${CUTLASS_INCLUDE_DIR}
            ${CUTLASS_EXAMPLE_INCLUDE_DIR})

endif()

target_link_libraries(nvshmem_host PRIVATE CUDA::cudart_static)
if (CCCL_FOUND AND CUDAToolkit_VERSION_MAJOR GREATER 12)
  target_link_libraries(nvshmem_host PUBLIC CCCL::CCCL)
endif()
nvshmem_library_set_base_config(nvshmem_host)
## End generic variable configuration

## Start transports
set(TRANSPORT_VERSION_MAJOR 6)
set(TRANSPORT_VERSION_MINOR 0)
set(TRANSPORT_VERSION_PATCH 0)

check_source_compiles(C
  "#include <infiniband/verbs.h>
  int main(void) { int x = IBV_ACCESS_RELAXED_ORDERING; return 1; }"
  HAVE_IBV_ACCESS_RELAXED_ORDERING
)

set(ACTIVE_TRANSPORTS)

if(NVSHMEM_UCX_SUPPORT)
  add_library(
    nvshmem_transport_ucx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ucx)
  target_sources(nvshmem_transport_ucx PRIVATE
                 modules/transport/ucx/ucx.cpp
                 modules/transport/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_ucx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ucx
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ucx PRIVATE ucx::ucs ucx::ucp CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ucx PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ucx PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ucx)
endif()

if(NVSHMEM_IBRC_SUPPORT)
  add_library(
    nvshmem_transport_ibrc SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibrc)
  target_sources(nvshmem_transport_ibrc PRIVATE
                 modules/transport/ibrc/ibrc.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
  )
  target_link_options(nvshmem_transport_ibrc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibrc
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ibrc PRIVATE CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ibrc PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ibrc PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibrc PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibrc)
endif()

if(NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT OR NVSHMEM_GPUNETIO_SUPPORT)
  find_library(MLX5_lib NAMES mlx5)
endif()

if(NVSHMEM_IBDEVX_SUPPORT)
  add_library(
    nvshmem_transport_ibdevx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibdevx)
  target_sources(nvshmem_transport_ibdevx PRIVATE
                 modules/transport/ibdevx/ibdevx.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibdevx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibdevx
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ibdevx PRIVATE ${MLX5_lib} CUDA::cudart_static)
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibdevx PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibdevx)
endif()

if(NVSHMEM_IBGDA_SUPPORT)
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED; return 1; }"
    HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
  )
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UMEM_MASK_DMABUF; return 1; }"
    HAVE_MLX5DV_UMEM_MASK_DMABUF
  )
  add_library(
    nvshmem_transport_ibgda SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibgda)
  target_sources(nvshmem_transport_ibgda PRIVATE
                 modules/transport/ibgda/ibgda.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibgda PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibgda
                             PRIVATE include
                             modules/transport/common

  )
  target_link_libraries(nvshmem_transport_ibgda PRIVATE ${MLX5_lib} CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ibgda PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ibgda PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  if(HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED=${HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED})
  endif()
  if(HAVE_MLX5DV_UMEM_MASK_DMABUF)
      target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UMEM_MASK_DMABUF=${HAVE_MLX5DV_UMEM_MASK_DMABUF})
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibgda)
endif()

if(NVSHMEM_GPUNETIO_SUPPORT)
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED; return 1; }"
    HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
  )
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UMEM_MASK_DMABUF; return 1; }"
    HAVE_MLX5DV_UMEM_MASK_DMABUF
  )
  add_library(
    nvshmem_transport_gpunetio SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_gpunetio)
  target_sources(nvshmem_transport_gpunetio PRIVATE
                 modules/transport/gpunetio/gpunetio.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_gpunetio PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_gpunetio
                             PRIVATE include
                             modules/transport/common

  )
  target_link_libraries(nvshmem_transport_gpunetio PRIVATE ${MLX5_lib} CUDA::cudart_static)
  target_link_libraries(nvshmem_transport_gpunetio PRIVATE doca_gpunetio)
  target_include_directories(nvshmem_transport_gpunetio PRIVATE ${GPUNETIO_INCLUDE})
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_gpunetio PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_gpunetio PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  if(HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED)
    target_compile_definitions(nvshmem_transport_gpunetio PRIVATE HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED=${HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED})
  endif()
  if(HAVE_MLX5DV_UMEM_MASK_DMABUF)
      target_compile_definitions(nvshmem_transport_gpunetio PRIVATE HAVE_MLX5DV_UMEM_MASK_DMABUF=${HAVE_MLX5DV_UMEM_MASK_DMABUF})
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_gpunetio PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_gpunetio)
endif()

if(NVSHMEM_LIBFABRIC_SUPPORT)
  find_library(FABRIC_lib NAMES fabric HINTS "${LIBFABRIC_HOME}/lib" "${LIBFABRIC_HOME}/lib64")
  add_library(
    nvshmem_transport_libfabric SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_libfabric)
  target_sources(nvshmem_transport_libfabric PRIVATE
                 modules/transport/libfabric/libfabric.cpp
                 modules/transport/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_libfabric PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_libfabric
                             PRIVATE include
                             modules/transport/common
                             "${LIBFABRIC_HOME}/include/"
  )
  target_link_libraries(nvshmem_transport_libfabric PRIVATE ${FABRIC_lib} CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_libfabric PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_libfabric PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_libfabric)
endif()

set_target_properties(${ACTIVE_TRANSPORTS}
                      PROPERTIES PREFIX ""
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH}
                      SOVERSION ${TRANSPORT_VERSION_MAJOR})
## End transports

## Start dependencies
if(NVSHMEM_USE_NCCL)
  target_include_directories(nvshmem_host PRIVATE ${NCCL_INCLUDE})
endif()

if(NVSHMEM_USE_DLMALLOC)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      host/mem/dlmalloc.cpp
  )
else()
set(NVSHMEM_HOST_SOURCES
    ${NVSHMEM_HOST_SOURCES}
    host/mem/custom_malloc.cpp
  )
endif()
## End dependecies

## Start Numba support requirements
set(INC_EXPORTS
    ${INC_EXPORTS}
    non_abi/device/pt-to-pt/transfer_device.cuh
)
configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
## End Numba support requirements

## Start inlining requirements
set(INC_EXPORTS
    ${INC_EXPORTS}
    non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh
)
if(NOT NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
set(NVSHMEM_DEVICE_SOURCES
    ${NVSHMEM_DEVICE_SOURCES}
    device/comm/transfer_device.cu
)
endif()

configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/device/comm/transfer_device.cu COPYONLY)

## End inlining requirements

## Start final lib prep
target_link_libraries(nvshmem_host PRIVATE nvshmem_device)
target_link_options(nvshmem_host PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_host.sym")
set_source_files_properties(${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem_host PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT})

## End final lib prep
# End nvshmem lib definitions

# Start nvshmem-info definitions.
add_executable(nvshmem-info)
target_sources(nvshmem-info PRIVATE bin/nvshmem-info.cpp host/util/env_vars.cpp host/util/util.cpp host/init/init_nvtx.cpp)
set_target_properties(nvshmem-info PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
target_link_libraries(nvshmem-info CUDA::cudart_static CUDA::cuda_driver nvshmem_host nvshmem_device)
target_include_directories(nvshmem-info PRIVATE include include/host/env)
target_compile_definitions(nvshmem-info PRIVATE NVSHMEM_INFO_ONLY)
# End nvshmem-info definitions.

# Start header configuration and export
configure_file(include/non_abi/nvshmem_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_version.h @ONLY)
configure_file(include/non_abi/nvshmem_build_options.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_build_options.h)

set(INC_EXPORTS
    ${INC_EXPORTS}
    bootstrap_device_host/nvshmem_uniqueid.h
    device_host/nvshmem_proxy_channel.h
    device_host/nvshmem_types.h
    device_host_transport/nvshmem_common_transport.h
    non_abi/device/common/nvshmemi_common_device.cuh
    non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh
    device/nvshmem_coll_defines.cuh
    device/nvshmem_defines.h
    device/nvshmem_device_macros.h
    device/nvshmemx_collective_launch_apis.h
    device/nvshmemx_coll_defines.cuh
    device/nvshmemx_defines.h
    device/tile/nvshmemx_tile_api.hpp
    device/tile/nvshmemx_tile_api_defines.cuh
    non_abi/device/coll/alltoall.cuh
    non_abi/device/coll/barrier.cuh
    non_abi/device/coll/broadcast.cuh
    non_abi/device/coll/defines.cuh
    non_abi/device/coll/fcollect.cuh
    non_abi/device/coll/reduce.cuh
    non_abi/device/coll/reducescatter.cuh
    non_abi/device/coll/utils.cuh
    non_abi/device/wait/nvshmemi_wait_until_apis.cuh
    non_abi/device/pt-to-pt/ibgda_device.cuh
    non_abi/device/pt-to-pt/gdaki_device.cuh
    non_abi/device/pt-to-pt/proxy_device.cuh
    non_abi/device/pt-to-pt/tma_device.cuh
    non_abi/device/pt-to-pt/utils_device.h
    non_abi/device/pt-to-pt/tile.cuh
    non_abi/device/common/nvshmemi_tile_utils.cuh
    device_host/nvshmem_common.cuh
    device_host/nvshmemi_extern_constant.h
    device_host/nvshmem_tensor.h
    device_host_transport/nvshmem_common_ibgda.h
    device_host_transport/nvshmem_common_gpunetio.h
    device_host_transport/nvshmem_constants.h
    non_abi/device/team/nvshmemi_team_defines.cuh
    host/nvshmem_api.h
    host/nvshmem_coll_api.h
    host/nvshmem_macros.h
    host/nvshmemx_api.h
    host/nvshmemx_coll_api.h
    non_abi/nvshmemi_cccl_compat.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
    non_abi/nvshmem_version.h
    nvshmem.h
    nvshmem_host.h
    nvshmemx.h
)

foreach(INC ${INC_EXPORTS})
  configure_file(include/${INC} ${CMAKE_CURRENT_BINARY_DIR}/include/${INC} COPYONLY)
endforeach(INC ${INC_EXPORTS})
# End header configuration and export

# start share configuration

configure_file(${CMAKE_SOURCE_DIR}/nvshmem_bootstrap.sym
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/nvshmem_bootstrap.sym COPYONLY)

set(BOOTSTRAP_EXPORTS
    common/bootstrap_util.cpp
    common/bootstrap_util.h
    common/env_defs.h
    mpi/bootstrap_mpi.c
    pmi/bootstrap_pmi.cpp
    pmix/bootstrap_pmix.c
    shmem/bootstrap_shmem.c
    uid/bootstrap_uid.cpp
    uid/bootstrap_uid_remap.h
    uid/ncclSocket/ncclsocket_checks.h
    uid/ncclSocket/ncclsocket_debug.h
    uid/ncclSocket/ncclsocket_nccl.h
    uid/ncclSocket/ncclsocket_param.h
    uid/ncclSocket/ncclsocket_socket.hpp
    uid/ncclSocket/ncclsocket_socket.cpp
    uid/ncclSocket/ncclsocket_utils.h
    uid/bootstrap_uid_types.hpp
)

foreach(SHARE ${BOOTSTRAP_EXPORTS})
  get_filename_component(SHAREFILE ${SHARE} NAME)
  configure_file(modules/bootstrap/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${BOOTSTRAP_EXPORTS})

set(BOOTSTRAP_INC_EXPORTS
    bootstrap_device_host/nvshmem_uniqueid.h
    bootstrap_host_transport/env_defs_internal.h
    internal/bootstrap_host/nvshmemi_bootstrap.h
    internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h
    non_abi/nvshmem_version.h
    non_abi/nvshmemx_error.h
)

foreach(SHARE ${BOOTSTRAP_INC_EXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/include/${SHARE} COPYONLY)
endforeach(SHARE ${BOOTSTRAP_INC_EXPORTS})

set(BOOTSTRAP_CMAKE_EXPORTS
    CMakeLists.txt
    common/CMakeLists.txt
    mpi/CMakeLists.txt
    pmi/CMakeLists.txt
    pmix/CMakeLists.txt
    shmem/CMakeLists.txt
    uid/CMakeLists.txt
)

foreach(SHARE ${BOOTSTRAP_CMAKE_EXPORTS})
  configure_file(modules/bootstrap/${SHARE}.in ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/${SHARE} @ONLY)
endforeach(SHARE ${BOOTSTRAP_CMAKE_EXPORTS})

configure_file(${CMAKE_SOURCE_DIR}/cmake_config/NVSHMEMCxxStandard.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/cmake_config/NVSHMEMCxxStandard.cmake COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/nvshmem_transport.sym
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/nvshmem_transport.sym COPYONLY)

set(TRANSPORT_INC_EXPORTS
    bootstrap_host_transport/env_defs_internal.h
    device_host_transport/nvshmem_common_ibgda.h
    device_host_transport/nvshmem_common_gpunetio.h
    device_host_transport/nvshmem_common_transport.h
    device_host_transport/nvshmem_constants.h
    internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h
    internal/host/scope_guard.h
    internal/host_transport/cudawrap.h
    internal/host_transport/nvshmemi_transport_defines.h
    internal/host_transport/transport.h
    non_abi/nvshmem_version.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
)

foreach(SHARE ${TRANSPORT_INC_EXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/include/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_INC_EXPORTS})

set(TRANSPORT_EXPORTS
    common/env_defs.h
    common/mlx5_ifc.h
    common/mlx5_prm.h
    common/transport_common.cpp
    common/transport_common.h
    common/transport_gdr_common.cpp
    common/transport_gdr_common.h
    common/transport_ib_common.cpp
    common/transport_ib_common.h
    common/transport_mlx5_common.cpp
    common/transport_mlx5_common.h
    ibdevx/ibdevx.cpp
    ibdevx/ibdevx.h
    gpunetio/gpunetio.cpp
    ibgda/ibgda.cpp
    ibrc/ibrc.cpp
    libfabric/libfabric.cpp
    libfabric/libfabric.h
    ucx/ucx.cpp
    ucx/ucx.h
)

foreach(SHARE ${TRANSPORT_EXPORTS})
  configure_file(modules/transport/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_EXPORTS})

set(TRANSPORT_CMAKE_EXPORTS
    CMakeLists.txt
    common/CMakeLists.txt
    gpunetio/CMakeLists.txt
    libfabric/CMakeLists.txt
    ibdevx/CMakeLists.txt
    ibgda/CMakeLists.txt
    ibrc/CMakeLists.txt
    ucx/CMakeLists.txt
)

foreach(SHARE ${TRANSPORT_CMAKE_EXPORTS})
  configure_file(modules/transport/${SHARE}.in ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} @ONLY)
endforeach(SHARE ${TRANSPORT_CMAKE_EXPORTS})

# Copy vendored GPUNetIO Open sources into the transport-plugins share tarball
file(COPY ${CMAKE_SOURCE_DIR}/externals/gpunetio/CMakeLists.txt
          ${CMAKE_SOURCE_DIR}/externals/gpunetio/include
          ${CMAKE_SOURCE_DIR}/externals/gpunetio/src
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/gpunetio/gpunetio_src)

configure_file(${CMAKE_SOURCE_DIR}/cmake_config/NVSHMEMCxxStandard.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/cmake_config/NVSHMEMCxxStandard.cmake COPYONLY)
# end share configuration

# start custom builds for nvshmem packages
add_custom_target(git_commit ALL COMMAND test -f git_commit.txt || git rev-parse HEAD > git_commit.txt || echo "not built from a git repo" > git_commit.txt
                  COMMAND grep -q NVSHMEM_MAJOR git_commit.txt || echo "NVSHMEM_MAJOR := ${PROJECT_VERSION_MAJOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_MINOR git_commit.txt || echo "NVSHMEM_MINOR := ${PROJECT_VERSION_MINOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PATCH git_commit.txt || echo "NVSHMEM_PATCH := ${PROJECT_VERSION_PATCH}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PACKAGE git_commit.txt || echo "NVSHMEM_PACKAGE := ${PROJECT_VERSION_TWEAK}" >> git_commit.txt
                  COMMAND cp git_commit.txt version.txt
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
                             ${CMAKE_CURRENT_SOURCE_DIR}/../version.txt
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(NVSHMEM_BUILD_PACKAGES AND NVSHMEM_BUILD_HYDRA_LAUNCHER)
  add_custom_target(hydra
                    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_hydra.sh ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build
                    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_nameserver
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_persist
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_pmi_proxy
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun.hydra
                  )
  add_dependencies(nvshmem_host git_commit hydra)
endif()
add_dependencies(nvshmem_host nvshmem_device_project-build)

# Start Installation definitions
install(CODE "
  # Patch NVSHMEM target files
  file(GLOB_RECURSE targets_files \"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Export/*/NVSHMEM*Targets*.cmake\")
  foreach(targets_file IN LISTS targets_files)
    message(STATUS \"Processing targets_file=\${targets_file}\")
    file(READ \"\${targets_file}\" _file_contents)
    string(REPLACE \"\\\${_IMPORT_PREFIX}/bin\" \"\\\${NVSHMEM_BIN_DIR}\" _file_contents \"\${_file_contents}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/include\" \"\\\${NVSHMEM_INCLUDE_DIR}\" _file_contents \"\${_file_contents}\")
    string(REPLACE \"\\\${_IMPORT_PREFIX}/lib\" \"\\\${NVSHMEM_LIB_DIR}\" _file_contents \"\${_file_contents}\")
    file(WRITE \"\${targets_file}\" \"\${_file_contents}\")
  endforeach()
")


# This makes sure the device config file is included as well
install(CODE "
  # Patch NVSHMEM target files
  file(GLOB_RECURSE targets_files \"${CMAKE_BINARY_DIR}/lib/cmake/NVSHMEM*Targets*.cmake\")
  foreach(targets_file IN LISTS targets_files)
    message(STATUS \"Processing targets_file=\${targets_file}\")
    file(READ \"\${targets_file}\" _file_contents)
    string(REPLACE \"\\\${_IMPORT_PREFIX}/bin\" \"\\\${NVSHMEM_BIN_DIR}\" _file_contents \"\${_file_contents}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/include\" \"\\\${NVSHMEM_INCLUDE_DIR}\" _file_contents \"\${_file_contents}\")
    string(REPLACE \"\\\${_IMPORT_PREFIX}/lib\" \"\\\${NVSHMEM_LIB_DIR}\" _file_contents \"\${_file_contents}\")
    file(WRITE \"\${targets_file}\" \"\${_file_contents}\")
  endforeach()
")

install(
  EXPORT NVSHMEMTargets
  NAMESPACE nvshmem::
  DESTINATION lib/cmake/nvshmem/
  COMPONENT Export
)

# ${CMAKE_CURRENT_BINARY_DIR}/NVSHMEMConfig.cmake
install(FILES ${CMAKE_BINARY_DIR}/NVSHMEMVersion.cmake
        DESTINATION lib/cmake/nvshmem
)

install(FILES ${CMAKE_BINARY_DIR}/NVSHMEMConfig.cmake
        DESTINATION lib/cmake/nvshmem
	)

export(
  EXPORT NVSHMEMTargets
  NAMESPACE nvshmem::
  FILE "NVSHMEMTargets.cmake"
)

install(
	TARGETS nvshmem_host nvshmem-info ${ACTIVE_BOOTSTRAPS} ${ACTIVE_TRANSPORTS}
  EXPORT NVSHMEMTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a
  DESTINATION lib/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(
  FILES ${CMAKE_BINARY_DIR}/lib/cmake/nvshmem/NVSHMEMDeviceTargets.cmake
  DESTINATION lib/cmake/nvshmem/
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ WORLD_READ
)

install(
  FILES ${CMAKE_BINARY_DIR}/lib/cmake/nvshmem/NVSHMEMDeviceTargets-$<LOWER_CASE:${CMAKE_BUILD_TYPE}>.cmake
  DESTINATION lib/cmake/nvshmem/
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ WORLD_READ
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
  DESTINATION lib
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
  DESTINATION bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/share/src/
  DESTINATION share/src/
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include/
  PATTERN "modules" EXCLUDE
)
if(NVSHMEM_GPUNETIO_SUPPORT)
  install(
    DIRECTORY ${GPUNETIO_INCLUDE}/
    DESTINATION include/gpunetio
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.cuh"
  )
endif()
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/
  DESTINATION share/src/examples/
  PATTERN "examples_install" EXCLUDE
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../perftest/
  DESTINATION share/src/perftest/
  PATTERN "perftest_install" EXCLUDE
  PATTERN "perftestCommon.py" EXCLUDE
  PATTERN "perftest-ib.list" EXCLUDE
  PATTERN "perftest-p2p-nvlink.list" EXCLUDE
  PATTERN "perftest-p2p-pcie.list" EXCLUDE
  PATTERN "perftestRunner.py" EXCLUDE
  PATTERN "perfTestRunnerSlurm.py" EXCLUDE
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../changelog
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../License.txt
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../version.txt
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
  DESTINATION ./
)

include(CMakePackageConfigHelpers)

if(NVSHMEM_BUILD_TGZ_PACKAGE)
  list(APPEND CPACK_GENERATOR "TGZ")
endif()

if(NVSHMEM_BUILD_TXZ_PACKAGE)
  list(APPEND CPACK_GENERATOR "TXZ")
endif()

if(NVSHMEM_BUILD_RPM_PACKAGE)
  list(APPEND CPACK_GENERATOR "RPM")
endif()

if(NVSHMEM_BUILD_DEB_PACKAGE)
  list(APPEND CPACK_GENERATOR "DEB")
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(PACKAGE_SUFFIX "x86_64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(PACKAGE_SUFFIX "ppc64le")
else()
  set(PACKAGE_SUFFIX "aarch64")
endif()

set(CPACK_SOURCE_INSTALLED_DIRECTORIES "${CMAKE_SOURCE_DIR}/;/")


set(CPACK_SOURCE_IGNORE_FILES
"/install/;\
/build/;\
/.vscode;\
/include_gdrcopy;\
/include_nccl;\
/.clang-format;\
/.gitignore;\
/.git/;\
/.gitlab-ci.yml;\
/bind.sh;\
/README;\
/testcommon.py;\
/examples/examples_install/;\
/include/non_abi/nvshmem_build_options.h$;\
/include/non_abi/nvshmem_version.h$;\
/nvshmem4py/cmake/addCybind.cmake;\
/nvshmem4py/cmake/addNumbast.cmake;\
/nvshmem4py/build_assets/;\
/perftest/perftest_install/;\
/test/test_install/;
/scripts/cleanup_ignore_files.sh
/scripts/create_imex.sh;\
/scripts/dependency_installer.sh;\
/scripts/include_checker.sh;\
/scripts/iwyu_c_cpp_check.sh;\
/scripts/iwyu_header_cuda_check.sh;\
/scripts/iwyu_output_parser.py;\
/scripts/iwyu_test_functionality.sh;\
/scripts/testrunner.pl;\
/src/build/;")

if(NOT NVSHMEM_INSTALL_FUNCTIONAL_TESTS)
  set(CPACK_SOURCE_IGNORE_FILES ${CPACK_SOURCE_IGNORE_FILES}"/test/;/docker/;/nvshmem4py/test/;")
endif()


set(CPACK_PACKAGE_NAME "libnvshmem")
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://developer.nvidia.com/nvshmem")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../pkg/nvshmem_package_description.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NVSHMEM - An implementation of OpenSHMEM for NVIDIA GPUs.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "libnvshmem_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}+cuda${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}_${PACKAGE_SUFFIX}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "nvshmem_src_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../License.txt")
set(CPACK_PACKAGE_CONTACT "nvshmem@nvidia.com")

# RPM-Specific Configurations
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS")
set(CPACK_RPM_PACKAGE_AUTOREQ 0)
set(CPACK_RPM_PACKAGE_REQUIRES "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PRE "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PREUN "/sbin/ldconfig")

include(CPack)

add_custom_target(package_src_target
    COMMAND "${CMAKE_COMMAND}" --build . --target package_source
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Package Build"
    DEPENDS build_bindings_cybind build_bindings_numbast build_bindings_cute
    VERBATIM
)
