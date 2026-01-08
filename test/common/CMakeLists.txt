add_library(nvshmem_test_helper STATIC data_check.cpp ring_alltoall.cpp
                                       test_teams.cpp utils.cu perf_utils.cu
                                       test-simple-pmi/test_simple_pmi.cpp test-simple-pmi/test_simple_pmiutil.cpp)
add_library(nvshmem_test_helper_no_device INTERFACE)

add_library(nvshmem_unit_test_helper INTERFACE)

add_library(nvshmem_gtest_unit_test_helper INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(nvshmem_gtest_unit_test_helper INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    target_link_libraries(nvshmem_gtest_unit_test_helper INTERFACE clang_rt.profile)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Add GCC-specific coverage flags
    target_compile_options(nvshmem_gtest_unit_test_helper INTERFACE -fprofile-arcs -ftest-coverage)
    target_link_libraries(nvshmem_gtest_unit_test_helper INTERFACE gcov)
endif()

set_target_properties(nvshmem_test_helper PROPERTIES
POSITION_INDEPENDENT_CODE ON
CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(nvshmem_test_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem_test_helper_no_device PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem_unit_test_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem_gtest_unit_test_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvshmem_test_helper
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nvshmem_test_helper_no_device
			   INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nvshmem_unit_test_helper
			   INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/include
			   ${CMAKE_SOURCE_DIR}/../src/include)
target_include_directories(nvshmem_gtest_unit_test_helper
         INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
         ${CMAKE_CURRENT_SOURCE_DIR} 
         ${CMAKE_SOURCE_DIR}/src/include/
         ${CMAKE_SOURCE_DIR}/src/include/host/env/
         ${CMAKE_SOURCE_DIR}/src/include/internal/host/
         ${CMAKE_SOURCE_DIR}/../src/include/
         ${CMAKE_SOURCE_DIR}/../src/include/host/env/
         ${CMAKE_SOURCE_DIR}/../src/include/internal/host/
         ${NVSHMEM_PREFIX}/include/
         )
message(PROJECT_SOURCE_DIR="${CUDA_INCLUDE_DIRS}")
find_package(CUDAToolkit REQUIRED COMPONENTS nvJitLink cudart cuda_driver nvrtc)
if(NOT TARGET CUDA::nvJitLink)
  find_library(NVJITLINK_LIB
    NAMES nvJitLink
    PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_ROOT}/lib64"
    NO_DEFAULT_PATH)
  if(NOT NVJITLINK_LIB)
    message(FATAL_ERROR "libnvJitLink not found under CUDAToolkit_ROOT='${CUDAToolkit_ROOT}'")
  endif()
  add_library(CUDA::nvJitLink UNKNOWN IMPORTED)
  set_target_properties(CUDA::nvJitLink PROPERTIES
    IMPORTED_LOCATION "${NVJITLINK_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
endif()
target_link_libraries(nvshmem_test_helper PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::nvJitLink)
target_link_libraries(nvshmem_test_helper_no_device INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::nvml)
target_link_libraries(nvshmem_unit_test_helper INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::nvml)
target_link_libraries(nvshmem_gtest_unit_test_helper INTERFACE gtest gmock pthread gcov)
target_link_libraries(nvshmem_test_helper PUBLIC nvshmem_host nvshmem_device)
target_link_libraries(nvshmem_test_helper_no_device INTERFACE nvshmem_host)
target_link_libraries(nvshmem_unit_test_helper INTERFACE nvshmem_host ${CMAKE_DL_LIBS})

if(NVSHMEM_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_test_helper PUBLIC shmem)
  target_compile_definitions(nvshmem_test_helper PUBLIC NVSHMEMTEST_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_test_helper_no_device INTERFACE shmem)
  target_compile_definitions(nvshmem_test_helper_no_device INTERFACE NVSHMEMTEST_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_unit_test_helper INTERFACE shmem)
  target_compile_definitions(nvshmem_unit_test_helper INTERFACE NVSHMEMTEST_SHMEM_SUPPORT)
endif()

if(NVSHMEM_MPI_SUPPORT)
   target_link_libraries(nvshmem_test_helper PUBLIC MPI::MPI_CXX)
   target_compile_definitions(nvshmem_test_helper PUBLIC NVSHMEMTEST_MPI_SUPPORT)
   target_link_libraries(nvshmem_test_helper_no_device INTERFACE MPI::MPI_CXX)
   target_compile_definitions(nvshmem_test_helper_no_device INTERFACE NVSHMEMTEST_MPI_SUPPORT)
   target_link_libraries(nvshmem_unit_test_helper INTERFACE MPI::MPI_CXX)
   target_compile_definitions(nvshmem_unit_test_helper INTERFACE NVSHMEMTEST_MPI_SUPPORT)
endif()

target_compile_options(nvshmem_test_helper
PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
$<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
)
