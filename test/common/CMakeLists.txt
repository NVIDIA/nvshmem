add_library(nvshmem_test_helper STATIC data_check.cpp ring_alltoall.cpp
                                       test_teams.cpp utils.cu perf_utils.cu
                                       test-simple-pmi/test_simple_pmi.cpp test-simple-pmi/test_simple_pmiutil.cpp)
add_library(nvshmem_test_helper_no_device INTERFACE)

set_target_properties(nvshmem_test_helper PROPERTIES
POSITION_INDEPENDENT_CODE ON
CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(nvshmem_test_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(nvshmem_test_helper_no_device PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvshmem_test_helper
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nvshmem_test_helper_no_device
			   INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
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
target_link_libraries(nvshmem_test_helper PUBLIC nvshmem_host nvshmem_device)
target_link_libraries(nvshmem_test_helper_no_device INTERFACE nvshmem_host)

if(NVSHMEM_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_test_helper PUBLIC shmem)
  target_compile_definitions(nvshmem_test_helper PUBLIC NVSHMEMTEST_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_test_helper_no_device INTERFACE shmem)
  target_compile_definitions(nvshmem_test_helper_no_device INTERFACE NVSHMEMTEST_SHMEM_SUPPORT)
endif()

if(NVSHMEM_MPI_SUPPORT)
   target_link_libraries(nvshmem_test_helper PUBLIC MPI::MPI_CXX)
   target_compile_definitions(nvshmem_test_helper PUBLIC NVSHMEMTEST_MPI_SUPPORT)
   target_link_libraries(nvshmem_test_helper_no_device INTERFACE MPI::MPI_CXX)
   target_compile_definitions(nvshmem_test_helper_no_device INTERFACE NVSHMEMTEST_MPI_SUPPORT)
endif()

target_compile_options(nvshmem_test_helper
PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
$<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
)
