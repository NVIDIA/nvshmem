add_library(nvshmem_perftest_helper STATIC utils.cu)

set_target_properties(nvshmem_perftest_helper PROPERTIES
POSITION_INDEPENDENT_CODE ON
CXX_STANDARD_REQUIRED ON
CUDA_STANDARD_REQUIRED ON
CXX_STANDARD ${PERFTEST_CXX_STANDARD}
CUDA_STANDARD ${PERFTEST_CXX_STANDARD}
CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(nvshmem_perftest_helper
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvshmem_perftest_helper PUBLIC CUDA::cudart CUDA::cuda_driver)
if(NVSHMEM_TEST_STATIC_LIB)
target_link_libraries(nvshmem_perftest_helper PUBLIC nvshmem)
else()
target_link_libraries(nvshmem_perftest_helper PUBLIC nvshmem_host nvshmem_device)
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  target_compile_definitions(nvshmem_perftest_helper PUBLIC NVSHMEMTEST_SHMEM_SUPPORT)
  target_include_directories(nvshmem_perftest_helper PUBLIC SHMEM_INCLUDE)
endif()

if(NVSHMEM_MPI_SUPPORT)
   target_compile_definitions(nvshmem_perftest_helper PUBLIC NVSHMEMTEST_MPI_SUPPORT)
   target_include_directories(nvshmem_perftest_helper PUBLIC $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_DIRS}>)
endif()

target_compile_options(nvshmem_perftest_helper
PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
$<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
)
