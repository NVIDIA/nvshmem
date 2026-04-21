add_library(nvshmem_bootstrap_common STATIC bootstrap_util.cpp)
target_compile_features(nvshmem_bootstrap_common PUBLIC cxx_std_17)

set_target_properties(nvshmem_bootstrap_common PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(nvshmem_bootstrap_common INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(nvshmem_bootstrap_common PRIVATE
                           ${CMAKE_SOURCE_DIR}/include
)
