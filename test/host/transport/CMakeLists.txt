nvshmem_add_test_no_device(ibgda_q_counter.cpp)
set(NVSHMEM_Q_COUNTER_COMMON_DIR "${NVSHMEM_TEST_TLD}/../src/modules/transport/common")
target_include_directories(host_ibgda_q_counter PRIVATE "${NVSHMEM_Q_COUNTER_COMMON_DIR}")

# Layout/retry checks are always available. Query tests need headers, but no RDMA device.
option(NVSHMEM_TEST_MLX5_QUERY
       "Build mocked DCT query tests with the mlx5 development library"
       ${NVSHMEM_IBGDA_SUPPORT})
if(NVSHMEM_TEST_MLX5_QUERY)
  find_library(NVSHMEM_Q_COUNTER_MLX5_LIBRARY NAMES mlx5 REQUIRED)
  target_sources(host_ibgda_q_counter PRIVATE
                 "${NVSHMEM_Q_COUNTER_COMMON_DIR}/transport_mlx5_common.cpp")
  target_compile_definitions(host_ibgda_q_counter PRIVATE NVSHMEM_TEST_MLX5_QUERY)
  target_include_directories(host_ibgda_q_counter PRIVATE "${NVSHMEM_TEST_TLD}/../src/include")
  if(SubBuild)
    target_include_directories(host_ibgda_q_counter PRIVATE "${CMAKE_BINARY_DIR}/src/include")
  else()
    target_include_directories(host_ibgda_q_counter PRIVATE "${NVSHMEM_PREFIX}/include")
  endif()
  target_link_libraries(host_ibgda_q_counter ${NVSHMEM_Q_COUNTER_MLX5_LIBRARY} ${CMAKE_DL_LIBS})
endif()
