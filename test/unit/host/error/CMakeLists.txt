if(GTest_FOUND)
  # gtest-based unit test
  nvshmem_add_gtest_unit_test_prefix(nvshmemx_status_string_unit_tests.cpp "host_" "")

  target_include_directories(host_nvshmemx_status_string_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
  )
endif()
