if(GTest_FOUND)
  # gtest-based unit test
  nvshmem_add_gtest_unit_test_prefix(conf_file_unit_tests.cpp "host_" "")

  # Needed because `bootstrap_host_transport/env_defs_internal.h` includes `env_defs.h`
  # with a relative include.
  target_include_directories(host_conf_file_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/host/env
  )
endif()


