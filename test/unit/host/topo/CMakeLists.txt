if(GTest_FOUND)
  set(TOPO_SOURCE "${NVSHMEM_TEST_TLD}/../src/host/topo/topo.cpp")
  set_source_files_properties(${TOPO_SOURCE} PROPERTIES
                              COMPILE_DEFINITIONS
                              "opendir=nvshmemi_test_opendir;realpath=nvshmemi_test_realpath")

  nvshmem_add_gtest_unit_test_prefix(topo_policy_unit_tests.cpp "host_"
                                     "${TOPO_SOURCE}")

  target_include_directories(host_topo_policy_unit_tests PRIVATE
    ${NVSHMEM_TEST_TLD}/../src/host/topo
    ${NVSHMEM_TEST_TLD}/../src/include
    ${NVSHMEM_TEST_TLD}/../src/include/host/env
  )
endif()
