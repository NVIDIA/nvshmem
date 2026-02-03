set(NVSHMEM_UNIT_TEST_PREFIX "host_")
macro(nvshmem_add_unit_test SOURCE)
  nvshmem_add_unit_test_prefix(${SOURCE} ${NVSHMEM_UNIT_TEST_PREFIX})
endmacro()

add_subdirectory(bootstrap)
add_subdirectory(env)
