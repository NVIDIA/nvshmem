find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(NVSHMEM_CXXFILT NAMES c++filt llvm-cxxfilt REQUIRED)

if(SubBuild)
  set(NVSHMEM_HOST_PUBLIC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/include")
else()
  set(NVSHMEM_HOST_PUBLIC_INCLUDE_DIR "${NVSHMEM_PREFIX}/include")
endif()

file(GLOB_RECURSE NVSHMEM_HOST_PUBLIC_HEADERS CONFIGURE_DEPENDS
     "${NVSHMEM_HOST_PUBLIC_INCLUDE_DIR}/*.h"
     "${NVSHMEM_HOST_PUBLIC_INCLUDE_DIR}/*.hpp"
     "${NVSHMEM_HOST_PUBLIC_INCLUDE_DIR}/*.cuh")

set(NVSHMEM_HOST_C_API_GENERATED_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/host_public_c_abi_generated.cu")

# The generator references every public header API that is dynamically exported
# by libnvshmem_host.so, and rejects public definitions with C++ linkage.
add_custom_command(
  OUTPUT "${NVSHMEM_HOST_C_API_GENERATED_SOURCE}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_host_c_api_abi.py"
          --nvcc "${CMAKE_CUDA_COMPILER}"
          --include-dir "${NVSHMEM_HOST_PUBLIC_INCLUDE_DIR}"
          --input "${CMAKE_CURRENT_SOURCE_DIR}/host_public_c_abi_input.cu"
          --output "${NVSHMEM_HOST_C_API_GENERATED_SOURCE}"
          --cxx-standard "${CMAKE_CXX_STANDARD}"
          --library "$<TARGET_FILE:nvshmem_host>"
          --device-library "$<TARGET_FILE:nvshmem_device>"
          --nm "${CMAKE_NM}"
          --cxxfilt "${NVSHMEM_CXXFILT}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_host_c_api_abi.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/host_public_c_abi_input.cu"
          ${NVSHMEM_HOST_PUBLIC_HEADERS}
          nvshmem_host
          nvshmem_device
  COMMENT "Generating host public C ABI link test"
  VERBATIM)

add_custom_target(generate_host_public_c_abi
                  DEPENDS "${NVSHMEM_HOST_C_API_GENERATED_SOURCE}")

nvshmem_add_test_no_device(host_public_c_abi.cu)
add_dependencies(host_host_public_c_abi generate_host_public_c_abi)
target_include_directories(host_host_public_c_abi
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
