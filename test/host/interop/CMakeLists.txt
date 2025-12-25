add_library(simplelib1 SHARED simplelib1.cu)
add_library(simplelib2 SHARED simplelib2.cu)
add_executable(app app.cu)
add_executable(app_multi_init app_multi_init.cu)

set_target_properties(simplelib1 simplelib2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(app simplelib1 simplelib2 PROPERTIES INSTALL_RPATH "$ORIGIN/../../../../lib" BUILD_WITH_INSTALL_RPATH TRUE)
target_link_options(simplelib1 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/simplelib1.sym")
target_link_options(simplelib2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/simplelib2.sym")

# Tile-granular APIs need C++ 17
set(TEST_CXX_STANDARD 17)
set(TEST_CXX_ARG c++17)
set_target_properties(simplelib1 PROPERTIES
CXX_STANDARD "${TEST_CXX_STANDARD}"
CUDA_STANDARD "${TEST_CXX_STANDARD}"
)
set_target_properties(simplelib2 PROPERTIES
CXX_STANDARD "${TEST_CXX_STANDARD}"
CUDA_STANDARD "${TEST_CXX_STANDARD}"
)
set_target_properties(app PROPERTIES
CXX_STANDARD "${TEST_CXX_STANDARD}"
CUDA_STANDARD "${TEST_CXX_STANDARD}"
)
set_target_properties(app_multi_init PROPERTIES
CXX_STANDARD "${TEST_CXX_STANDARD}"
CUDA_STANDARD "${TEST_CXX_STANDARD}"
)

target_include_directories(simplelib1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(simplelib2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(app_multi_init PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(simplelib1 PRIVATE nvshmem_test_helper)
target_link_libraries(simplelib2 PRIVATE nvshmem_test_helper)
target_link_libraries(app PRIVATE simplelib1 simplelib2 nvshmem_test_helper)
target_link_libraries(app_multi_init PRIVATE simplelib1 simplelib2 nvshmem_test_helper)

#
# Some toolchains/distros enable PIE by default for executables. NVCC-produced objects for these
# CUDA test apps may not be compiled with -fPIE, which then fails to link with:
#   relocation R_X86_64_32 ... can not be used when making a PIE object
# Disable PIE for these interop test executables to keep the build portable.
#
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(app PRIVATE -no-pie)
    target_link_options(app_multi_init PRIVATE -no-pie)
endif()

install(TARGETS simplelib1 LIBRARY DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/interop")
install(TARGETS simplelib2 LIBRARY DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/interop")
install(TARGETS app RUNTIME DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/interop")
install(TARGETS app_multi_init RUNTIME DESTINATION "${NVSHMEM_TEST_INSTALL_PREFIX}/host/interop")
