# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select the default C++ / CUDA standard used to build NVSHMEM. The default
# is C++17. Users may override via the NVSHMEM_CXX_STANDARD environment
# variable or -DCMAKE_CXX_STANDARD=XX on the CMake command line. The C++17
# minimum is enforced per target via target_compile_features(cxx_std_17) in
# nvshmem_library_set_base_config().

if(DEFINED ENV{NVSHMEM_CXX_STANDARD})
  set(NVSHMEM_CXX_STANDARD_DEFAULT $ENV{NVSHMEM_CXX_STANDARD})
else()
  set(NVSHMEM_CXX_STANDARD_DEFAULT 17)
endif()
set(CMAKE_CXX_STANDARD ${NVSHMEM_CXX_STANDARD_DEFAULT} CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED On)
set(CMAKE_CUDA_EXTENSIONS Off)
