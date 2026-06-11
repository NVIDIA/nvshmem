find_path(NVSHMEM_TEST_NCCL_DEVICE_INCLUDE_DIR
        NAMES nccl_device.h
        HINTS ${NCCL_HOME} $ENV{NCCL_HOME} ${NCCL_INCLUDE}
        PATH_SUFFIXES include
)

find_library(NVSHMEM_TEST_NCCL_RUNTIME_LIBRARY
        NAMES nccl
        HINTS ${NCCL_HOME} $ENV{NCCL_HOME}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NVSHMEM_TEST_NCCL_DEVICE_INCLUDE_DIR AND NVSHMEM_TEST_NCCL_RUNTIME_LIBRARY)
  nvshmem_add_test(interop_nccl.cu)

  target_include_directories(device_interop_nccl BEFORE PRIVATE
          ${NVSHMEM_TEST_NCCL_DEVICE_INCLUDE_DIR}
  )
  target_compile_options(device_interop_nccl PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  )

  target_link_libraries(device_interop_nccl ${NVSHMEM_TEST_NCCL_RUNTIME_LIBRARY})

else()
  message(STATUS
          "Skipping device_interop_nccl: NCCL runtime library and nccl_device.h were not both found")
endif()
