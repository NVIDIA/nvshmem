# GPUNetIO Open - vendored source build
#
# DOCA_HOME allows pointing at an external GPUNetIO Open. When empty (the default), 
# the vendored copy in this directory is used. Set via env var DOCA_HOME or
# cmake -DDOCA_HOME=/path.
if(NOT DOCA_HOME)
  set(DOCA_HOME ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(GPUNETIO_SOURCES
    ${DOCA_HOME}/src/doca_gpunetio.cpp
    ${DOCA_HOME}/src/doca_gpunetio_cuda_wrapper.cpp
    ${DOCA_HOME}/src/doca_gpunetio_gdrcopy.cpp
    ${DOCA_HOME}/src/doca_gpunetio_high_level.cpp
    ${DOCA_HOME}/src/doca_gpunetio_log.cpp
    ${DOCA_HOME}/src/doca_gpunetio_sdk_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_cq.cpp
    ${DOCA_HOME}/src/doca_verbs_cq_sdk_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_dev.cpp
    ${DOCA_HOME}/src/doca_verbs_dev_sdk_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_device_attr.cpp
    ${DOCA_HOME}/src/doca_verbs_ibv_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_mlx5dv_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_qp.cpp
    ${DOCA_HOME}/src/doca_verbs_qp_sdk_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_srq.cpp
    ${DOCA_HOME}/src/doca_verbs_uar.cpp
    ${DOCA_HOME}/src/doca_verbs_uar_sdk_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_umem.cpp
    ${DOCA_HOME}/src/doca_verbs_umem_sdk_wrapper.cpp
)

add_library(doca_gpunetio OBJECT ${GPUNETIO_SOURCES})
set_property(TARGET doca_gpunetio PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(doca_gpunetio
  PUBLIC ${DOCA_HOME}/include
  PRIVATE ${DOCA_HOME}/include/gpunetio
          ${DOCA_HOME}/src
)
target_compile_definitions(doca_gpunetio PRIVATE
  DOCA_VERBS_USE_NET_WRAPPER
  DOCA_GPUNETIO_USE_CUDA_WRAPPER
)
target_link_libraries(doca_gpunetio PUBLIC CUDA::cudart_static dl)
