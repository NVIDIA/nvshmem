# stagePlugins.cmake — invoked at build time via cmake -P
#
# Copies optional library artefacts from the build tree into the wheel staging
# directory: bootstrap/transport plugins (globbed) and the device bitcode file
# (if present).  Using cmake -P avoids shell-escaping issues with globs.
#
# The SOVERSION symlinks (e.g. nvshmem_bootstrap_pmi.so.3) point to the
# full-version file (e.g. nvshmem_bootstrap_pmi.so.3.0.0).  We resolve the
# symlink and copy the real file under the SOVERSION name so the wheel
# contains regular files, not dangling symlinks.
#
# Expected -D variables:
#   SOURCE_DIR  — directory containing the built .so files (e.g. build/src/lib)
#   DEST_DIR    — wheel staging lib directory

# Stage the host shared library — resolve the SOVERSION symlink.
set(_host "${SOURCE_DIR}/libnvshmem_host.so.${HOST_SOVERSION}")
if(EXISTS "${_host}")
    get_filename_component(_real "${_host}" REALPATH)
    get_filename_component(_name "${_host}" NAME)
    file(COPY "${_real}" DESTINATION "${DEST_DIR}")
    get_filename_component(_real_name "${_real}" NAME)
    if(NOT "${_real_name}" STREQUAL "${_name}")
        file(RENAME "${DEST_DIR}/${_real_name}" "${DEST_DIR}/${_name}")
    endif()
endif()

# Stage the device static library.
if(EXISTS "${SOURCE_DIR}/libnvshmem_device.a")
    file(COPY "${SOURCE_DIR}/libnvshmem_device.a" DESTINATION "${DEST_DIR}")
endif()

# Stage bootstrap and transport plugins.
file(GLOB _plugins
    "${SOURCE_DIR}/nvshmem_bootstrap_*.so.3"
    "${SOURCE_DIR}/nvshmem_transport_*.so.5"
)
foreach(_f IN LISTS _plugins)
    # Resolve symlink to the real file.
    get_filename_component(_real "${_f}" REALPATH)
    get_filename_component(_name "${_f}" NAME)
    file(COPY "${_real}" DESTINATION "${DEST_DIR}")
    # Rename the full-version filename to the SOVERSION name.
    get_filename_component(_real_name "${_real}" NAME)
    if(NOT "${_real_name}" STREQUAL "${_name}")
        file(RENAME "${DEST_DIR}/${_real_name}" "${DEST_DIR}/${_name}")
    endif()
endforeach()

# Stage device bitcode if it was built.
if(EXISTS "${SOURCE_DIR}/libnvshmem_device.bc")
    file(COPY "${SOURCE_DIR}/libnvshmem_device.bc" DESTINATION "${DEST_DIR}")
endif()
