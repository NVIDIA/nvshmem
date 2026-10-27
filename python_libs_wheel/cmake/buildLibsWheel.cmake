# buildLibsWheel.cmake
#
# Provides the BuildLibsWheel() function used to assemble and build a
# pure-data Python wheel containing the NVSHMEM C libraries.
#
# The wheel layout mirrors the externally-published nvidia-nvshmem-cu<N> libs
# wheels:
#
#   nvidia/nvshmem/include/nvshmem.h
#   nvidia/nvshmem/include/...
#   nvidia/nvshmem/lib/libnvshmem_host.so.3
#   nvidia/nvshmem/lib/libnvshmem_device.a
#   nvidia/nvshmem/lib/libnvshmem_device.bc
#   nvidia/nvshmem/lib/libnvshmem_device.ltoir.fatbin
#   nvidia/nvshmem/lib/libnvshmem_device.tilelib (with the cuTile TileLib)
#   nvidia/nvshmem/lib/nvshmem_bootstrap_*.so.3
#   nvidia/nvshmem/lib/nvshmem_transport_*.so.5
#
# Library files are staged into a build-tree directory before the wheel is
# built so the source tree is never polluted with compiled artefacts.

function(BuildLibsWheel WHEEL_TARGET LIBS_SOURCE_DIR CUDA_MAJOR)

    # Staging area inside the build tree; the wheel is built from here.
    set(STAGING_DIR "${CMAKE_BINARY_DIR}/python_libs_wheel_staging")
    set(LIB_STAGING_DIR "${STAGING_DIR}/nvidia/nvshmem/lib")
    set(INCLUDE_STAGING_DIR "${STAGING_DIR}/nvidia/nvshmem/include")
    set(BUILD_DIR "${CMAKE_BINARY_DIR}/dist")

    # Lightweight venv used only for the `build` package (no C compilation).
    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv_libs_wheel")
    set(VENV_PYTHON "${VENV_DIR}/bin/python3")

    # ------------------------------------------------------------------
    # Configure pyproject.toml from the template into the staging area.
    # NVSHMEM_ARTIFACT_VERSION is the PEP 440 package version (e.g. 3.7.0rc1).
    # ------------------------------------------------------------------
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in"
        "${STAGING_DIR}/pyproject.toml"
        @ONLY
    )

    # Copy the namespace package stubs into the staging area at configure time.
    # The nvidia/ directory intentionally has no __init__.py (implicit namespace
    # package, standard for the nvidia:: namespace).
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/nvshmem/__init__.py"
        "${STAGING_DIR}/nvidia/nvshmem/__init__.py"
        COPYONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/nvshmem/lib/__init__.py"
        "${STAGING_DIR}/nvidia/nvshmem/lib/__init__.py"
        COPYONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/nvshmem/include/__init__.py"
        "${STAGING_DIR}/nvidia/nvshmem/include/__init__.py"
        COPYONLY
    )

    # ------------------------------------------------------------------
    # Venv setup target (created once per build tree).
    # ------------------------------------------------------------------
    if(NOT TARGET make_venv_libs_wheel)
        add_custom_target(make_venv_libs_wheel
            COMMAND ${Python3_EXECUTABLE} -m venv "${VENV_DIR}"
            COMMAND "${VENV_PYTHON}" -m pip install --upgrade pip
            COMMAND "${VENV_PYTHON}" -m pip install --upgrade build setuptools wheel
            COMMENT "Creating Python venv for libs wheel build"
            USES_TERMINAL
        )
    endif()

    # Source directory for headers (generated into the build tree by src/).
    set(INCLUDE_SOURCE_DIR "${CMAKE_BINARY_DIR}/src/include")

    # Stage, and build first, exactly the plugins this build enables, so a
    # plugin left in the build tree by an earlier configuration cannot ship.
    # cmake -P takes the list with "|" separators.
    get_property(_PLUGIN_TARGETS GLOBAL PROPERTY NVSHMEM_PLUGIN_TARGETS)
    set(_PLUGIN_FILES "")
    foreach(_PLUGIN IN LISTS _PLUGIN_TARGETS)
        list(APPEND _PLUGIN_FILES "$<TARGET_SONAME_FILE:${_PLUGIN}>")
    endforeach()
    list(JOIN _PLUGIN_FILES "|" _PLUGIN_FILES)

    # Stage the cuTile TileLib only if this configuration builds it, so one
    # left in the build tree by an earlier configuration cannot ship.
    set(_TILELIB_FILE "")
    if(TARGET libnvshmem_device_tilelib)
        get_target_property(_TILELIB_FILE libnvshmem_device_tilelib NVSHMEM_TILELIB_FILE)
    endif()

    # ------------------------------------------------------------------
    # Main wheel build target.
    #
    # Stages only the files that appear in the published wheel:
    #   - libnvshmem_host.so.<SOVERSION>  (not the unversioned .so or full .so.X.Y.Z)
    #   - libnvshmem_device.a
    #   - libnvshmem_device.bc            (if bitcode library was built)
    #   - nvshmem_bootstrap_*.so.<SOVERSION>
    #   - nvshmem_transport_*.so.<SOVERSION>
    #   - include/ headers
    # ------------------------------------------------------------------
    add_custom_target(
        ${WHEEL_TARGET}
        # Recreate the package payload on every invocation so a transport,
        # SOVERSION, or header removed by a reconfigure cannot survive in a
        # reused build tree.
        COMMAND ${CMAKE_COMMAND} -E rm -rf
                "${LIB_STAGING_DIR}"
                "${INCLUDE_STAGING_DIR}"
        # Stage library files — only SOVERSION symlinks, matching published wheel
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIB_STAGING_DIR}"
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${LIBS_SOURCE_DIR} -DDEST_DIR=${LIB_STAGING_DIR} -DHOST_SOVERSION=${PROJECT_VERSION_MAJOR} -DPLUGINS=${_PLUGIN_FILES} -DTILELIB=${_TILELIB_FILE} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stagePlugins.cmake"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/nvshmem/lib/__init__.py"
                "${LIB_STAGING_DIR}/__init__.py"
        # Stage headers
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${INCLUDE_SOURCE_DIR}" "${INCLUDE_STAGING_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/nvshmem/include/__init__.py"
                "${INCLUDE_STAGING_DIR}/__init__.py"
        # Remove internal headers not shipped in the published wheel
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${INCLUDE_STAGING_DIR}/modules"
        # setuptools creates these below the staging source.  Remove them so a
        # repeated wheel build cannot rediscover its previous build/ tree as a
        # nested namespace package.
        COMMAND ${CMAKE_COMMAND} -E rm -rf
                "${STAGING_DIR}/build"
                "${STAGING_DIR}/nvidia_nvshmem_cu${CUDA_MAJOR}.egg-info"
        # Build wheel
        COMMAND ${CMAKE_COMMAND} -E make_directory "${BUILD_DIR}"
        COMMAND "${VENV_PYTHON}" -m build --wheel --outdir "${BUILD_DIR}" --no-isolation
        COMMAND ${CMAKE_COMMAND} -E echo "Libs wheel placed in ${BUILD_DIR}"
        COMMAND ls "${BUILD_DIR}"
        WORKING_DIRECTORY "${STAGING_DIR}"
        COMMENT "Building NVSHMEM libs-only Python wheel for CUDA ${CUDA_MAJOR}..."
        USES_TERMINAL
        VERBATIM
    )

    # The wheel depends on every library it stages being compiled first;
    # the optional device libraries are guarded.
    set(_WHEEL_DEP_TARGETS
        nvshmem_host
        nvshmem_device_project
        libnvshmem_device_bitcode
        libnvshmem_device_ltoir
        libnvshmem_device_tilelib
        ${_PLUGIN_TARGETS}
    )
    foreach(_DEP IN LISTS _WHEEL_DEP_TARGETS)
        if(TARGET ${_DEP})
            add_dependencies(${WHEEL_TARGET} ${_DEP})
        endif()
    endforeach()

    # Ensure the venv exists before attempting to build.
    add_dependencies(${WHEEL_TARGET} make_venv_libs_wheel)

endfunction()
