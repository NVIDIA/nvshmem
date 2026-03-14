# buildLibsWheel.cmake
#
# Provides the BuildLibsWheel() function used to assemble and build a
# pure-data Python wheel containing the NVSHMEM C libraries.
#
# The wheel layout mirrors the externally-published nvidia-nvshmem-cu<N> libs
# wheels:
#
#   nvidia/nvshmem/lib/libnvshmem_host.so*
#   nvidia/nvshmem/lib/libnvshmem_bootstrap_*.so*
#   nvidia/nvshmem/lib/libnvshmem_transport_*.so*
#   nvidia/nvshmem/lib/libnvshmem_device.a
#
# Library files are staged into a build-tree directory before the wheel is
# built so the source tree is never polluted with compiled artefacts.

function(BuildLibsWheel WHEEL_TARGET LIBS_SOURCE_DIR CUDA_MAJOR)

    # Staging area inside the build tree; the wheel is built from here.
    set(STAGING_DIR "${CMAKE_BINARY_DIR}/python_libs_wheel_staging")
    set(LIB_STAGING_DIR "${STAGING_DIR}/nvidia/nvshmem/lib")
    set(BUILD_DIR "${CMAKE_BINARY_DIR}/dist")

    # Lightweight venv used only for the `build` package (no C compilation).
    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv_libs_wheel")
    set(VENV_PYTHON "${VENV_DIR}/bin/python3")

    # ------------------------------------------------------------------
    # Configure pyproject.toml from the template into the staging area.
    # PROJECT_VERSION comes from the top-level project() call (e.g. 3.6.3.1).
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

    # ------------------------------------------------------------------
    # Venv setup target (created once per build tree).
    # ------------------------------------------------------------------
    if(NOT TARGET make_venv_libs_wheel)
        add_custom_target(make_venv_libs_wheel
            COMMAND ${Python3_EXECUTABLE} -m venv "${VENV_DIR}"
            COMMAND "${VENV_PYTHON}" -m pip install --upgrade pip
            COMMAND "${VENV_PYTHON}" -m pip install build setuptools wheel
            COMMENT "Creating Python venv for libs wheel build"
            USES_TERMINAL
        )
    endif()

    # ------------------------------------------------------------------
    # Main wheel build target.
    #
    # Steps:
    #   1. Create the staging lib directory.
    #   2. Copy shared library files (preserving symlinks) from the build tree.
    #   3. Copy static library files.
    #   4. Invoke `python -m build --wheel` from the staging directory.
    #   5. The finished wheel lands in ${CMAKE_BINARY_DIR}/dist/.
    # ------------------------------------------------------------------
    add_custom_target(
        ${WHEEL_TARGET}
        # Stage library files
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LIB_STAGING_DIR}"
        COMMAND bash -c "cp -P \"${LIBS_SOURCE_DIR}/\"*.so* \"${LIB_STAGING_DIR}/\" 2>/dev/null; true"
        COMMAND bash -c "cp \"${LIBS_SOURCE_DIR}/\"*.a \"${LIB_STAGING_DIR}/\" 2>/dev/null; true"
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

    # The wheel depends on the NVSHMEM libraries being compiled first.
    if(TARGET nvshmem_host)
        add_dependencies(${WHEEL_TARGET} nvshmem_host)
    endif()

    # Ensure the venv exists before attempting to build.
    if(NOT EXISTS "${VENV_PYTHON}")
        add_dependencies(${WHEEL_TARGET} make_venv_libs_wheel)
    endif()

endfunction()
