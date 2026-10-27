# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/ensureNumbast.cmake")

# Build the NVSHMEM cuTile TileLib, which cuTile kernels call through
# cuda.tile_preview.foreign_call.  The generator reads the device headers and
# writes the Tile IR interface, the CUDA shims, and the linkage provider's
# catalogs into the build tree, and the Python bindings into the package.  The
# shims are compiled against NVSHMEM's declarations only; at kernel compile
# time the provider links them against NVSHMEM's device LTOIR library.  The
# TileLib is a device library too, installed and shipped beside that one.
function(generateCutileTileLib)
    if(CUDAToolkit_VERSION VERSION_LESS "13.0")
        message(FATAL_ERROR "The cuTile TileLib requires CUDA 13 or newer.")
    endif()
    if(NOT TARGET libnvshmem_device_ltoir)
        message(FATAL_ERROR "The cuTile TileLib requires NVSHMEM_BUILD_LTOIR_LIBRARY=ON.")
    endif()
    foreach(_tool IN ITEMS NVSHMEM4PY_CUTILE_TILEIRAS NVSHMEM4PY_CUTILE_TRANSLATOR)
        if(NOT EXISTS "${${_tool}}")
            message(FATAL_ERROR "Set ${_tool} to a CUDA 13.5 executable.")
        endif()
    endforeach()
    # The provider implements this C ABI, which ships with tileiras.
    get_filename_component(_tileiras_dir "${NVSHMEM4PY_CUTILE_TILEIRAS}" DIRECTORY)
    find_path(NVSHMEM4PY_CUTILE_LINKAGE_PROVIDER_INCLUDE_DIR
        NAMES tile_ir/linkage_provider/LinkageProviderAPI.h
        HINTS "${CUDA_HOME}/include" "${_tileiras_dir}/../include"
        DOC "Directory containing tile_ir/linkage_provider/LinkageProviderAPI.h")
    if(NOT NVSHMEM4PY_CUTILE_LINKAGE_PROVIDER_INCLUDE_DIR)
        message(FATAL_ERROR
            "Set NVSHMEM4PY_CUTILE_LINKAGE_PROVIDER_INCLUDE_DIR to the directory containing "
            "tile_ir/linkage_provider/LinkageProviderAPI.h.")
    endif()

    set(VENV_PYTHON_EXECUTABLE "${CMAKE_BINARY_DIR}/externals/venv/bin/python3")
    nvshmem_ensure_numbast(VERSION "0.9.0")

    set(CUTILE_TREE_DIR "${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cutile")
    set(CUTILE_WORK_DIR "${CMAKE_BINARY_DIR}/externals/cutile")
    set(CUTILE_PYTHON_DIR "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/tile")
    set(CUTILE_TILELIB "${CMAKE_BINARY_DIR}/src/lib/libnvshmem_device.tilelib")

    # The shims cover the architectures of the device library they link
    # against.  The generator reads the headers as the oldest of them sees
    # them, so every generated function exists on every architecture.
    get_target_property(CUTILE_ARCHS libnvshmem_device_ltoir NVSHMEM_LTOIR_ARCHS)
    list(SORT CUTILE_ARCHS COMPARE NATURAL)
    list(GET CUTILE_ARCHS 0 CUTILE_HEADER_ARCH)

    set(CUTILE_INTERFACE "${CUTILE_WORK_DIR}/nvshmem_tile_interface.mlir")
    set(CUTILE_PROVIDER_CATALOG "${CUTILE_WORK_DIR}/nvshmem_tile_provider.json")
    set(CUTILE_SHIM_SOURCE "${CUTILE_WORK_DIR}/nvshmem_tile_ffi.cu")
    set(CUTILE_OUTPUTS
        "${CUTILE_INTERFACE}"
        "${CUTILE_PROVIDER_CATALOG}"
        "${CUTILE_SHIM_SOURCE}"
        "${CUTILE_PYTHON_DIR}/generated.py"
        "${CUTILE_PYTHON_DIR}/direct.py"
        "${CUTILE_PYTHON_DIR}/rma.py"
        "${CUTILE_PYTHON_DIR}/collective.py"
        "${CUTILE_PYTHON_DIR}/amo.py"
        "${CUTILE_PYTHON_DIR}/mem.py")
    file(GLOB CUTILE_GENERATOR_INPUTS CONFIGURE_DEPENDS
        "${CUTILE_TREE_DIR}/*.py"
        "${CUTILE_TREE_DIR}/templates/core/device/tile/*.j2")
    file(GLOB_RECURSE CUTILE_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/include/*.h"
        "${CMAKE_SOURCE_DIR}/src/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/include/*.cuh")
    add_custom_command(
        OUTPUT ${CUTILE_OUTPUTS}
        COMMAND ${CMAKE_COMMAND} -E env "CUDA_HOME=${CUDA_HOME}"
                ${VENV_PYTHON_EXECUTABLE} "${CUTILE_TREE_DIR}/generate_tilelib.py" sources
                --output-dir "${CUTILE_WORK_DIR}"
                --python-dir "${CUTILE_PYTHON_DIR}"
                --include-dir "${CMAKE_SOURCE_DIR}/src/include"
                --arch "${CUTILE_HEADER_ARCH}"
        DEPENDS pip_install_numbast ${CUTILE_GENERATOR_INPUTS} ${CUTILE_HEADERS}
        COMMENT "Generating the NVSHMEM cuTile TileLib sources"
        VERBATIM)

    # One LTOIR image of the shims per architecture, bundled like the device
    # library.  With NVSHMEM_HOSTLIB_ONLY every NVSHMEM call stays unresolved
    # until the provider links the device library.
    set(CUTILE_SHIM_IMAGES "")
    set(CUTILE_SHIM_IMAGE_ARGS "")
    foreach(_arch IN LISTS CUTILE_ARCHS)
        string(REPLACE "sm_" "" _arch_number "${_arch}")
        set(_image "${CUTILE_WORK_DIR}/nvshmem_tile_shims_${_arch}.ltoir")
        add_custom_command(
            OUTPUT "${_image}"
            COMMAND cuda::nvcc -MMD -MF "${_image}.d" -MT "${_image}" -std=c++17 -arch=${_arch} -dlto --ltoir
                    -I "${CMAKE_SOURCE_DIR}/src/include"
                    -DNVSHMEM_HOSTLIB_ONLY
                    -o "${_image}" "${CUTILE_SHIM_SOURCE}"
            DEPENDS "${CUTILE_SHIM_SOURCE}"
            DEPFILE "${_image}.d"
            COMMENT "Compiling the NVSHMEM cuTile TileLib shims for ${_arch}"
            VERBATIM)
        list(APPEND CUTILE_SHIM_IMAGES "${_image}")
        list(APPEND CUTILE_SHIM_IMAGE_ARGS "--image3=kind=nvvm,sm=${_arch_number},file=${_image}")
    endforeach()
    set(CUTILE_SHIMS "${CUTILE_WORK_DIR}/nvshmem_tile_shims.fatbin")
    add_custom_command(
        OUTPUT "${CUTILE_SHIMS}"
        COMMAND cuda::fatbinary --64 "--create=${CUTILE_SHIMS}" ${CUTILE_SHIM_IMAGE_ARGS}
        DEPENDS ${CUTILE_SHIM_IMAGES}
        COMMENT "Bundling the NVSHMEM cuTile TileLib shims"
        VERBATIM)

    # A Tile operand's extent is part of its type, so the compiler names each
    # specialization separately and a static TileLib cannot define them.  The
    # linkage provider serves one catalog per declaration and forwards every
    # specialization to its shim.
    set(CUTILE_PROVIDER_DIR "${CUTILE_WORK_DIR}/provider")
    set(CUTILE_PROVIDER_SOURCE "${CUTILE_PROVIDER_DIR}/nvshmem_tile_provider.c")
    set(CUTILE_PROVIDER_BLOB "${CUTILE_PROVIDER_DIR}/nvshmem_tile_provider_catalogs.bin")
    add_custom_command(
        OUTPUT "${CUTILE_PROVIDER_SOURCE}" "${CUTILE_PROVIDER_BLOB}"
        COMMAND ${VENV_PYTHON_EXECUTABLE} "${CUTILE_TREE_DIR}/generate_tilelib.py" provider
                --catalog "${CUTILE_PROVIDER_CATALOG}"
                --translator "${NVSHMEM4PY_CUTILE_TRANSLATOR}"
                --shims "${CUTILE_SHIMS}"
                --output-dir "${CUTILE_PROVIDER_DIR}"
        DEPENDS "${CUTILE_PROVIDER_CATALOG}" "${CUTILE_SHIMS}"
                "${CUTILE_TREE_DIR}/templates/tilelib_provider.c.j2"
        COMMENT "Generating the NVSHMEM cuTile TileLib linkage provider"
        VERBATIM)
    # The provider source embeds the catalogs and shims with .incbin.
    set_source_files_properties("${CUTILE_PROVIDER_SOURCE}" PROPERTIES
        OBJECT_DEPENDS "${CUTILE_PROVIDER_BLOB};${CUTILE_SHIMS}")
    add_library(nvshmem_cutile_tilelib_provider SHARED "${CUTILE_PROVIDER_SOURCE}")
    target_include_directories(nvshmem_cutile_tilelib_provider PRIVATE
        "${NVSHMEM4PY_CUTILE_LINKAGE_PROVIDER_INCLUDE_DIR}")
    target_link_libraries(nvshmem_cutile_tilelib_provider PRIVATE ${CMAKE_DL_LIBS})
    # tileiras loads the provider, which is plain C and exports only its
    # Tile IR entry point.
    target_link_options(nvshmem_cutile_tilelib_provider PRIVATE
        "LINKER:--version-script=${CUTILE_TREE_DIR}/tilelib_provider.map")
    set_target_properties(nvshmem_cutile_tilelib_provider PROPERTIES
        OUTPUT_NAME nvshmem_cutile_provider
        C_STANDARD 11
        C_VISIBILITY_PRESET hidden
        LINK_DEPENDS "${CUTILE_TREE_DIR}/tilelib_provider.map"
        LIBRARY_OUTPUT_DIRECTORY "${CUTILE_PROVIDER_DIR}")

    set(CUTILE_INTERFACE_BYTECODE "${CUTILE_WORK_DIR}/nvshmem_tile_interface.tilebc")
    add_custom_command(
        OUTPUT "${CUTILE_INTERFACE_BYTECODE}"
        COMMAND "${NVSHMEM4PY_CUTILE_TRANSLATOR}" -mlir-to-cudatilebc -no-implicit-module
                --preview=simt -bytecode-version=13.5
                "${CUTILE_INTERFACE}" -o "${CUTILE_INTERFACE_BYTECODE}"
        DEPENDS "${CUTILE_INTERFACE}"
        COMMENT "Translating the NVSHMEM cuTile TileLib interface"
        VERBATIM)
    add_custom_command(
        OUTPUT "${CUTILE_TILELIB}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/src/lib"
        COMMAND "${NVSHMEM4PY_CUTILE_TILEIRAS}" --preview-create-tile-lib
                "--preview-interface=${CUTILE_INTERFACE_BYTECODE}"
                "--preview-implementation=$<TARGET_FILE:nvshmem_cutile_tilelib_provider>"
                -o "${CUTILE_TILELIB}"
        DEPENDS "${CUTILE_INTERFACE_BYTECODE}" nvshmem_cutile_tilelib_provider
        COMMENT "Packaging the NVSHMEM cuTile TileLib"
        VERBATIM)
    # The default build assembles the TileLib, which installs with the other
    # device libraries.
    add_custom_target(libnvshmem_device_tilelib ALL DEPENDS "${CUTILE_TILELIB}")
    # The libraries wheel stages this file only while this target exists.
    set_target_properties(libnvshmem_device_tilelib PROPERTIES NVSHMEM_TILELIB_FILE "${CUTILE_TILELIB}")
    install(FILES "${CUTILE_TILELIB}" DESTINATION lib)

    add_custom_target(build_bindings_cutile DEPENDS ${CUTILE_OUTPUTS})
    # The provider links the device library at kernel compile time.
    add_dependencies(build_bindings_cutile libnvshmem_device_tilelib libnvshmem_device_ltoir)
endfunction()
