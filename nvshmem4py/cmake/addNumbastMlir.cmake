# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(AddNumbastMlir VERSION)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")

    cmake_parse_arguments(PARSE_ARGV 0 ADDNUMBASTMLIR "" "VERSION" "")

    if(NOT DEFINED ADDNUMBASTMLIR_VERSION)
        message(FATAL_ERROR "VERSION not provided to AddNumbastMlir")
    endif()

    set(NUMBAST_CONFIG_VERSION "0.1.0")
    set(PACKAGE_NAME "numbast_mlir")
    set(WORKDIR "${CMAKE_BINARY_DIR}/externals/${PACKAGE_NAME}")
    set(ASSET_DIR "${WORKDIR}/build_assets")
    set(NUMBAST_OUTPUT_DIR "${WORKDIR}/out")
    set(OUTPUT_NAME "nvshmem_device_binding_generated.py")
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/externals/output")
    set(HIGH_LEVEL_BINDINGS_OUTPUT_DIR "${WORKDIR}/high_level_bindings")

    set(ASTCANOPY_CMAKE_INSTALL_PREFIX "${ASSET_DIR}/ast_canopy/install")
    set(NUMBAST_LD_LIBRARY_PATH "LD_LIBRARY_PATH=${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib:${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}")
    set(NUMBAST_PATH "PATH=${VENV_DIR}/bin:$ENV{PATH}")
    set(NUMBAST_ENV "${NUMBAST_LD_LIBRARY_PATH}" "${NUMBAST_PATH}")
    if(DEFINED ENV{NUMBA_CUDA_MLIR_SOURCE_DIR})
        set(NUMBA_CUDA_MLIR_PYTHONPATH "PYTHONPATH=$ENV{NUMBA_CUDA_MLIR_SOURCE_DIR}/src:$ENV{PYTHONPATH}")
        list(APPEND NUMBAST_ENV "${NUMBA_CUDA_MLIR_PYTHONPATH}")
    endif()
    set(NUMBAST_MLIR_COMMAND
        ${CMAKE_COMMAND} -E env ${NUMBAST_ENV} "${VENV_PYTHON_EXECUTABLE}" "-c"
        "__import__('numbast.experimental.mlir.tools.static_binding_generator', fromlist=['static_binding_generator']).static_binding_generator()"
        "--cfg-path" "${ASSET_DIR}/numbast-mlir/config_nvshmem.yml"
        "--output-dir" "${NUMBAST_OUTPUT_DIR}"
        "--bypass-parse-error" "true"
    )

    # Select the CUDA-specific wheel variant of numba-cuda-mlir ([cu12]/[cu13]) to match the
    # toolkit NVSHMEM is built against. The extra pins cuda-bindings/cuda-toolkit to the same
    # major version; without it the base pin (cuda-bindings>=12.9.1,<14) could install a 12.x
    # binding on a CUDA 13 toolkit. Requires >= 0.4.0 so ExternFunction lives in device_declarations.
    if(DEFINED CUDAToolkit_VERSION_MAJOR)
        set(NUMBA_CUDA_MLIR_PIP_SPEC "numba-cuda-mlir[cu${CUDAToolkit_VERSION_MAJOR}]>=0.4.0")
    else()
        set(NUMBA_CUDA_MLIR_PIP_SPEC "numba-cuda-mlir>=0.4.0")
    endif()

    if(DEFINED ENV{NUMBA_CUDA_MLIR_SOURCE_DIR})
        # A developer-provided source checkout takes precedence: use it directly via
        # PYTHONPATH instead of installing the published wheel.
        set(PIP_INSTALL_NUMBA_CUDA_MLIR_COMMAND
            ${CMAKE_COMMAND} -E echo "Using numba_cuda_mlir from NUMBA_CUDA_MLIR_SOURCE_DIR=$ENV{NUMBA_CUDA_MLIR_SOURCE_DIR}"
        )
        set(INSTALL_NUMBA_CUDA_MLIR_COMMAND
            ${CMAKE_COMMAND} -E env "${NUMBA_CUDA_MLIR_PYTHONPATH}" ${VENV_PYTHON_EXECUTABLE} -c "import numba_cuda_mlir"
        )
    else()
        # A plain `numbast` install does not pull its optional "mlir" extra (numba-cuda-mlir),
        # and we want the CUDA-matched wheel anyway, so install it explicitly here.
        set(PIP_INSTALL_NUMBA_CUDA_MLIR_COMMAND
            ${VENV_PYTHON_EXECUTABLE} -m pip install "${NUMBA_CUDA_MLIR_PIP_SPEC}"
        )
        set(INSTALL_NUMBA_CUDA_MLIR_COMMAND
            ${VENV_PYTHON_EXECUTABLE} -c "import numba_cuda_mlir"
        )
    endif()

    file(REMOVE_RECURSE "${WORKDIR}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/externals")
    file(MAKE_DIRECTORY "${WORKDIR}")
    file(MAKE_DIRECTORY "${ASSET_DIR}")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${NUMBAST_OUTPUT_DIR}")

    add_custom_target(
        clean_${PACKAGE_NAME}
        COMMAND rm -rvf ${WORKDIR}
        COMMAND mkdir -p ${WORKDIR}
        COMMAND mkdir -p ${OUTPUT_DIR}
        COMMAND touch ${OUTPUT_DIR}/clean_${PACKAGE_NAME}.txt
        COMMENT "Cleaning and recreating Numbast MLIR work directory"
    )

    add_custom_target(
        pip_install_${PACKAGE_NAME}
        COMMAND mkdir -p ${ASTCANOPY_CMAKE_INSTALL_PREFIX}
        COMMAND mkdir -p ${OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install click Jinja2 PyYAML ruff "ast_canopy>=0.5.0"
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install --no-deps numbast==${ADDNUMBASTMLIR_VERSION}
        COMMAND ${PIP_INSTALL_NUMBA_CUDA_MLIR_COMMAND}
        COMMAND ${INSTALL_NUMBA_CUDA_MLIR_COMMAND}
        WORKING_DIRECTORY ${WORKDIR}
        USES_TERMINAL
        DEPENDS clean_${PACKAGE_NAME}
        DEPENDS setup_py_bindings_env
        COMMAND touch ${OUTPUT_DIR}/install_${PACKAGE_NAME}.txt
        COMMENT "Installing Numbast MLIR binding generation environment"
    )

    add_custom_target(
        copy_source_${PACKAGE_NAME}
        COMMAND mkdir -p ${ASSET_DIR}
        COMMAND cp -rvf ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/numbast-mlir/ ${ASSET_DIR}
        COMMAND touch ${OUTPUT_DIR}/copy_source_${PACKAGE_NAME}.txt
        DEPENDS pip_install_${PACKAGE_NAME}
        COMMENT "Copying assets for Numbast MLIR binding generation"
    )

    add_custom_target(
        generate_numbast_mlir_config
        COMMAND mkdir -p ${ASSET_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast-mlir/config_nvshmem.py
            --nvshmem-home ${CMAKE_SOURCE_DIR}
            --config-version ${NUMBAST_CONFIG_VERSION}
            --entry-point-path ${ASSET_DIR}/numbast-mlir/numbast_entry_point.h
            --binding-name ${OUTPUT_NAME}
            --input-path ${ASSET_DIR}/numbast-mlir/templates/config_nvshmem.yml.j2
            --output-path ${ASSET_DIR}/numbast-mlir/config_nvshmem.yml
        DEPENDS copy_source_${PACKAGE_NAME}
        COMMAND touch ${OUTPUT_DIR}/copy_config_${PACKAGE_NAME}.txt
        COMMENT "Generating Numbast MLIR config for nvshmem"
    )

    add_custom_target(
        run_numbast_mlir
        COMMAND mkdir -p ${NUMBAST_OUTPUT_DIR}
        COMMAND ${NUMBAST_MLIR_COMMAND}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        VERBATIM
        DEPENDS pip_install_${PACKAGE_NAME}
        DEPENDS copy_source_${PACKAGE_NAME}
        DEPENDS generate_numbast_mlir_config
        COMMAND touch ${OUTPUT_DIR}/run_${PACKAGE_NAME}.txt
        COMMENT "Generating Numbast MLIR bindings..."
        COMMENT "Numbast MLIR command: ${NUMBAST_MLIR_COMMAND}"
    )

    add_custom_target(
        get_numbast_mlir_output
        COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba_cuda_mlir
        COMMAND cp -rvf ${NUMBAST_OUTPUT_DIR}/${OUTPUT_NAME} ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba_cuda_mlir/_numbast.py
        COMMAND cp -rvf ${ASSET_DIR}/numbast-mlir/numbast_entry_point.h ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba_cuda_mlir/entry_point.h
        DEPENDS run_numbast_mlir
    )

    add_custom_target(
        generate_high_level_bindings_${PACKAGE_NAME}
        COMMAND mkdir -p ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast-mlir/generate_rma.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast-mlir/generate_coll.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast-mlir/generate_amo.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast-mlir/generate_mem.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND touch ${OUTPUT_DIR}/generate_high_level_bindings_${PACKAGE_NAME}.txt
        COMMENT "Generating Numbast MLIR high level bindings..."
        DEPENDS get_numbast_mlir_output
    )

    add_custom_target(
        get_high_level_bindings_${PACKAGE_NAME}
        COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/numba_cuda_mlir
        COMMAND cp -rvf ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}/* ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/numba_cuda_mlir/
        COMMENT "Copying Numbast MLIR high level bindings into nvshmem4py"
        DEPENDS generate_high_level_bindings_${PACKAGE_NAME}
    )

    add_custom_target(build_bindings_${PACKAGE_NAME} DEPENDS get_high_level_bindings_${PACKAGE_NAME})

endfunction()
