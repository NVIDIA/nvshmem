function(AddNumbast GIT_TAG)
    
    # Locate Git
    find_package(Git REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "Git not found on the system!")
    else()
        message(STATUS "Git found: ${GIT_EXECUTABLE}")
    endif()

    set(VENV_DIR "${CMAKE_SOURCE_DIR}/build/externals/venv")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")

    cmake_parse_arguments(PARSE_ARGV 0 ADDNUMBAST "" "GIT_TAG" "")

    if(NOT DEFINED GIT_TAG)
        message(FATAL_ERROR "GIT_TAG not provided to AddNumbast")
    endif()

    
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build/externals")
    
    set(NUMBAST_CONFIG_VERSION "0.1.0")
    
    set(PACKAGE_NAME "numbast")
    # WORKDIR is the directory where Numbast binding generation happens
    set(WORKDIR "${CMAKE_SOURCE_DIR}/build/externals/${PACKAGE_NAME}")
    # BINDGEN_TOOL_REPO is the directory where Numbast binding generation tool is cloned
    set(BINDGEN_TOOL_REPO "${WORKDIR}/${PACKAGE_NAME}")
    # ASSET_DIR is the directory where `build_assets/numbast/` is cloned
    set(ASSET_DIR "${WORKDIR}/build_assets")
    # NUMBAST_OUTPUT_DIR is the directory where Numbast output is generated
    set(NUMBAST_OUTPUT_DIR "${WORKDIR}/out")
    # OUTPUT_NAME is the name of the output binding file
    set(OUTPUT_NAME "nvshmem_device_binding_generated.py")

    # Path to install libastcnopy.so
    set(ASTCANOPY_CMAKE_INSTALL_PREFIX "${ASSET_DIR}/ast_canopy/install")
    set(NUMBAST_LD_LIBRARY_PATH "LD_LIBRARY_PATH=${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}")
    set(NUMBAST_COMMAND "env" "${NUMBAST_LD_LIBRARY_PATH}" "${VENV_PYTHON_EXECUTABLE}" "-m" "numbast" "--cfg-path" "${ASSET_DIR}/numbast/config_nvshmem.yml" "--output-dir" "${NUMBAST_OUTPUT_DIR}")
    
    # OUTPUT_DIR is the place to store each steps output file for cmake to validate step
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build/externals/output")

    file(REMOVE_RECURSE "${WORKDIR}")
    # Ensure directories are created at configure time
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build/externals")
    file(MAKE_DIRECTORY "${WORKDIR}")
    file(MAKE_DIRECTORY "${ASSET_DIR}")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${NUMBAST_OUTPUT_DIR}")


    # Step 0: Clean the the numbast working directory
    add_custom_target(
        clean_${PACKAGE_NAME}
        COMMAND rm -rvf ${WORKDIR}

        COMMAND mkdir -p ${OUTPUT_DIR}
        COMMAND touch ${OUTPUT_DIR}/clean.txt
        COMMENT "Cleaning Numbast repository"
    )

    # Step 1: Clone the Numbast repository
    add_custom_target(
        clone_${PACKAGE_NAME}
        COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/build/externals
        COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch ${ADDNUMBAST_GIT_TAG} https://github.com/nvidia/numbast.git ${BINDGEN_TOOL_REPO}
        COMMAND touch ${OUTPUT_DIR}/git_clone.txt
        RESULT_VARIABLE clone_result
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build/externals
        # DEPENDS install_llvm
        DEPENDS clean_${PACKAGE_NAME}
        COMMENT "Cloning Numbast repository at tag ${ADDNUMBAST_GIT_TAG}"
    )

    # Step 2: install numbast from source
    add_custom_target(
        pip_install_numbast
        COMMAND mkdir -p ${ASTCANOPY_CMAKE_INSTALL_PREFIX}
        COMMAND mkdir -p ${OUTPUT_DIR}
        COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install --upgrade pip
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/nvshmem4py/requirements_build.txt
	COMMAND env PYTHON_EXECUTABLE=${VENV_PYTHON_EXECUTABLE} ASTCANOPY_INSTALL_PATH=${ASTCANOPY_CMAKE_INSTALL_PREFIX} ${BINDGEN_TOOL_REPO}/ast_canopy/build.sh
        COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install -e ${BINDGEN_TOOL_REPO}/numbast/
        WORKING_DIRECTORY ${WORKDIR}
        USES_TERMINAL
        DEPENDS clone_${PACKAGE_NAME}
        COMMAND touch ${OUTPUT_DIR}/install_from_source.txt
        COMMENT "Installing Numbast from source..."
    )

    # Step 3: Copy binding generation assets into build directory
    add_custom_target(
        copy_source_${PACKAGE_NAME}

        # Make dirs
        COMMAND mkdir -p ${ASSET_DIR}

        # Copy assets, but stop at the first failure
        COMMAND cp -rvf ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/numbast/ ${ASSET_DIR}
	    COMMAND touch ${OUTPUT_DIR}/copy_source.txt
        DEPENDS pip_install_numbast
        COMMENT "Copying assets for Numbast binding generation"
    )

    # Step 4: Generate Numbast config for nvshmem
    add_custom_target(
        generate_numbast_config
        # Make dirs
        COMMAND mkdir -p ${ASSET_DIR}

        # Generate Numbast config for nvshmem (if applicable)
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${ASSET_DIR}/numbast/config_nvshmem.py
            --nvshmem-home ${CMAKE_SOURCE_DIR}
            --config-version ${NUMBAST_CONFIG_VERSION}
            --entry-point-path ${ASSET_DIR}/numbast/numbast_entry_point.h
            --binding-name ${OUTPUT_NAME}
            --input-path ${ASSET_DIR}/numbast/templates/config_nvshmem.yml.j2
            --output-path ${ASSET_DIR}/numbast/config_nvshmem.yml

        DEPENDS pip_install_numbast # The binding generation depends on numbast dependencies (e.g. pyyaml, click)
        DEPENDS copy_source_${PACKAGE_NAME}
        COMMAND touch ${OUTPUT_DIR}/copy_config.txt
        COMMENT "Generating Numbast config for nvshmem"
    )

    # Step 5: Run Numbast
    add_custom_target(
        run_numbast
        # Set up environment for Numbast
        COMMAND mkdir -p ${NUMBAST_OUTPUT_DIR}
        COMMAND ${NUMBAST_COMMAND}
        COMMAND echo "Running Numbast" > ${OUTPUT_DIR}/result.txt
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        DEPENDS pip_install_numbast
        DEPENDS copy_source_${PACKAGE_NAME}
        DEPENDS generate_numbast_config
        COMMAND touch ${OUTPUT_DIR}/run_numbast.txt
        COMMENT "Generating Numbast bindings..."
        COMMENT "Numbast command: ${NUMBAST_COMMAND}"
    )

    # Step 6: Copy generated bindings into nvshmem4py
    add_custom_target(
        get_numbast_output
        COMMAND cp -rf ${NUMBAST_OUTPUT_DIR}/${OUTPUT_NAME} ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba/_numbast.py
        # entry_point.h is both the entry point for parsing decls, and the entry point for Numba runtime compilation.
    	COMMAND cp -rf ${ASSET_DIR}/numbast/numbast_entry_point.h ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/numba/entry_point.h
        DEPENDS run_numbast
    )

    # Final target to trigger everything
    add_custom_target(build_bindings_${PACKAGE_NAME} DEPENDS get_numbast_output)

endfunction()
