function(generateCuteBindings)
    
    # Locate Git
    find_package(Git REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "Git not found on the system!")
    else()
        message(STATUS "Git found: ${GIT_EXECUTABLE}")
    endif()

    set(VENV_DIR "${CMAKE_BINARY_DIR}/externals/venv")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")


    
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/externals")
    
    set(NUMBAST_CONFIG_VERSION "0.1.0")
    
    set(PACKAGE_NAME "numbast")
    # WORKDIR is the directory where Numbast binding generation happens
    # Use a separate directory for CuTe to avoid conflicts with Numbast bindings
    set(WORKDIR "${CMAKE_BINARY_DIR}/externals/${PACKAGE_NAME}_cute")
    # BINDGEN_TOOL_REPO is the directory where Numbast binding generation tool is cloned
    set(BINDGEN_TOOL_REPO "${WORKDIR}/${PACKAGE_NAME}_cute")
    # ASSET_DIR is the directory where `build_assets/numbast/` is cloned
    set(ASSET_DIR "${WORKDIR}/build_assets")
    # NUMBAST_OUTPUT_DIR is the directory where Numbast output is generated
    set(NUMBAST_OUTPUT_DIR "${WORKDIR}/out")
    # OUTPUT_NAME is the name of the output binding file
    set(OUTPUT_NAME "nvshmem_cute_device_binding_generated.py")

    # Path to install libastcanopy.so
    set(ASTCANOPY_CMAKE_INSTALL_PREFIX "${ASSET_DIR}/ast_canopy/install")
    set(NUMBAST_LD_LIBRARY_PATH "LD_LIBRARY_PATH=${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib:${ASTCANOPY_CMAKE_INSTALL_PREFIX}/lib64:$ENV{LD_LIBRARY_PATH}")
    set(NUMBAST_COMMAND "env" "${NUMBAST_LD_LIBRARY_PATH}" "${VENV_PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute/generate_cute_bindings.py)
    
    # High Level Bindings Settings
    set(HIGH_LEVEL_BINDINGS_OUTPUT_DIR ${NUMBAST_OUTPUT_DIR}/high_level/)

    # OUTPUT_DIR is the place to store each steps output file for cmake to validate step
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/externals/output")

    file(REMOVE_RECURSE "${WORKDIR}")
    # Ensure directories are created at configure time
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/externals")
    file(MAKE_DIRECTORY "${WORKDIR}")
    file(MAKE_DIRECTORY "${ASSET_DIR}")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${NUMBAST_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}")

    set(CUTE_TREE_DIR ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute/)

    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Step 1: Generate Numbast config for nvshmem/CuTe
    add_custom_target(
        generate_numbast_config_cute
        # Make dirs
        COMMAND mkdir -p ${ASSET_DIR}
        COMMAND mkdir -p ${OUTPUT_DIR}
        # Generate Numbast config for nvshmem (if applicable)
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${CUTE_TREE_DIR}/generate_cute_config.py
            --nvshmem-home ${CMAKE_SOURCE_DIR}
            --config-version ${NUMBAST_CONFIG_VERSION}
            --entry-point-path ${CUTE_TREE_DIR}/entry_point.h
            --binding-name ${OUTPUT_NAME}
            --input-path ${CUTE_TREE_DIR}/templates/config_nvshmem.yml.j2
            --output-path ${CUTE_TREE_DIR}/config_nvshmem.yml

        # Depends on the Numbast stuff being set up already.
        DEPENDS pip_install_numbast
        COMMAND touch ${OUTPUT_DIR}/copy_config.txt
        COMMENT "Generating Numbast config for nvshmem"
    )

    # Step 2: Run Numbast
    add_custom_target(
        run_numbast_cute
        # Set up environment for Numbast
        COMMAND mkdir -p ${NUMBAST_OUTPUT_DIR}
        COMMAND ${NUMBAST_COMMAND}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        DEPENDS pip_install_numbast
        DEPENDS generate_numbast_config_cute
        COMMAND touch ${OUTPUT_DIR}/run_numbast_cute.txt
        COMMENT "Generating Numbast bindings..."
        COMMENT "Numbast command: ${NUMBAST_COMMAND}"
    )


    # Step 3: Copy generated bindings into nvshmem4py
    add_custom_target(
        get_numbast_output_cute
        COMMAND cp -rvf ${CUTE_TREE_DIR}/_cuteast.py ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/cute/_cuteast.py
        # entry_point.h is both the entry point for parsing decls, and the entry point for CuTe DSL runtime compilation.
        # TODO: Do we need this with the linked bitcode?
    	COMMAND cp -rvf ${CUTE_TREE_DIR}/entry_point.h ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings/device/cute/entry_point.h
        DEPENDS run_numbast_cute
    )

    # Step 4: Generate CuTe high level bindings
    add_custom_target(
        generate_high_level_bindings_cute
        COMMAND mkdir -p ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute/generate_rma.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute/generate_collective.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND ${VENV_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/nvshmem4py/build_assets/cute/generate_amo.py --output-dir ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}
        COMMAND touch ${OUTPUT_DIR}/generate_cute_high_level_bindings.txt
        COMMENT "Generating High Level Bindings..."
        DEPENDS get_numbast_output_cute
    )

    # Step 5: Copy generated high level bindings into nvshmem4py
    add_custom_target(
        get_high_level_bindings_cute
        COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/cute
        COMMAND cp -rvf ${HIGH_LEVEL_BINDINGS_OUTPUT_DIR}/* ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/core/device/cute/
        COMMENT "Copying High Level Bindings into nvshmem4py"
        DEPENDS generate_high_level_bindings_cute
    )

    # Final target to trigger everything
    # Note: build_bindings_numbast may not exist if Numbast bindings weren't generated
    if(TARGET build_bindings_numbast)
        add_custom_target(build_bindings_cute DEPENDS build_bindings_numbast get_high_level_bindings_cute)
    else()
        add_custom_target(build_bindings_cute DEPENDS get_high_level_bindings_cute)
    endif()

endfunction()
