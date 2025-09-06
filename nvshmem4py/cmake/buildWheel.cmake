include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64)")
    set(ARCH_NAME "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)")
    set(ARCH_NAME "x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

function(BuildWheel WHEEL_TARGET PY_VER PYTHON_EXECUTABLE CUDA_VER)

    set(PYPROJECT_TOML "${CMAKE_SOURCE_DIR}/nvshmem4py/pyproject.toml")
    set(BUILD_DIR "${CMAKE_SOURCE_DIR}/build/dist")


    set(VENV_DIR "${CMAKE_SOURCE_DIR}/build/externals/venv_${PY_VER}")
    set(VENV_PYTHON_EXECUTABLE "${VENV_DIR}/bin/python3")

    if(NOT TARGET make_venv_${PY_VER})
        add_custom_target(make_venv_${PY_VER}
            COMMAND ${PYTHON_EXECUTABLE} -m venv ${VENV_DIR}
            COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install --upgrade pip
            COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/nvshmem4py/requirements_build.txt
            COMMAND ${VENV_PYTHON_EXECUTABLE} -m pip install --upgrade wheel setuptools
        )
    endif()


    set(PYPROJECT_PATH "${CMAKE_SOURCE_DIR}/nvshmem4py/pyproject.toml")
    set(SETUP_PATH "${CMAKE_SOURCE_DIR}/nvshmem4py/setup.py")
    message("BUILD DIR IS ${BUILD_DIR}")
    message("BINARY DIR IS ${CMAKE_BINARY_DIR}")
    message("SOURCE DIR IS ${CMAKE_SOURCE_DIR}")

    set(WHEEL_NAME "nvshmem4py-cu${CUDA_VER}")
    set(WHEEL_STR "nvshmem4py_cu${CUDA_VER}")

    add_custom_target(
        ${WHEEL_TARGET}
        COMMAND echo "Editing files for package ver ${CUDA_VER}"
        # Patch version numbers
        COMMAND ${CMAKE_SOURCE_DIR}/nvshmem4py/scripts/generate_pyproject_toml.py ${CUDA_VER} ${CMAKE_SOURCE_DIR}/nvshmem4py > ${CMAKE_SOURCE_DIR}/nvshmem4py/pyproject.toml
        COMMAND sed -i -e "s/^name = \"nvshmem4py\"/name = \"${WHEEL_NAME}\"/" ${PYPROJECT_PATH}
        COMMAND sed -i -e "s|requirements.txt|requirements_cuda${CUDA_VER}.txt|" ${SETUP_PATH}
        COMMAND mkdir -p ${BUILD_DIR}
        COMMAND echo "Building whl and tgz for packages ${CUDA_VER}"
        COMMAND ${CMAKE_COMMAND} -E env "CPPFLAGS=-I${CUDA_HOME}/include/" "PACKAGE_NAME=${WHEEL_NAME}" ${VENV_PYTHON_EXECUTABLE} -m build --outdir ${BUILD_DIR} --no-isolation
        COMMAND bash -c "export PATH=${VENV_DIR}/bin:$PATH; ls ${BUILD_DIR}/${WHEEL_STR}*.whl | xargs ${VENV_PYTHON_EXECUTABLE} -m auditwheel repair --plat manylinux_2_34_${ARCH_NAME} -w ${BUILD_DIR}/"
        # Undo patching version numbers
        COMMAND sed -i -e "s/^name = \"${WHEEL_NAME}\"/name = \"nvshmem4py\"/" ${PYPROJECT_PATH}
        COMMAND sed -i -e "s|requirements_cuda${CUDA_VER}.txt|requirements.txt|" ${SETUP_PATH}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/nvshmem4py/
        COMMAND echo "Listing outdir ${BUILD_DIR} for packages ${CUDA_VER}"
        COMMAND ls ${BUILD_DIR}
        USES_TERMINAL
        COMMENT "Building Python wheel and tarball..."
        VERBATIM
    )

    if(TARGET build_bindings_cybind)
        add_dependencies(${WHEEL_TARGET} build_bindings_cybind)
    endif()

    if(NOT EXISTS ${VENV_PYTHON_EXECUTABLE})
        add_dependencies(${WHEEL_TARGET} make_venv_${PY_VER})
    endif()


endfunction()