# When NVSHMEM_FORCE_REBUILD_PYTHON_LIB is ON, remove generated binding files
# so that addCybind, addNumbast, and generateCuteBindings will run again.
# This must run at configure time, before those modules are included.

set(BINDINGS_BASE "${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/bindings")

set(CYBIND_OUTPUT "${BINDINGS_BASE}/_internal/nvshmem.pyx")
set(NUMBAST_OUTPUT "${BINDINGS_BASE}/device/numba/_numbast.py")
set(CUTEAST_OUTPUT "${BINDINGS_BASE}/device/cute/_cuteast.py")

if(EXISTS "${CYBIND_OUTPUT}")
    file(REMOVE "${CYBIND_OUTPUT}")
    message(STATUS "Force rebuild: removed ${CYBIND_OUTPUT}")
endif()

if(EXISTS "${NUMBAST_OUTPUT}")
    file(REMOVE "${NUMBAST_OUTPUT}")
    message(STATUS "Force rebuild: removed ${NUMBAST_OUTPUT}")
endif()

if(EXISTS "${CUTEAST_OUTPUT}")
    file(REMOVE "${CUTEAST_OUTPUT}")
    message(STATUS "Force rebuild: removed ${CUTEAST_OUTPUT}")
endif()
