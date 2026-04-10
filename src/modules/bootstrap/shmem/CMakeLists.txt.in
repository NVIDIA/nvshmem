set(SOURCE_LIST bootstrap_shmem.c)

nvshmem_add_bootstrap(nvshmem_bootstrap_shmem ${SOURCE_LIST})

find_library(
SHMEM_LIB
NAMES oshmem
HINTS ${SHMEM_HOME}
PATH_SUFFIXES lib lib64)
find_path(SHMEM_INCLUDE NAME shmem.h HINTS ${SHMEM_HOME}
          PATH_SUFFIXES include
)
add_library(shmem IMPORTED INTERFACE)
target_link_libraries(shmem INTERFACE ${SHMEM_LIB})
target_include_directories(shmem INTERFACE ${SHMEM_INCLUDE})

target_link_libraries(nvshmem_bootstrap_shmem PRIVATE shmem)

# Detect OpenSHMEM 1.5+ before enabling teams-based bootstrap support.
# OpenSHMEM 1.4 implementations such as OpenMPI oshmem expose incompatible
# teams signatures (e.g. shmem_team_sync returns void), so gate the teams path
# on the standard version macros first and only then verify the byte-level
# collectives used by the bootstrap are available.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${SHMEM_INCLUDE})
set(CMAKE_REQUIRED_LIBRARIES ${SHMEM_LIB})
check_c_source_compiles("
#include <shmem.h>
#if !defined(SHMEM_MAJOR_VERSION) || !defined(SHMEM_MINOR_VERSION)
#error OpenSHMEM version macros are not defined
#endif
#if (SHMEM_MAJOR_VERSION < 1) || (SHMEM_MAJOR_VERSION == 1 && SHMEM_MINOR_VERSION < 5)
#error OpenSHMEM version is older than 1.5
#endif
int main(void) { return 0; }
" NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)

if(NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)
    check_c_source_compiles("
#include <shmem.h>
int main(void) {
    shmem_team_t team = SHMEM_TEAM_WORLD;
    int (*f1)(shmem_team_t, uint8_t *, const uint8_t *, size_t) = shmem_uint8_fcollect;
    int (*f2)(shmem_team_t, uint8_t *, const uint8_t *, size_t) = shmem_uint8_alltoall;
    (void)team; (void)f1; (void)f2;
    return 0;
}
" NVSHMEM_SHMEM_HAS_TEAMS)
endif()

if(NVSHMEM_SHMEM_HAS_TEAMS)
    message(STATUS "OpenSHMEM >= 1.5 teams byte-level collectives detected")
    target_compile_definitions(nvshmem_bootstrap_shmem PRIVATE NVSHMEM_SHMEM_HAS_TEAMS)
    # OpenMPI's oshmem may advertise newer interfaces yet still return
    # ERR_NOT_IMPLEMENTED at runtime for teams collectives. Warn at build time;
    # the bootstrap will also check return codes at runtime.
    if(SHMEM_LIB MATCHES "oshmem")
        message(WARNING "OpenMPI oshmem detected: teams-based collectives may not be "
                "functional at runtime (shmem_uint8_alltoall returns ERR_NOT_IMPLEMENTED). "
                "The bootstrap will automatically fall back to legacy active-set "
                "collectives if this occurs.")
    endif()
elseif(NVSHMEM_SHMEM_VERSION_AT_LEAST_1_5)
    message(STATUS "OpenSHMEM >= 1.5 detected, but required teams byte-level collectives were not found")
else()
    message(STATUS "OpenSHMEM < 1.5 detected; disabling teams-based bootstrap support")
endif()

# Detect deprecated OpenSHMEM active-set collectives (shmem_collect32, shmem_alltoall32).
# These are the pre-1.5 APIs that require manual pSync scratch buffers.
# They may be removed from future OpenSHMEM implementations. When both are
# available, the teams path is preferred with active-set as a runtime fallback.
check_c_source_compiles("
#include <shmem.h>
int main(void) {
    long scratch[SHMEM_COLLECT_SYNC_SIZE];
    scratch[0] = SHMEM_SYNC_VALUE;
    shmem_collect32(scratch, scratch, 1, 0, 0, 1, scratch);
    shmem_alltoall32(scratch, scratch, 1, 0, 0, 1, scratch);
    return 0;
}
" NVSHMEM_SHMEM_HAS_ACTIVE_SET)
if(NVSHMEM_SHMEM_HAS_ACTIVE_SET)
    message(STATUS "OpenSHMEM active-set collectives detected (runtime fallback available)")
    target_compile_definitions(nvshmem_bootstrap_shmem PRIVATE NVSHMEM_SHMEM_HAS_ACTIVE_SET)
elseif(NOT NVSHMEM_SHMEM_HAS_TEAMS)
    message(WARNING "OpenSHMEM implementation supports neither teams-based nor "
            "active-set collectives. Skipping SHMEM bootstrap.")
    return()
endif()
