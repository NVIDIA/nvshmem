# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED NVSHMEM_GIT_COMMIT_WORKDIR)
  message(FATAL_ERROR "NVSHMEM_GIT_COMMIT_WORKDIR must be set")
endif()

set(_nvshmem_git_commit_file "${NVSHMEM_GIT_COMMIT_WORKDIR}/git_commit.txt")

set(_nvshmem_source_commit "not built from a git repo")

if(EXISTS "${_nvshmem_git_commit_file}")
  file(STRINGS "${_nvshmem_git_commit_file}" _nvshmem_existing_git_commit LIMIT_COUNT 1)
  if(NOT "${_nvshmem_existing_git_commit}" STREQUAL "")
    set(_nvshmem_source_commit "${_nvshmem_existing_git_commit}")
  endif()
endif()

if(EXISTS "${NVSHMEM_GIT_COMMIT_WORKDIR}/.git")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${NVSHMEM_GIT_COMMIT_WORKDIR}"
    RESULT_VARIABLE _nvshmem_git_result
    OUTPUT_VARIABLE _nvshmem_git_commit
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if("${_nvshmem_git_result}" STREQUAL "0" AND NOT "${_nvshmem_git_commit}" STREQUAL "")
    set(_nvshmem_source_commit "${_nvshmem_git_commit}")
  endif()
endif()

file(WRITE "${_nvshmem_git_commit_file}" "${_nvshmem_source_commit}\n")
foreach(_nvshmem_version_var IN ITEMS
    NVSHMEM_MAJOR
    NVSHMEM_MINOR
    NVSHMEM_PATCH
    NVSHMEM_PACKAGE
    NVSHMEM_BASE_VERSION
    NVSHMEM_NUMERIC_VERSION
    NVSHMEM_ARTIFACT_VERSION
    NVSHMEM_PACKAGE_VERSION
    NVSHMEM_PACKAGE_RELEASE
    NVSHMEM_RPM_PACKAGE_VERSION
    NVSHMEM_RPM_PACKAGE_RELEASE
    NVSHMEM_VERSION_STAGE
    NVSHMEM_VERSION_STAGE_NUMBER)
  file(APPEND "${_nvshmem_git_commit_file}"
       "${_nvshmem_version_var} := ${${_nvshmem_version_var}}\n")
endforeach()
