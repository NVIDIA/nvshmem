# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Branch-local defaults for the publishable artifact version stage.
#
# Keep the CMake project version numeric for ABI and compatibility checks. Use
# these defaults to distinguish non-final artifacts in package names and wheels:
#
#   devel branch:          dev, with a CI/DVS-provided build number when available
#   release/vX.Y.Z RC:     rc, with NVSHMEM_VERSION_STAGE_NUMBER set to the RC number
#   patch branch baseline: rc, with NVSHMEM_VERSION_STAGE_NUMBER set to 0
#   release/vX.Y.Z final:  release, with no stage number
#
# Build environments can override these defaults with NVSHMEM_VERSION_STAGE /
# NVSHMEM_VERSION_STAGE_NUMBER environment values, or with
# -DNVSHMEM_VERSION_STAGE_OVERRIDE=... and
# -DNVSHMEM_VERSION_STAGE_NUMBER_OVERRIDE=.... CMake override values take
# precedence over environment values when both are set.

set(NVSHMEM_VERSION_STAGE_DEFAULT "rc")
set(NVSHMEM_VERSION_STAGE_NUMBER_DEFAULT "8")
