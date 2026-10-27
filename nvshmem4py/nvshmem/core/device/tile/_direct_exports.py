# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hand-written top-level exports of ``direct``: NVSHMEM enums for kernels.

CuTile folds ``IntEnum`` members to compile-time constants, so a kernel passes
them to NVSHMEM calls directly, as in ``signal_wait_until(signal,
ComparisonType.CMP_EQ, 1)``.  ``Teams`` holds the predefined teams: the host
``nvshmem.core.Teams`` also tracks teams created at run time, so it is not a
constant; pass such a team's handle into the kernel instead.
"""

from nvshmem.bindings import Cmp_type as ComparisonType
from nvshmem.bindings import Signal_op as SignalOp
from nvshmem.bindings import Team_id as Teams

__all__ = ["ComparisonType", "SignalOp", "Teams"]
