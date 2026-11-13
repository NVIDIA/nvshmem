# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hand-written top-level exports of ``mem``.

They name the generated pointer operations for the CuTile API: a symmetric
Array's counterpart on a peer, or its multicast view, with the Array's shape.
"""

from __future__ import annotations

import cuda.tile as ct

from . import generated as _raw
from ._adapter import as_scalar

__all__ = ["get_peer_array", "get_multicast_array"]


def get_peer_array(local, pe):
    """Return ``local``'s symmetric counterpart on ``pe``, with ``local``'s shape.

    ``nvshmem_ptr`` addresses only peers that NVSHMEM reaches over P2P; for
    any other peer the returned Array's base is null.
    """
    return _raw.nvshmem_ptr(local, as_scalar(pe, ct.int32))


def get_multicast_array(team, local):
    """Return the multicast view of ``local`` across ``team``, with ``local``'s shape.

    The returned Array's base is null unless the team supports multicast.
    """
    return _raw.nvshmemx_mc_ptr(as_scalar(team, ct.int32), local)
