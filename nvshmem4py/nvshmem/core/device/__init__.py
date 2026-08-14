# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

__all__ = ["numba", "numba_cuda_mlir", "cute"]


def __getattr__(name):
    """Lazily import device modules when they are first accessed."""
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
