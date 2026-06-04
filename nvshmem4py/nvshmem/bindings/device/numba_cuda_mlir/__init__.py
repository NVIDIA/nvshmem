# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.pathfinder import find_nvidia_header_directory

from numba_cuda_mlir import cuda

import os
import warnings

from nvshmem.core.nvshmem_types import NvshmemWarning


def _find_nvshmem_include_path():
    nvshmem_home = os.environ.get("NVSHMEM_HOME")
    if nvshmem_home:
        for include_path in (
            os.path.join(nvshmem_home, "src", "include"),
            os.path.join(nvshmem_home, "include"),
        ):
            if os.path.exists(os.path.join(include_path, "nvshmem.h")):
                return include_path
    return find_nvidia_header_directory("nvshmem")


def _append_nvrtc_search_paths(*paths):
    existing = cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS
    search_paths = [] if not existing else existing.split(":")
    for path in paths:
        if path and path not in search_paths:
            search_paths.append(path)
    cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = ":".join(search_paths)


if os.path.exists(os.path.join(os.path.dirname(__file__), "_numbast.py")):
    from . import _numbast
    from ._numbast import *
    from numba_cuda_mlir import types as _types
    from numba_cuda_mlir.extending import lowering_registry as _lowering_registry
    from numba_cuda_mlir.extending import typing_registry as _typing_registry
    from numba_cuda_mlir.numba_cuda.typing import signature as _signature
    from numba_cuda_mlir.numba_cuda.typing.templates import (
        ConcreteTemplate as _ConcreteTemplate,
    )
    from numba_cuda_mlir.types import CPointer as _CPointer
    from numba_cuda_mlir.types import int16 as _int16
    from numba_cuda_mlir.types import int32 as _int32
    from numba_cuda_mlir.types import void as _void

    def ptr():
        pass

    class _typing_ptr(_ConcreteTemplate):
        key = ptr
        cases = [
            _signature(_CPointer(_void), _CPointer(_void), _int32),
            _signature(_CPointer(_int16), _CPointer(_int16), _int32),
        ]

    _typing_registry.register_global(ptr, _types.Function(_typing_ptr))

    def _register_ptr_lowering():
        shim_raw_str = """
    extern "C" __device__ int
    _Z11nvshmem_ptr_nbst(void * &retval , void ** ptr, int* pe) {
    #ifdef NUMBAST_SHIM_PRE_CALL
        NUMBAST_SHIM_PRE_CALL();
    #endif
        retval = nvshmem_ptr(*ptr, *pe);
        return 0;
    }
        """

        @_lowering_registry.lower(ptr, _CPointer(_void), _int32)
        def impl(builder, target, args, kws):
            callconv = _numbast.FunctionCallConv(
                itanium_mangled_name="_Z11nvshmem_ptr",
                shim_writer=_numbast.shim_writer,
                shim_code=shim_raw_str,
                arg_is_ref=[False, False],
                intent_plan=None,
                out_return_types=None,
                cxx_return_type=None,
            )
            _numbast._numbast_link_shim(builder, _numbast.shim_obj)
            return callconv(builder, target, args, kws)

        shim_int16_raw_str = """
    extern "C" __device__ int
    nvshmem_ptr_int16_nbst(short * &retval , short ** ptr, int* pe) {
    #ifdef NUMBAST_SHIM_PRE_CALL
        NUMBAST_SHIM_PRE_CALL();
    #endif
        retval = static_cast<short *>(nvshmem_ptr(*ptr, *pe));
        return 0;
    }
        """

        @_lowering_registry.lower(ptr, _CPointer(_int16), _int32)
        def impl_int16(builder, target, args, kws):
            callconv = _numbast.FunctionCallConv(
                itanium_mangled_name="nvshmem_ptr_int16",
                shim_writer=_numbast.shim_writer,
                shim_code=shim_int16_raw_str,
                arg_is_ref=[False, False],
                intent_plan=None,
                out_return_types=None,
                cxx_return_type=None,
            )
            _numbast._numbast_link_shim(builder, _numbast.shim_obj)
            return callconv(builder, target, args, kws)

        return impl

    _register_ptr_lowering()

    INCLUDE_PATH = _find_nvshmem_include_path()
    PACKAGED_INCLUDE_PATH = find_nvidia_header_directory("nvshmem")
    if "nvshmem.h" not in os.listdir(INCLUDE_PATH):
        raise RuntimeError("nvshmem.h not found, package may not be properly installed")

    if not os.path.exists(INCLUDE_PATH):
        raise RuntimeError(
            f"NVSHMEM headers not found at {INCLUDE_PATH}. Please confirm that nvshmem is installed correctly."
        )

    CCCL_INCLUDE_PATH = find_nvidia_header_directory("cccl")

    if not os.path.exists(CCCL_INCLUDE_PATH):
        raise RuntimeError(
            f"CCCL headers not found at {CCCL_INCLUDE_PATH}. Please confirm that cccl is installed correctly."
        )

    this_folder = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(this_folder, "entry_point.h")):
        raise RuntimeError(
            "entry_point.h not found, package may not be properly installed"
        )

    _append_nvrtc_search_paths(
        INCLUDE_PATH, PACKAGED_INCLUDE_PATH, CCCL_INCLUDE_PATH, this_folder
    )

else:
    warnings.warn("Numba-CUDA-MLIR device bindings are not enabled", NvshmemWarning)
    _numbast = None
