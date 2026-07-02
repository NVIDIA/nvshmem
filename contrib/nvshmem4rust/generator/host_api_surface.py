# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy

CYBIND_HOST_FUNCTIONS = {
    # Init/Fini
    "nvshmemx_hostlib_init_attr": {},
    "nvshmemx_get_uniqueid": {},
    "nvshmemx_set_attr_uniqueid_args": {},
    "nvshmemx_set_attr_mpi_comm_args": {},
    "nvshmemx_hostlib_finalize": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_cumodule_init": {
        "except?": -1,
        "return": "TRANSPARENT",
    },
    "nvshmemx_cumodule_finalize": {
        "except?": -1,
        "return": "TRANSPARENT",
    },
    "nvshmemx_culibrary_init": {
        "except?": -1,
        "return": "TRANSPARENT",
    },
    "nvshmemx_culibrary_finalize": {
        "except?": -1,
        "return": "TRANSPARENT",
    },
    # Memory Management
    "nvshmem_malloc": {
        "return": "TRANSPARENT",
        "except?": 0,
    },
    "nvshmem_free": {
        "return": "TRANSPARENT",
    },
    "nvshmem_calloc": {
        "arg_names": ("count", "size"),
        "return": "TRANSPARENT",
        "except?": 0,
    },
    "nvshmem_align": {
        "arg_names": ("count", "size"),
        "return": "TRANSPARENT",
        "except?": 0,
    },
    "nvshmem_ptr": {
        "return": "TRANSPARENT",
        "except?": 0,
        "arg_names": ("dest", "pe"),
    },
    "nvshmemx_mc_ptr": {
        "return": "TRANSPARENT",
        "except?": 0,
        "arg_names": ("team", "ptr"),
    },
    "nvshmemx_buffer_register_symmetric": {
        "return": "TRANSPARENT",
        "except?": 0,
    },
    "nvshmemx_buffer_unregister_symmetric": {
        "return": "TRANSPARENT",
        "except?": 0,
    },
    # Team Management
    "nvshmem_my_pe": {
        "return": "TRANSPARENT",
        "except?": -1,
    },
    "nvshmem_n_pes": {
        "return": "TRANSPARENT",
        "except?": -1,
    },
    "nvshmem_team_my_pe": {
        "return": "TRANSPARENT",
        "except?": -1,
    },
    "nvshmem_team_n_pes": {
        "return": "TRANSPARENT",
        "except?": -1,
    },
    # RMA
    "nvshmemx_putmem_on_stream": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_getmem_on_stream": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_putmem_signal_on_stream": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_signal_op_on_stream": {
        "return": "TRANSPARENT",
    },
    # Synchronization
    "nvshmem_barrier": {},
    "nvshmem_barrier_all": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_barrier_on_stream": {
        "except?": -1,
    },
    "nvshmemx_team_sync_on_stream": {
        "return": "TRANSPARENT",
        "except?": 0,
    },
    "nvshmemx_barrier_all_on_stream": {
        "arg_names": ["stream"],
        "return": "TRANSPARENT",
    },
    "nvshmemx_sync_all_on_stream": {
        "arg_names": ["stream"],
        "return": "TRANSPARENT",
    },
    "nvshmemx_signal_wait_until_on_stream": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_quiet_on_stream": {
        "return": "TRANSPARENT",
    },
    # Misc.
    "nvshmem_info_get_version": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_vendor_get_version_info": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_init_status": {
        "return": "TRANSPARENT",
        "except?": 0,
    },
    # Team creation
    "nvshmem_team_get_config": {
        "return": "TRANSPARENT",
    },
    "nvshmem_team_translate_pe": {
        "return": "TRANSPARENT",
        "except?": -1,
    },
    "nvshmem_team_split_strided": {
        "except?": -1,
    },
    "nvshmem_team_split_2d": {
        "except?": -1,
    },
    "nvshmem_team_destroy": {
        "return": "TRANSPARENT",
    },
    "nvshmemx_team_init": {
        "except?": -1,
    },
    "nvshmemx_team_get_uniqueid": {},
}

RUST_EXTRA_HOST_FUNCTIONS = (
    "nvshmemx_putmem_signal_nbi_on_stream",
    "nvshmemx_putmem_nbi_on_stream",
    "nvshmemx_getmem_nbi_on_stream",
    "nvshmemx_alltoallmem_on_stream",
    "nvshmemx_broadcastmem_on_stream",
    "nvshmemx_fcollectmem_on_stream",
    "nvshmemx_flush_on_stream",
)

COLLECTIVE_KINDS = ("fcollect", "alltoall", "reduce", "reducescatter",
                    "broadcast")
COLLECTIVE_KINDS_WITHOUT_OPERATORS = frozenset(
    ("fcollect", "alltoall", "broadcast"))

CYBIND_COLLECTIVE_TYPES = (
    "float",
    "half",
    "double",
    "bfloat16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "short",
    "int16",
    "int32",
    "int",
    "int64",
    "long",
    "longlong",
    "size",
    "char",
    "schar",
)

STANDARD_RMA_TYPES = (
    "bfloat16",
    "half",
    "float",
    "double",
    "char",
    "short",
    "schar",
    "int",
    "long",
    "longlong",
    "uchar",
    "ushort",
    "uint",
    "ulong",
    "ulonglong",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "size",
    "ptrdiff",
)

BITWISE_REDUCE_TYPES = (
    "uchar",
    "ushort",
    "uint",
    "ulong",
    "ulonglong",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "size",
)

STANDARD_REDUCE_TYPES = tuple(dtype for dtype in STANDARD_RMA_TYPES
                              if dtype != "ptrdiff")

CYBIND_REDUCE_OPERATOR_TYPES = {
    op: CYBIND_COLLECTIVE_TYPES
    for op in ("sum", "min", "max")
}

RUST_REDUCE_OPERATOR_TYPES = {
    **{
        op: BITWISE_REDUCE_TYPES
        for op in ("and", "or", "xor")
    },
    **{
        op: STANDARD_REDUCE_TYPES
        for op in ("max", "min", "sum", "prod")
    },
}

RUST_STREAM_API_PATTERNS = (
    r"^nvshmemx_[A-Za-z0-9]+_"
    r"(?:p|g|put|get|iput|iget|put_nbi|get_nbi|put_signal|put_signal_nbi)_on_stream$",
    r"^nvshmemx_(?:put|get|iput|iget)(?:8|16|32|64|128)(?:_nbi)?_on_stream$",
    r"^nvshmemx_put(?:8|16|32|64|128)_signal(?:_nbi)?_on_stream$",
    r"^nvshmemx_[A-Za-z0-9]+_wait_until(?:_all|_all_vector)?_on_stream$",
)


def _collective_names(types, reduce_operator_types):
    reduce_type_sets = {
        op: frozenset(op_types)
        for op, op_types in reduce_operator_types.items()
    }

    names = []
    for dtype in types:
        for collective in COLLECTIVE_KINDS:
            if collective in COLLECTIVE_KINDS_WITHOUT_OPERATORS:
                names.append(f"nvshmemx_{dtype}_{collective}_on_stream")
            else:
                for op, op_types in reduce_type_sets.items():
                    if dtype in op_types:
                        names.append(
                            f"nvshmemx_{dtype}_{op}_{collective}_on_stream")
    return names


def python_cybind_functions():
    functions = deepcopy(CYBIND_HOST_FUNCTIONS)
    for name in _collective_names(CYBIND_COLLECTIVE_TYPES,
                                  CYBIND_REDUCE_OPERATOR_TYPES):
        functions[name] = {"except?": -1}
    return functions


def rust_host_allowlist():
    return (set(CYBIND_HOST_FUNCTIONS)
            | set(RUST_EXTRA_HOST_FUNCTIONS)
            | set(
                _collective_names(STANDARD_RMA_TYPES,
                                  RUST_REDUCE_OPERATOR_TYPES)))


def rust_host_stream_api_patterns():
    return tuple(re.compile(pattern) for pattern in RUST_STREAM_API_PATTERNS)
