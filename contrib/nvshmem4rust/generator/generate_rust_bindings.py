# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
from pathlib import Path

import yaml
from ast_canopy import parse_declarations_from_source

from host_api_surface import rust_host_allowlist, rust_host_stream_api_patterns

_TYPE_MAP = {
    "CUlib_st": "core::ffi::c_void",
    "CUmod_st": "core::ffi::c_void",
    "CUstream_st": "core::ffi::c_void",
    "__half": "nvshmem_half",
    "__nv_bfloat16": "nvshmem_bfloat16",
    "bool": "bool",
    "char": "i8",
    "double": "f64",
    "double2": "double2",
    "float": "f32",
    "int": "i32",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "long": "i64",
    "long long": "i64",
    "nvshmem_team_t": "nvshmem_team_t",
    "nvshmem_team_config_t": "nvshmem_team_config_t",
    "nvshmem_team_config_v1": "nvshmem_team_config_t",
    "nvshmem_team_config_v2": "nvshmem_team_config_t",
    "nvshmemx_init_args_t": "nvshmemx_init_args_t",
    "nvshmemx_init_args_v1": "nvshmemx_init_args_t",
    "nvshmemx_init_args_v2": "nvshmemx_init_args_t",
    "nvshmemx_init_attr_t": "nvshmemx_init_attr_t",
    "nvshmemx_init_attr_v1": "nvshmemx_init_attr_t",
    "nvshmemx_init_attr_v2": "nvshmemx_init_attr_t",
    "nvshmemx_team_uniqueid_t": "nvshmemx_team_uniqueid_t",
    "nvshmemx_smem_amount_t": "nvshmemx_smem_amount_t",
    "nvshmemx_team_t": "nvshmemx_team_t",
    "nvshmemx_uniqueid_args_t": "nvshmemx_uniqueid_args_t",
    "nvshmemx_uniqueid_args_v1": "nvshmemx_uniqueid_args_t",
    "nvshmemx_uniqueid_t": "nvshmemx_uniqueid_t",
    "nvshmemx_uniqueid_v1": "nvshmemx_uniqueid_t",
    "ptrdiff_t": "isize",
    "short": "i16",
    "signed char": "i8",
    "size_t": "usize",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "uintptr_t": "usize",
    "unsigned char": "u8",
    "unsigned int": "u32",
    "unsigned long": "u64",
    "unsigned long long": "u64",
    "unsigned short": "u16",
    "void": "core::ffi::c_void",
}

_HOST_STREAM_API_PATTERNS = rust_host_stream_api_patterns()

_RUST_KEYWORDS = {
    "as",
    "async",
    "await",
    "abstract",
    "become",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "do",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "gen",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "macro",
    "match",
    "mod",
    "move",
    "mut",
    "override",
    "priv",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "unsized",
    "unsafe",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
}
_RUST_NON_RAW_IDENTIFIERS = {"crate", "self", "Self", "super"}


def _qualified_type_name(type_obj):
    match = re.fullmatch(r"<Type: (.*)>", str(type_obj))
    if match:
        return match.group(1).strip()
    return type_obj.unqualified_non_ref_type_name.strip()


def _base_type(type_name):
    return (type_name.replace("const", "").replace("volatile", "").replace(
        "__restrict__", "").replace("restrict", "").replace("*", "").strip())


def _is_const_pointer(type_obj):
    qualified = _qualified_type_name(type_obj)
    return "*" in qualified and bool(re.search(r"\bconst\b.*\*", qualified))


def _rust_type(type_obj, *, is_return=False):
    unqualified = type_obj.unqualified_non_ref_type_name.strip()
    pointer_depth = unqualified.count("*")
    base = _base_type(unqualified)

    if base not in _TYPE_MAP:
        raise RuntimeError(
            f"Unsupported NVSHMEM type in Rust binding generation: '{base}'")

    rust_base = _TYPE_MAP[base]
    if pointer_depth == 0:
        if base == "void" and is_return:
            return None
        return rust_base

    pointer_kind = "*const" if _is_const_pointer(type_obj) else "*mut"
    rust_type = rust_base
    for _ in range(pointer_depth):
        rust_type = f"{pointer_kind} {rust_type}"
    return rust_type


def _rust_param_name(name, index):
    if not name:
        return f"arg{index}"
    sanitized = re.sub(r"\W", "_", name)
    if sanitized[0].isdigit():
        sanitized = f"arg_{sanitized}"
    return _rust_identifier(sanitized)


def _rust_identifier(identifier):
    if identifier in _RUST_NON_RAW_IDENTIFIERS:
        return f"{identifier}_"
    if identifier in _RUST_KEYWORDS:
        return f"r#{identifier}"
    return identifier


def _generate_function_decl(function):
    params = []
    for idx, param in enumerate(function.params):
        params.append(
            f"{_rust_param_name(param.name, idx)}: {_rust_type(param.type_)}")

    return_type = _rust_type(function.return_type, is_return=True)
    return_suffix = f" -> {return_type}" if return_type is not None else ""
    return f"    pub fn {function.name}({', '.join(params)}){return_suffix};"


def _rust_api_name(function_name):
    if function_name == "nvshmemx_cumodule_init":
        return "raw_cumodule_init"
    if function_name == "nvshmemx_cumodule_finalize":
        return "raw_cumodule_finalize"

    if function_name.startswith("nvshmemx_"):
        api_name = function_name.removeprefix("nvshmemx_")
    elif function_name.startswith("nvshmem_"):
        api_name = function_name.removeprefix("nvshmem_")
    else:
        api_name = function_name

    return _rust_identifier(api_name)


def _is_safe_host_wrapper(function_name):
    return function_name in {
        "nvshmem_my_pe",
        "nvshmem_n_pes",
        "nvshmem_barrier_all",
        "nvshmemx_init_status",
    }


def _generate_host_api_wrapper(function):
    params = []
    arg_names = []
    for idx, param in enumerate(function.params):
        param_name = _rust_param_name(param.name, idx)
        params.append(f"{param_name}: {_rust_type(param.type_)}")
        arg_names.append(param_name)

    api_name = _rust_api_name(function.name)
    return_type = _rust_type(function.return_type, is_return=True)
    return_suffix = f" -> {return_type}" if return_type is not None else ""
    safety_prefix = "pub fn" if _is_safe_host_wrapper(
        function.name) else "pub unsafe fn"
    call = f"sys::{function.name}({', '.join(arg_names)})"
    if return_type is None:
        body = [f"    unsafe {{ {call}; }}"]
    else:
        body = [f"    unsafe {{ {call} }}"]

    return "\n".join([
        f"{safety_prefix} {api_name}({', '.join(params)}){return_suffix} {{",
        *body,
        "}",
    ])


def _host_function_allowlist():
    return rust_host_allowlist()


def _matches_host_stream_api(name):
    return any(
        pattern.fullmatch(name) for pattern in _HOST_STREAM_API_PATTERNS)


def _select_host_functions(decls):
    allowlist = _host_function_allowlist()
    selected = []
    seen = set()
    for function in decls.functions:
        if (function.name in allowlist or _matches_host_stream_api(
                function.name)) and function.name not in seen:
            selected.append(function)
            seen.add(function.name)

    missing = sorted(allowlist - seen)
    if missing:
        raise RuntimeError(
            "Host Rust binding generation did not find expected NVSHMEM host APIs: "
            + ", ".join(missing))
    return selected


def _common_header_lines():
    return [
        "/*",
        " * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "",
        "// Generated by contrib/nvshmem4rust/generator/generate_rust_bindings.py.",
    ]


def _common_allow_lines():
    return [
        "",
        "#[allow(dead_code)]",
        "#[allow(non_camel_case_types)]",
        "#[allow(non_snake_case)]",
        "#[allow(non_upper_case_globals)]",
    ]


def _device_prelude_lines(config):
    return [
        "// Include this module from CUDA-Oxide device code and link with NVSHMEM LTOIR.",
        *_common_allow_lines(),
        "",
        "use cuda_device::device;",
        "",
        "pub type nvshmem_team_t = i32;",
        "pub type nvshmemx_team_t = nvshmem_team_t;",
        "pub type nvshmemx_smem_amount_t = i32;",
        "",
        "#[repr(transparent)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmem_half(pub u16);",
        "",
        "#[repr(transparent)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmem_bfloat16(pub u16);",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct double2 {",
        "    pub x: f64,",
        "    pub y: f64,",
        "}",
        "",
        "pub const NVSHMEM_TEAM_INVALID: nvshmem_team_t = -1;",
        "pub const NVSHMEM_TEAM_WORLD: nvshmem_team_t = 0;",
        "pub const NVSHMEM_TEAM_SHARED: nvshmem_team_t = 1;",
        "pub const NVSHMEMX_TEAM_NODE: nvshmem_team_t = 2;",
        "pub const NVSHMEMX_TEAM_SAME_MYPE_NODE: nvshmem_team_t = 3;",
        "pub const NVSHMEM_TEAM_MC_SHARED: nvshmem_team_t = 6;",
        "",
        "pub const NVSHMEM_CMP_EQ: i32 = 0;",
        "pub const NVSHMEM_CMP_NE: i32 = 1;",
        "pub const NVSHMEM_CMP_GT: i32 = 2;",
        "pub const NVSHMEM_CMP_LE: i32 = 3;",
        "pub const NVSHMEM_CMP_LT: i32 = 4;",
        "pub const NVSHMEM_CMP_GE: i32 = 5;",
        "",
        "pub const NVSHMEM_SIGNAL_SET: i32 = 9;",
        "pub const NVSHMEM_SIGNAL_ADD: i32 = 10;",
        "",
        "pub const NVSHMEMX_SMEM_RECOMMENDED: nvshmemx_smem_amount_t = 0;",
        "pub const NVSHMEMX_SMEM_MINIMUM: nvshmemx_smem_amount_t = 1;",
        "pub const NVSHMEMX_SMEM_BARRIERS_ONLY: nvshmemx_smem_amount_t = 2;",
        "pub const NVSHMEMX_SMEM_AMOUNT_MAX: nvshmemx_smem_amount_t = i32::MAX;",
        "",
        f"// Source config: {config.get('Name', 'NVSHMEM Device Bindings')}",
        "#[device]",
        'unsafe extern "C" {',
    ]


def _host_prelude_lines(config):
    return [
        "// Link this module with libnvshmem_host.",
        *_common_allow_lines(),
        "",
        "pub const UNIQUEID_PADDING: usize = 124;",
        "pub const INIT_ARGS_V2_PADDING: usize = 92;",
        "pub const TEAM_CONFIG_V2_PADDING: usize = 48;",
        "",
        "pub type nvshmem_team_t = i32;",
        "pub type nvshmemx_team_t = nvshmem_team_t;",
        "pub type nvshmemx_team_uniqueid_t = u64;",
        "pub type cudaStream_t = *mut core::ffi::c_void;",
        "pub type CUmodule = *mut core::ffi::c_void;",
        "pub type CUlibrary = *mut core::ffi::c_void;",
        "",
        "#[repr(transparent)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmem_half(pub u16);",
        "",
        "#[repr(transparent)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmem_bfloat16(pub u16);",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmemx_uniqueid_t {",
        "    pub version: i32,",
        "    pub internal: [core::ffi::c_char; UNIQUEID_PADDING],",
        "}",
        "",
        "pub type nvshmemx_uniqueid_v1 = nvshmemx_uniqueid_t;",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmemx_uniqueid_args_t {",
        "    pub version: i32,",
        "    pub id: *mut nvshmemx_uniqueid_t,",
        "    pub myrank: i32,",
        "    pub nranks: i32,",
        "}",
        "",
        "pub type nvshmemx_uniqueid_args_v1 = nvshmemx_uniqueid_args_t;",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmemx_init_args_t {",
        "    pub version: i32,",
        "    pub uid_args: nvshmemx_uniqueid_args_t,",
        "    pub cuda_device_id: i32,",
        "    pub content: [core::ffi::c_char; INIT_ARGS_V2_PADDING],",
        "}",
        "",
        "pub type nvshmemx_init_args_v1 = nvshmemx_init_args_t;",
        "pub type nvshmemx_init_args_v2 = nvshmemx_init_args_t;",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmemx_init_attr_t {",
        "    pub version: i32,",
        "    pub mpi_comm: *mut core::ffi::c_void,",
        "    pub args: nvshmemx_init_args_t,",
        "}",
        "",
        "pub type nvshmemx_init_attr_v1 = nvshmemx_init_attr_t;",
        "pub type nvshmemx_init_attr_v2 = nvshmemx_init_attr_t;",
        "",
        "#[repr(C)]",
        "#[derive(Clone, Copy)]",
        "pub struct nvshmem_team_config_t {",
        "    pub version: i32,",
        "    pub num_contexts: i32,",
        "    pub uniqueid: nvshmemx_team_uniqueid_t,",
        "    pub padding: [core::ffi::c_char; TEAM_CONFIG_V2_PADDING],",
        "}",
        "",
        "pub type nvshmem_team_config_v1 = nvshmem_team_config_t;",
        "pub type nvshmem_team_config_v2 = nvshmem_team_config_t;",
        "",
        "pub const NVSHMEMX_UNIQUEID_VERSION: i32 =",
        "    (1_i32 << 16) + core::mem::size_of::<nvshmemx_uniqueid_t>() as i32;",
        "pub const NVSHMEMX_UNIQUEID_ARGS_VERSION: i32 =",
        "    (1_i32 << 16) + core::mem::size_of::<nvshmemx_uniqueid_args_t>() as i32;",
        "pub const NVSHMEM_INIT_ARGS_V2_IDENTIFIER: i32 =",
        "    (2_i32 << 16) + core::mem::size_of::<nvshmemx_init_args_t>() as i32;",
        "pub const NVSHMEM_INIT_ATTR_V2_IDENTIFIER: i32 =",
        "    (2_i32 << 16) + core::mem::size_of::<nvshmemx_init_attr_t>() as i32;",
        "pub const NVSHMEMI_TEAM_CONFIG_VERSION_2_IDENTIFIER: i32 =",
        "    (2_i32 << 16) + core::mem::size_of::<nvshmem_team_config_t>() as i32;",
        "",
        "pub const NVSHMEMX_INIT_THREAD_PES: u32 = 1;",
        "pub const NVSHMEMX_INIT_WITH_MPI_COMM: u32 = 1 << 1;",
        "pub const NVSHMEMX_INIT_WITH_SHMEM: u32 = 1 << 2;",
        "pub const NVSHMEMX_INIT_WITH_UNIQUEID: u32 = 1 << 3;",
        "",
        "pub const NVSHMEM_TEAM_INVALID: nvshmem_team_t = -1;",
        "pub const NVSHMEM_TEAM_WORLD: nvshmem_team_t = 0;",
        "pub const NVSHMEM_TEAM_SHARED: nvshmem_team_t = 1;",
        "pub const NVSHMEMX_TEAM_NODE: nvshmem_team_t = 2;",
        "pub const NVSHMEMX_TEAM_SAME_MYPE_NODE: nvshmem_team_t = 3;",
        "pub const NVSHMEM_TEAM_MC_SHARED: nvshmem_team_t = 6;",
        "pub const NVSHMEM_TEAM_CONFIG_MASK_NUM_CONTEXTS: i64 = 0x0000000000000001;",
        "pub const NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID: i64 = 0x0000000000000002;",
        "",
        "pub const NVSHMEM_CMP_EQ: i32 = 0;",
        "pub const NVSHMEM_CMP_NE: i32 = 1;",
        "pub const NVSHMEM_CMP_GT: i32 = 2;",
        "pub const NVSHMEM_CMP_LE: i32 = 3;",
        "pub const NVSHMEM_CMP_LT: i32 = 4;",
        "pub const NVSHMEM_CMP_GE: i32 = 5;",
        "",
        "pub const NVSHMEM_SIGNAL_SET: i32 = 9;",
        "pub const NVSHMEM_SIGNAL_ADD: i32 = 10;",
        "",
        "pub const NVSHMEM_THREAD_SINGLE: i32 = 0;",
        "pub const NVSHMEM_THREAD_FUNNELED: i32 = 1;",
        "pub const NVSHMEM_THREAD_SERIALIZED: i32 = 2;",
        "pub const NVSHMEM_THREAD_MULTIPLE: i32 = 3;",
        "",
        "pub const NVSHMEM_STATUS_NOT_INITIALIZED: i32 = 0;",
        "pub const NVSHMEM_STATUS_IS_BOOTSTRAPPED: i32 = 1;",
        "pub const NVSHMEM_STATUS_IS_INITIALIZED: i32 = 2;",
        "pub const NVSHMEM_STATUS_LIMITED_MPG: i32 = 3;",
        "pub const NVSHMEM_STATUS_FULL_MPG: i32 = 4;",
        "pub const NVSHMEM_STATUS_INVALID: i32 = i32::MAX;",
        "",
        "impl Default for nvshmemx_uniqueid_t {",
        "    fn default() -> Self {",
        "        Self {",
        "            version: NVSHMEMX_UNIQUEID_VERSION,",
        "            internal: [0; UNIQUEID_PADDING],",
        "        }",
        "    }",
        "}",
        "",
        "impl Default for nvshmemx_uniqueid_args_t {",
        "    fn default() -> Self {",
        "        Self {",
        "            version: NVSHMEMX_UNIQUEID_ARGS_VERSION,",
        "            id: core::ptr::null_mut(),",
        "            myrank: -1,",
        "            nranks: -1,",
        "        }",
        "    }",
        "}",
        "",
        "impl Default for nvshmemx_init_args_t {",
        "    fn default() -> Self {",
        "        Self {",
        "            version: NVSHMEM_INIT_ARGS_V2_IDENTIFIER,",
        "            uid_args: nvshmemx_uniqueid_args_t::default(),",
        "            cuda_device_id: -1,",
        "            content: [0; INIT_ARGS_V2_PADDING],",
        "        }",
        "    }",
        "}",
        "",
        "impl Default for nvshmemx_init_attr_t {",
        "    fn default() -> Self {",
        "        Self {",
        "            version: NVSHMEM_INIT_ATTR_V2_IDENTIFIER,",
        "            mpi_comm: core::ptr::null_mut(),",
        "            args: nvshmemx_init_args_t::default(),",
        "        }",
        "    }",
        "}",
        "",
        "impl Default for nvshmem_team_config_t {",
        "    fn default() -> Self {",
        "        Self {",
        "            version: NVSHMEMI_TEAM_CONFIG_VERSION_2_IDENTIFIER,",
        "            num_contexts: -1,",
        "            uniqueid: u64::MAX,",
        "            padding: [0; TEAM_CONFIG_V2_PADDING],",
        "        }",
        "    }",
        "}",
        "",
        f"// Source config: {config.get('Name', 'NVSHMEM Host Bindings')}",
        "#[link(name = \"nvshmem_host\")]",
        'unsafe extern "C" {',
    ]


def _render_device_bindings(decls, config):
    lines = _common_header_lines()
    lines.extend(_device_prelude_lines(config))

    generated_count = 0
    for function in decls.functions:
        lines.append(_generate_function_decl(function))
        generated_count += 1

    lines.extend(["}", ""])
    return "\n".join(lines), generated_count


def _render_host_bindings(decls, config):
    lines = _common_header_lines()
    lines.extend(_host_prelude_lines(config))

    generated_count = 0
    for function in _select_host_functions(decls):
        lines.append(_generate_function_decl(function))
        generated_count += 1
    lines.extend(["}", ""])
    return "\n".join(lines), generated_count


def _render_host_api(decls, config):
    lines = _common_header_lines()
    lines.extend([
        "// Prefix-stripped Rust wrappers around the raw NVSHMEM host FFI.",
        "",
        "use crate::sys;",
        "use crate::sys::*;",
        "",
        f"// Source config: {config.get('Name', 'NVSHMEM Host Bindings')}",
    ])

    generated_count = 0
    seen = set()
    for function in _select_host_functions(decls):
        api_name = _rust_api_name(function.name)
        if api_name in seen:
            raise RuntimeError(
                "Host Rust API wrapper name collision after prefix stripping: "
                f"{api_name}")
        lines.append("")
        lines.append(_generate_host_api_wrapper(function))
        generated_count += 1
        seen.add(api_name)

    lines.append("")
    return "\n".join(lines), generated_count


def generate(config_path, output_path, binding_kind):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    decls = parse_declarations_from_source(
        config["Entry Point"],
        config["File List"],
        cudatoolkit_include_dirs=config.get("CUDA Toolkit Include Paths", []),
        additional_includes=config["Clang Include Paths"],
        compute_capability=config["GPU Arch"][0],
        bypass_parse_error=True,
    )

    if binding_kind == "device":
        rendered, generated_count = _render_device_bindings(decls, config)
        binding_name = "CUDA-Oxide Rust NVSHMEM device"
    elif binding_kind == "host":
        rendered, generated_count = _render_host_bindings(decls, config)
        binding_name = "Rust NVSHMEM host"
    elif binding_kind == "host-api":
        rendered, generated_count = _render_host_api(decls, config)
        binding_name = "Rust NVSHMEM host API"
    else:
        raise ValueError(f"Unsupported Rust binding kind: {binding_kind}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"Generated {generated_count} {binding_name} bindings to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA-Oxide Rust bindings for NVSHMEM device APIs"
    )
    parser.add_argument("--config-path",
                        default=os.path.join(os.path.dirname(__file__),
                                             "config_nvshmem.yml"))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--binding-kind",
                        choices=("device", "host", "host-api"),
                        default="device")
    args = parser.parse_args()

    generate(args.config_path, args.output_path, args.binding_kind)


if __name__ == "__main__":
    main()
