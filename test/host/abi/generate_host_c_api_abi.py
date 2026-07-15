#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate and validate the public C API ABI link test."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys


PUBLIC_API_NAME = re.compile(r"\b(nvshmem(?:x|id)?_[A-Za-z0-9_]+)\s*\(")
DEMANGLED_FUNCTION_NAME = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(")
LINE_DIRECTIVE = re.compile(r"^\s*#.*$", re.MULTILINE)


def public_api_symbols(preprocessed: str) -> set[str]:
    """Return public API names after expanding the public headers' macros."""

    symbols = set(PUBLIC_API_NAME.findall(LINE_DIRECTIVE.sub("", preprocessed)))
    if not symbols:
        raise RuntimeError("did not find any public C API declarations")
    return symbols


def preprocess(args: argparse.Namespace) -> str:
    command = [
        args.nvcc,
        "-E",
        "-x",
        "cu",
        f"-std=c++{args.cxx_standard}",
        f"-I{args.include_dir}",
        args.input,
    ]
    # CUDA's headers use #include_next.  Ambient include-path variables can
    # therefore make nvcc select an incompatible math.h before its own headers.
    environment = os.environ.copy()
    for variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        environment.pop(variable, None)
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    if result.returncode:
        sys.stderr.write("failed to preprocess the public NVSHMEM headers:\n")
        sys.stderr.write(" ".join(command) + "\n")
        sys.stderr.write(result.stderr)
        raise RuntimeError("nvcc preprocessing failed")
    return result.stdout


def nm_output(nm: str, *arguments: str) -> str:
    result = subprocess.run([nm, *arguments], capture_output=True, text=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"symbol inspection failed for {arguments[-1]}")
    return result.stdout


def dynamic_function_symbols(nm: str, library: str) -> set[str]:
    """Return raw function names exported by libnvshmem_host.so."""

    symbols: set[str] = set()
    for line in nm_output(nm, "-D", "--defined-only", "--format=posix", library).splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] not in {"T", "W", "I", "i"}:
            continue
        symbols.add(fields[0].split("@", 1)[0])
    return symbols


def mangled_public_definitions(
    nm: str, cxxfilt: str, library: str, public_symbols: set[str]
) -> list[tuple[str, str]]:
    """Return public declarations implemented with C++ linkage in an artifact."""

    mangled = []
    for line in nm_output(nm, "--defined-only", "--format=posix", library).splitlines():
        fields = line.split()
        if fields and fields[0].startswith("_Z"):
            mangled.append(fields[0])
    if not mangled:
        return []

    result = subprocess.run(
        [cxxfilt], input="\n".join(mangled) + "\n", capture_output=True, text=True, check=False
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"failed to demangle symbols from {library}")

    demangled_symbols = result.stdout.splitlines()
    if len(demangled_symbols) != len(mangled):
        raise RuntimeError(f"could not demangle every symbol from {library}")

    violations = []
    for raw, demangled in zip(mangled, demangled_symbols):
        match = DEMANGLED_FUNCTION_NAME.match(demangled)
        if match and match.group(1) in public_symbols:
            violations.append((raw, demangled))
    return violations


def validate_c_linkage(args: argparse.Namespace, public_symbols: set[str]) -> None:
    """Reject public APIs with a C++-mangled implementation in either library."""

    violations = []
    for library in (args.library, args.device_library):
        for raw, demangled in mangled_public_definitions(
            args.nm, args.cxxfilt, library, public_symbols
        ):
            violations.append(f"  {library}: {raw} ({demangled})")
    if violations:
        raise RuntimeError(
            "public C APIs have C++-mangled definitions:\n" + "\n".join(violations)
        )


def generated_source(symbols: list[str]) -> str:
    references = "\n".join(
        "[[maybe_unused]] __attribute__((used)) static auto const "
        f"nvshmem_host_c_api_reference_{index} = &{symbol};"
        for index, symbol in enumerate(symbols)
    )
    return f"""/* This file is generated. Do not edit. */
#include <nvshmem.h>
#include <nvshmemx.h>

namespace {{
{references}
}}  // namespace

int main() {{ return 0; }}
"""


def write_if_changed(path: pathlib.Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == contents:
        return
    path.write_text(contents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--include-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cxx-standard", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device-library", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--cxxfilt", required=True)
    args = parser.parse_args()

    try:
        public_symbols = public_api_symbols(preprocess(args))
        validate_c_linkage(args, public_symbols)
        symbols = sorted(public_symbols & dynamic_function_symbols(args.nm, args.library))
        if not symbols:
            raise RuntimeError("did not find any dynamically exported public C APIs")
        write_if_changed(pathlib.Path(args.output), generated_source(symbols))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"generated {len(symbols)} public C API references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
