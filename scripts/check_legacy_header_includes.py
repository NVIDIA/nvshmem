#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject NVSHMEM source includes of compatibility forwarding headers."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


FORWARDING_MARKER = "Compatibility forwarding header"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".in",
    ".py",
}
EXCLUDED_PREFIXES = ("externals/",)
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')


def find_forwarding_headers(include_root: Path) -> dict[str, str]:
    forwarding_headers = {}
    for header in sorted(path for path in include_root.rglob("*") if path.is_file()):
        contents = header.read_text(encoding="utf-8")
        if FORWARDING_MARKER not in contents:
            continue

        includes = [
            match.group(1)
            for line in contents.splitlines()
            if (match := INCLUDE_PATTERN.match(line))
        ]
        if len(includes) != 1:
            raise RuntimeError(
                f"{header}: expected one canonical include in compatibility forwarding header"
            )
        forwarding_headers[header.relative_to(include_root).as_posix()] = includes[0]
    return forwarding_headers


def tracked_source_files(repository_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    source_files = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        if relative_path.startswith(EXCLUDED_PREFIXES):
            continue
        if Path(relative_path).suffix in SOURCE_SUFFIXES:
            source_files.append(repository_root / relative_path)
    return source_files


def find_legacy_includes(contents: str, forwarding_headers: dict[str, str]):
    for line_number, line in enumerate(contents.splitlines(), start=1):
        match = INCLUDE_PATTERN.match(line)
        if match and match.group(1) in forwarding_headers:
            legacy_header = match.group(1)
            yield line_number, legacy_header, forwarding_headers[legacy_header]


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    include_root = repository_root / "src" / "include"

    try:
        forwarding_headers = find_forwarding_headers(include_root)
        violations = []
        for source in tracked_source_files(repository_root):
            contents = source.read_text(encoding="utf-8", errors="surrogateescape")
            for line_number, legacy_header, canonical_header in find_legacy_includes(
                contents, forwarding_headers
            ):
                violations.append(
                    (
                        source.relative_to(repository_root),
                        line_number,
                        legacy_header,
                        canonical_header,
                    )
                )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"legacy-header include check failed: {error}", file=sys.stderr)
        return 2

    if not violations:
        print(
            f"No tracked source files include the {len(forwarding_headers)} "
            "compatibility forwarding headers."
        )
        return 0

    print("NVSHMEM source files must include canonical headers instead of compatibility headers:")
    for source, line_number, legacy_header, canonical_header in violations:
        print(f'{source}:{line_number}: legacy include "{legacy_header}"')
        print(f'  use "{canonical_header}"')
    return 1


if __name__ == "__main__":
    sys.exit(main())
