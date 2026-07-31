# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Numbast while retaining only explicitly configured extra enums."""

import argparse
import os
import sys

import yaml

from numbast.tools import static_binding_generator as static_generator

_parse_declarations = static_generator.parse_declarations_from_source


def _load_extra_enum_sources(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Numbast config must be a mapping")
    extra_sources = config.get("Extra Enum Sources", [])
    if not isinstance(extra_sources, list):
        raise ValueError("Extra Enum Sources must be a list")

    parsed_sources = []
    for source in extra_sources:
        path = source.get("File")
        names = source.get("Names")
        if not isinstance(path, str) or not os.path.isfile(path):
            raise ValueError(f"Extra enum source does not exist: {path}")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Extra enum source has invalid Names: {source}")
        parsed_sources.append((path, names))

    return parsed_sources


def _run_static_generator(config_path):
    extra_sources = _load_extra_enum_sources(config_path)
    if not extra_sources:
        static_generator.static_binding_generator()
        return

    def parse_declarations(source_file_path, files_to_retain, *args, **kwargs):
        """Add configured enums without expanding the normal generated API surface."""

        declarations = _parse_declarations(source_file_path, files_to_retain, *args, **kwargs)
        existing_names = {enum.name for enum in declarations.enums}

        for header, enum_names in extra_sources:
            extra_declarations = _parse_declarations(source_file_path, [header], *args, **kwargs)
            extra_enums = {enum.name: enum for enum in extra_declarations.enums}
            missing = set(enum_names) - extra_enums.keys()
            if missing:
                raise ValueError(f"Missing configured enums in {header}: {sorted(missing)}")

            for name in enum_names:
                if name not in existing_names:
                    declarations.enums.append(extra_enums[name])
                    existing_names.add(name)

        return declarations

    static_generator.parse_declarations_from_source = parse_declarations
    try:
        static_generator.static_binding_generator()
    finally:
        static_generator.parse_declarations_from_source = _parse_declarations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg-path")
    arguments, _ = parser.parse_known_args(sys.argv[1:])
    if arguments.cfg_path:
        _run_static_generator(arguments.cfg_path)
    else:
        static_generator.static_binding_generator()
