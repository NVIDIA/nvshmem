# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Foreign calls into NVSHMEM's cuTile TileLib."""

import hashlib
import os

import cuda.tile as ct
from cuda import tile_preview as tp

from nvshmem.core import DeviceLibLanguage, find_device_bitcode_library, get_version

# tileiras runs the TileLib's linkage provider, which links the shims against
# this library.
_device_ltoir = find_device_bitcode_library(language=DeviceLibLanguage.LTOIR)
os.environ["NVSHMEM_DEVICE_LTOIR"] = _device_ltoir
_tilelib = find_device_bitcode_library(language=DeviceLibLanguage.TILELIB)

# cuTile reuses a compiled kernel while a TileLib's path and version are
# unchanged, so the version also names the contents of the TileLib and of the
# library it links: rebuilding either in place recompiles the kernels.
_contents = hashlib.sha256()
for _path in (_tilelib, _device_ltoir):
    with open(_path, "rb") as _file:
        _contents.update(_file.read())
TILELIB = tp.Tilelib(path=_tilelib, version=f"{get_version().libnvshmem_version}:{_contents.hexdigest()}")


def foreign_call(symbol, inputs=(), output_types=(), *, constant_params=()):
    """Call ``symbol`` in the NVSHMEM TileLib."""
    return tp.foreign_call(tilelibs=(ct.static_eval(TILELIB), ),
                           symbol_name=symbol,
                           constant_params=constant_params,
                           inputs=inputs,
                           output_types=output_types)
