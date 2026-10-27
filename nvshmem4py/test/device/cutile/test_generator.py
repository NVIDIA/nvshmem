# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The generated TileLib files cover the device API and agree with one another."""

import json
import re
from pathlib import Path

import pytest

DEVICE_INCLUDE = Path(__file__).resolve().parents[4] / "src" / "include"


def _header_functions() -> set[str]:
    """The public device functions, parsed independently of the generator."""
    ast_canopy = pytest.importorskip("ast_canopy")
    headers = ("nvshmem_coll_defines.cuh", "nvshmem_defines.h", "nvshmemx_coll_defines.cuh", "nvshmemx_defines.h")
    parsed = ast_canopy.parse_declarations_from_source(str(DEVICE_INCLUDE / "nvshmem.h"),
                                                       [str(DEVICE_INCLUDE / "device" / header) for header in headers],
                                                       additional_includes=[str(DEVICE_INCLUDE)],
                                                       compute_capability="sm_90",
                                                       bypass_parse_error=True)
    return {
        function.name
        for function in parsed.functions
        if function.name.startswith(("nvshmem_", "nvshmemx_")) and "device" in getattr(function, "attributes", ())
    }


def _shims(output_dir) -> dict[str, tuple[str, str]]:
    """Each shim's parameter list and body."""
    return {
        name: (parameters, body)
        for name, parameters, body in re.findall(r'void plugin_(\w+)\(\n(.*?)\n\) \{\n(.*?)\n\}\n', (
            output_dir / "nvshmem_tile_ffi.cu").read_text(), re.S)
    }


def _declaration(interface: str, symbol: str) -> str:
    """A declaration's text after its symbol."""
    return re.search(rf"preview\$func extern @{symbol}[<(](.*?)(?:preview\$func|\Z)", interface, re.S).group(1)


def test_every_public_device_function_has_a_binding(generated_tilelib):
    generated = (generated_tilelib / "python" / "generated.py").read_text()
    assert set(re.findall(r"(?m)^def (nvshmemx?_\w+)\(", generated)) == _header_functions()


def test_declarations_implementations_and_shims_agree(generated_tilelib):
    interface = (generated_tilelib / "nvshmem_tile_interface.mlir").read_text()
    catalog = json.loads((generated_tilelib / "nvshmem_tile_provider.json").read_text())
    shims = _shims(generated_tilelib)

    assert set(re.findall(r"preview\$func extern @(\w+)", interface)) == set(catalog["declarations"])
    for symbol, module in catalog["declarations"].items():
        assert f"implements @{symbol}" in module and "preview$func" not in module
        # A per-element declaration of an erased Array operand shares its shim.
        assert symbol in shims or symbol.rpartition("__")[0] in shims
    # The provider forwards each specialization with its shim's parameter widths.
    assert set(catalog["abi"]) == {f"plugin_{name}" for name in shims}
    for name, (parameters, _) in shims.items():
        assert len(catalog["abi"][f"plugin_{name}"]) == len(parameters.split(",\n"))


def test_calls_keep_their_scope_and_tiles_replace_local_data_at_cta_scope(generated_tilelib):
    interface = (generated_tilelib / "nvshmem_tile_interface.mlir").read_text()
    modules = json.loads((generated_tilelib / "nvshmem_tile_provider.json").read_text())["declarations"]
    shims = _shims(generated_tilelib)

    for symbol, synchronization in (("nvshmem_float_put", "none"), ("nvshmemx_float_put_warp", "warp"),
                                    ("nvshmemx_float_put_block", "cta")):
        assert f"synchronization<{synchronization}>" in modules[symbol]
    # A warpgroup can be part of a larger CTA, so it must not wait on the CTA.
    warpgroup = shims["nvshmemx_barrier_warpgroup"][1]
    assert "(threadIdx.x & 127) == 0" in warpgroup and "__syncthreads" not in warpgroup

    for tile in ("nvshmemx_float_put_block__tile_source", "nvshmemx_float_get_block__tile_dest",
                 "nvshmemx_float_broadcast_block__tile_source", "nvshmemx_float_alltoall_block__tile_source",
                 "nvshmemx_putmem_block__tile_source__f32", "nvshmemx_give_smem__tile_smem",
                 "nvshmemx_release_smem__tile_smem"):
        assert tile in modules
    # A reduction may read its source through the symmetric heap, so the source is an Array.
    for no_tile in ("nvshmemx_float_put_warp__tile_", "nvshmemx_float_sum_reduce_block__tile_",
                    "nvshmemx_float_sum_reducescatter_block__tile_", "nvshmemx_double2_maxloc_reduce_block__tile_"):
        assert not any(symbol.startswith(no_tile) for symbol in modules)
    # The released Tile only keeps give_smem's storage allocated; NVSHMEM's release takes no operand.
    assert "nvshmemx_release_smem();" in shims["nvshmemx_release_smem__tile_smem"][1]
    # An NBI call with a Tile runs its blocking counterpart, so the Tile outlives the call.
    assert "nvshmemx_float_put_block(" in shims["nvshmemx_float_put_nbi_block__tile_source"][1]
    functions = set(
        re.findall(r"(?m)^def (nvshmemx?_\w+)\(", (generated_tilelib / "python" / "generated.py").read_text()))
    for name, (_, body) in shims.items():
        if "_nbi" in name and "__tile_" in name:
            assert re.search(r"\b(nvshmemx?_\w+)\(", body).group(1) in functions, name
    # A Tile destination is the last operand, and the call returns it.
    get = _declaration(interface, "nvshmemx_float_get_block__tile_dest")
    assert get.index("%pe") < get.index("%dest") and "access<inout>" in get
    assert "-> (!cuda_tile.tile<[N]xf32>, !cuda_tile.tile<i32>)" in get

    # A ``void *`` Array operand makes one declaration per element type, and a
    # returned pointer shares it.
    ptr = _declaration(interface, "nvshmem_ptr__f32")
    assert "%ptr: !cuda_tile.tile<ptr<f32>>" in ptr and "%_result: !cuda_tile.tile<ptr<f32>>" in ptr
    # cuTile orders every foreign call against memory, so quiet needs no operand.
    assert _declaration(interface, "nvshmem_quiet").lstrip().startswith("%_result")
