# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the TileLib bindings, the generated bindings, and the adapter against fake cuTile modules."""

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[4] / "nvshmem4py" / "nvshmem"
BINDINGS = PACKAGE / "bindings" / "device" / "tile" / "__init__.py"
CORE = PACKAGE / "core" / "device" / "tile"
DTYPES = ("bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "bfloat16",
          "float32", "float64", "tfloat32", "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")


class Array:
    """A fake ``ct.Array``."""

    def __init__(self, dtype="float32"):
        self.dtype = dtype
        self.shape = (8, )
        self.strides = (1, )
        self.foreign_pointer = types.SimpleNamespace(array=self, dtype=("pointer", dtype))

    @staticmethod
    def from_foreign(pointer, shape, strides):
        return ("view", pointer, shape, strides)


class Tile:
    """A fake shared-memory ``ct.Tile``."""

    def __init__(self, dtype="float32", shape=(16, )):
        self.dtype = dtype
        self.shape = shape


def _static_assert(condition, message=None):
    if not condition:
        raise TypeError(message)


def install_modules(monkeypatch, modules: dict[str, types.ModuleType]) -> None:
    """Install fake modules and their packages, with a fake ``cuda.tile`` whose dtypes are their names."""
    ct = modules.setdefault("cuda.tile", types.ModuleType("cuda.tile"))
    for dtype in DTYPES:
        setattr(ct, dtype, dtype)
    ct.Array, ct.Tile, ct.static_eval, ct.static_assert = Array, Tile, lambda expression: expression, _static_assert
    for name, module in list(modules.items()):
        parent, _, child = name.rpartition(".")
        while parent:
            package = modules.setdefault(parent, types.ModuleType(parent))
            package.__path__ = getattr(package, "__path__", [])
            setattr(package, child, module)
            module, (parent, _, child) = package, parent.rpartition(".")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def load(monkeypatch, path: Path, name: str) -> types.ModuleType:
    """Execute ``path`` as the module ``name``."""
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def test_bindings_call_the_device_tilelib_and_name_the_device_library(monkeypatch, tmp_path):
    calls = []

    class Tilelib:

        def __init__(self, *, path, version):
            self.path = path
            self.version = version

    preview = types.ModuleType("cuda.tile_preview")
    preview.Tilelib = Tilelib
    preview.foreign_call = lambda **arguments: calls.append(arguments) or "result"
    core = types.ModuleType("nvshmem.core")
    core.DeviceLibLanguage = types.SimpleNamespace(LTOIR="ltoir", TILELIB="tilelib")
    core.find_device_bitcode_library = lambda *, language: str(tmp_path / f"libnvshmem_device.{language}")
    (tmp_path / "libnvshmem_device.tilelib").write_bytes(b"tilelib")
    (tmp_path / "libnvshmem_device.ltoir").write_bytes(b"ltoir")
    core.get_version = lambda: types.SimpleNamespace(libnvshmem_version="3.9.0")
    install_modules(monkeypatch, {"cuda.tile_preview": preview, "nvshmem.core": core})
    # Record the variable, so the value the bindings set is undone.
    monkeypatch.setenv("NVSHMEM_DEVICE_LTOIR", "")
    bindings = load(monkeypatch, BINDINGS, "nvshmem.bindings.device.tile")

    # The release and the contents of the TileLib and the library it links key cuTile's cache.
    assert bindings.TILELIB.path == str(tmp_path / "libnvshmem_device.tilelib")
    assert bindings.TILELIB.version == "3.9.0:" + hashlib.sha256(b"tilelibltoir").hexdigest()
    (tmp_path / "libnvshmem_device.ltoir").write_bytes(b"rebuilt ltoir")
    assert load(monkeypatch, BINDINGS, "nvshmem.bindings.device.tile").TILELIB.version != bindings.TILELIB.version
    # The linkage provider links the shims against the library this names.
    assert os.environ["NVSHMEM_DEVICE_LTOIR"] == str(tmp_path / "libnvshmem_device.ltoir")
    assert bindings.foreign_call("nvshmem_quiet", ("status", ), ("i32", )) == "result"
    assert calls == [{
        "tilelibs": (bindings.TILELIB, ),
        "symbol_name": "nvshmem_quiet",
        "constant_params": (),
        "inputs": ("status", ),
        "output_types": ("i32", ),
    }]


def test_public_python_path_never_extracts_raw_pointers():
    sources = BINDINGS.read_text(encoding="utf-8") + (CORE / "_adapter.py").read_text(encoding="utf-8")
    for forbidden in (".to_raw()", "broadcast_to", "foreign_decl", "@extern_impl", "link_objects"):
        assert forbidden not in sources


def test_raw_bindings_pass_foreign_pointers_and_rebuild_array_views(generated_tilelib, monkeypatch):
    calls = []

    def foreign_call(symbol, inputs=(), output_types=(), *, constant_params=()):
        calls.append((symbol, inputs, output_types, constant_params))
        return ("returned-" + symbol, )

    ct = types.ModuleType("cuda.tile")
    ct.zeros = lambda shape, dtype: ("zero", dtype)
    bindings = types.ModuleType("nvshmem.bindings.device.tile")
    bindings.foreign_call = foreign_call
    adapter = types.ModuleType("nvshmem.core.device.tile._adapter")
    adapter.require_array = lambda operand, name: operand
    adapter.require_rank_one_tile = lambda operand, name: operand.shape[0]
    install_modules(monkeypatch, {
        "cuda.tile": ct,
        "nvshmem.bindings.device.tile": bindings,
        "nvshmem.core.device.tile._adapter": adapter,
    })
    generated = load(monkeypatch, generated_tilelib / "python" / "generated.py", "nvshmem.core.device.tile.generated")

    dest, source, tile = Array(), Array(), Tile()
    bytes_dest, bytes_source = Array("int8"), Array("uint8")
    status = ("zero", "int32")
    generated.nvshmem_float_put(dest, source, 4, 1)
    generated.nvshmemx_float_put_block(dest, tile, 4, 1)
    generated.nvshmem_putmem(dest, source, 16, 1)
    generated.nvshmem_putmem(bytes_dest, bytes_source, 16, 1)
    with pytest.raises(TypeError, match="source has dtype float16, but dest has dtype float32"):
        generated.nvshmem_putmem(dest, Array("float16"), 16, 1)
    with pytest.raises(TypeError, match="ptr has dtype bool_"):
        generated.nvshmem_ptr(Array("bool_"), 1)
    # A returned pointer becomes a view with its reference Array's shape.
    view = generated.nvshmem_ptr(source, 1)
    assert view == ("view", "returned-nvshmem_ptr__f32", source.shape, source.strides)
    assert calls == [
        ("nvshmem_float_put", (dest.foreign_pointer, source.foreign_pointer, 4, 1, status), ("int32", ), ()),
        # A Tile passes as itself, with its extent as a declaration parameter.
        ("nvshmemx_float_put_block__tile_source", (dest.foreign_pointer, tile, 4, 1, status), ("int32", ), (16, )),
        # Erased ``void *`` operands select the declaration for their dtype;
        # signed and unsigned share Tile IR's signless integers.
        ("nvshmem_putmem__f32", (dest.foreign_pointer, source.foreign_pointer, 16, 1, status), ("int32", ), ()),
        ("nvshmem_putmem__i8", (bytes_dest.foreign_pointer, bytes_source.foreign_pointer, 16, 1, status), ("int32", ),
         ()),
        ("nvshmem_ptr__f32", (source.foreign_pointer, 1, source.foreign_pointer), (source.foreign_pointer.dtype, ), ()),
    ]


def test_top_level_api_infers_counts_converts_scalars_and_returns_destinations(generated_tilelib, monkeypatch):
    calls = []
    raw = types.ModuleType("nvshmem.core.device.tile.generated")
    raw.__getattr__ = lambda name: lambda *args: calls.append((name, args)) or name
    raw.nvshmemx_float_get_block = lambda *args: calls.append(("nvshmemx_float_get_block", args)) or (("updated", ), 0)
    adapter = types.ModuleType("nvshmem.core.device.tile._adapter")
    adapter.element_count = lambda operand: ("elements", operand)
    adapter.byte_count = lambda operand: ("bytes", operand)
    adapter.divide_count = lambda count, divisor: ("quotient", count, divisor)
    adapter.as_scalar = lambda value, dtype: (value, dtype)
    modules = {"nvshmem.core.device.tile.generated": raw, "nvshmem.core.device.tile._adapter": adapter}
    for module, name in (("direct", "Teams"), ("mem", "get_peer_array")):
        exports = modules[f"nvshmem.core.device.tile._{module}_exports"] = types.ModuleType(module)
        setattr(exports, name, f"hand-written {name}")
        exports.__all__ = [name]
    install_modules(monkeypatch, modules)
    direct, rma, collective, amo, mem = (load(monkeypatch, generated_tilelib / "python" / f"{name}.py",
                                              f"nvshmem.core.device.tile.{name}")
                                         for name in ("direct", "rma", "collective", "amo", "mem"))
    for module in (direct, rma, collective, amo, mem):
        assert all(hasattr(module, name) for name in module.__all__)
    # A module's hand-written exports are part of its API.
    assert "Teams" in direct.__all__ and mem.get_peer_array == "hand-written get_peer_array"

    dst, src, tile = Array(), Array(), Tile()
    assert rma.put_block(dst, tile, 1) is None
    assert rma.put_nbi_block(dst=dst, src=src, pe=1) is None
    assert collective.broadcast_block(team=0, dst=dst, src=src, root=2) == "nvshmemx_float_broadcast_block"
    assert direct.quiet() is None
    # A get returns its destination: a Tile updated, an Array as it is.
    assert rma.get_block(tile, src, 1) == ("updated", )
    assert rma.get_warp(dst, src, 1) is dst
    # A native result stays the function's result.
    assert rma.g(src, 1) == "nvshmem_float_g"
    assert amo.atomic_fetch(dst, 1) == "nvshmem_float_atomic_fetch"
    # NVSHMEM counts a void * operand in bytes, whatever the parameter's name.
    assert rma.putmem_signal_block(dst, src, dst, 1, 0, 1) is None
    # release_smem takes the Tile that give_smem registered, but NVSHMEM does not.
    assert mem.release_smem(tile) is None
    pe, root, team = (1, "int32"), (2, "int32"), (0, "int32")
    assert calls == [
        ("nvshmemx_float_put_block", (dst, tile, ("elements", tile), pe)),
        ("nvshmemx_float_put_nbi_block", (dst, src, ("elements", src), pe)),
        ("nvshmemx_float_broadcast_block", (team, dst, src, ("elements", src), root)),
        ("nvshmem_quiet", ()),
        ("nvshmemx_float_get_block", (tile, src, ("elements", tile), pe)),
        ("nvshmemx_float_get_warp", (dst, src, ("elements", dst), pe)),
        ("nvshmem_float_g", (src, pe)),
        ("nvshmem_float_atomic_fetch", (dst, pe)),
        ("nvshmemx_putmem_signal_block", (dst, src, ("bytes", src), dst, (1, "uint64"), (0, "int32"), pe)),
        ("nvshmemx_release_smem", (tile, )),
    ]


def test_scalar_arguments_become_scalars_of_the_declared_dtype(monkeypatch):
    """Numbers and enum members fill a new scalar; scalar Tiles convert."""
    ct = types.ModuleType("cuda.tile")
    ct.full = lambda shape, value, dtype: ("full", shape, value, dtype)
    ct.astype = lambda value, dtype: ("astype", value, dtype)
    install_modules(monkeypatch, {"cuda.tile": ct})
    adapter = load(monkeypatch, CORE / "_adapter.py", "nvshmem.core.device.tile._adapter")
    tile = Tile()
    assert adapter.as_scalar(3, "int32") == ("full", (), 3, "int32")
    assert adapter.as_scalar(tile, "uint64") == ("astype", tile, "uint64")


def test_package_reexports_every_generated_module(generated_tilelib):
    package_init = (CORE / "__init__.py").read_text(encoding="utf-8")
    modules = [path.stem for path in (generated_tilelib / "python").glob("*.py")]
    # The generator writes Python only into the package, as the CuTe generator does.
    assert modules and not list(generated_tilelib.glob("*.py"))
    for module in modules:
        assert f"from .{module} import *" in package_init and f"*{module}.__all__" in package_init
