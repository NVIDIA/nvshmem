# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for helper-owned CUDA object lifetime management."""

import types

import pytest

import nvshmem.core._internal_tracking as tracking
import nvshmem.core.init_fini as init_fini
from nvshmem.core.nvshmem_types import NvshmemError, NvshmemKernelObject


@pytest.fixture(autouse=True)
def reset_helper_libraries():
    previous_status = tracking._is_initialized["status"]
    tracking._helper_library_references.clear()
    tracking._helper_module_references.clear()
    yield
    tracking._helper_library_references.clear()
    tracking._helper_module_references.clear()
    tracking._is_initialized["status"] = previous_status


def _tracked_library(handle):
    library = NvshmemKernelObject.from_handle(handle)
    owner = object()
    tracking._helper_library_references[library] = owner
    return library, owner


def test_register_cute_library_tracks_executor(monkeypatch):
    pytest.importorskip("cutlass.cute")
    import nvshmem.core.interop.cute as cute_interop

    executor = types.SimpleNamespace(jit_module=types.SimpleNamespace(cuda_library=[0x1234]))
    calls = []
    monkeypatch.setattr(cute_interop.nvshmem.core, "library_init", calls.append)

    library = cute_interop.register_cute_library(executor)

    assert calls == [library]
    assert tracking._helper_library_references == {library: executor}


def test_register_cute_module_tracks_executor(monkeypatch):
    pytest.importorskip("cutlass.cute")
    import nvshmem.core.interop.cute as cute_interop

    executor = object()
    calls = []
    monkeypatch.setattr(cute_interop.nvshmem.core, "module_init", calls.append)

    module = cute_interop.register_cute_module(executor, 0x5678)

    assert calls == [module]
    assert tracking._helper_module_references == {module: executor}


def test_register_cute_library_does_not_track_failed_init(monkeypatch):
    pytest.importorskip("cutlass.cute")
    import nvshmem.core.interop.cute as cute_interop

    executor = types.SimpleNamespace(jit_module=types.SimpleNamespace(cuda_library=[0x1234]))

    def fail_init(library):
        raise NvshmemError("injected library initialization failure")

    monkeypatch.setattr(cute_interop.nvshmem.core, "library_init", fail_init)

    with pytest.raises(NvshmemError):
        cute_interop.register_cute_library(executor)

    assert tracking._helper_library_references == {}


def test_generic_library_init_is_not_tracked(monkeypatch):
    library = NvshmemKernelObject.from_handle(0x1234)
    monkeypatch.setattr(init_fini.bindings, "culibrary_init", lambda handle: None)

    init_fini.library_init(library)

    assert tracking._helper_library_references == {}


def test_explicit_library_finalize_releases_helper_owner(monkeypatch):
    library, owner = _tracked_library(0x1234)
    calls = []
    monkeypatch.setattr(init_fini.bindings, "culibrary_finalize", lambda handle: calls.append(int(handle)))

    init_fini.library_finalize(library)

    assert calls == [library.handle]
    assert owner not in tracking._helper_library_references.values()
    assert tracking._helper_library_references == {}


def test_finalize_drains_helper_cuda_objects_before_host_teardown(monkeypatch):
    module = NvshmemKernelObject.from_handle(0x9ABC)
    module_owner = object()
    tracking._helper_module_references[module] = module_owner
    first, first_owner = _tracked_library(0x1234)
    second, second_owner = _tracked_library(0x5678)
    calls = []
    monkeypatch.setattr(init_fini.bindings, "cumodule_finalize", lambda handle: calls.append(("module", int(handle))))
    monkeypatch.setattr(
        init_fini.bindings,
        "culibrary_finalize",
        lambda handle: calls.append(("library", int(handle))),
    )
    monkeypatch.setattr(init_fini.memory, "_free_all_buffers", lambda: calls.append(("buffers", None)))
    monkeypatch.setattr(init_fini.bindings, "hostlib_finalize", lambda: calls.append(("host", None)))

    init_fini.finalize()

    assert calls == [
        ("module", module.handle),
        ("library", second.handle),
        ("library", first.handle),
        ("buffers", None),
        ("host", None),
    ]
    assert module_owner not in tracking._helper_module_references.values()
    assert first_owner not in tracking._helper_library_references.values()
    assert second_owner not in tracking._helper_library_references.values()
    assert tracking._helper_module_references == {}
    assert tracking._helper_library_references == {}


def test_finalize_retains_failed_library_and_skips_host_teardown(monkeypatch):
    successful, successful_owner = _tracked_library(0x1234)
    failed, failed_owner = _tracked_library(0x5678)
    calls = []

    def finalize_library(handle):
        calls.append(("library", int(handle)))
        return 1 if int(handle) == failed.handle else None

    monkeypatch.setattr(init_fini.bindings, "culibrary_finalize", finalize_library)
    monkeypatch.setattr(init_fini.memory, "_free_all_buffers", lambda: calls.append(("buffers", None)))
    monkeypatch.setattr(init_fini.bindings, "hostlib_finalize", lambda: calls.append(("host", None)))

    with pytest.raises(NvshmemError):
        init_fini.finalize()

    assert calls == [("library", failed.handle), ("library", successful.handle)]
    assert tracking._helper_library_references == {failed: failed_owner}
    assert successful_owner not in tracking._helper_library_references.values()
