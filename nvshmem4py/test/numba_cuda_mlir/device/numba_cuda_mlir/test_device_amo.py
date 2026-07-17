# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from numba_cuda_mlir import cuda
from cuda.bindings import runtime
from cuda.core import Device
from mpi4py import MPI

import nvshmem.core
import nvshmem.core.device.numba_cuda_mlir


def _is_cross_node_job():
    """True when TEAM_WORLD spans more than one NVLink/NVSwitch domain.

    Within a single NVLink/NVSwitch domain device AMOs take the GPU LD/ST
    atomics path and all dtypes are supported. Across nodes the AMO is
    sent to a remote transport (IBRC/UCX/libfabric/...). On IBRC without
    a GDRCopy/CPU-atomics fallback only 8-byte native ADD is wired up;
    other cases call ``NVSHMEMI_ERROR_EXIT`` which aborts the MPI job
    and cannot be caught from Python.
    """
    return nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) < nvshmem.core.n_pes()


def _has_native_peer_atomics():
    """Return whether this rank can issue native atomics to its local peers."""
    node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()

    for peer in range(node_comm.Get_size()):
        if peer == local_rank:
            continue
        status, supported = runtime.cudaDeviceGetP2PAttribute(
            runtime.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported,
            local_rank,
            peer,
        )
        if status != runtime.cudaError_t.cudaSuccess or not supported:
            return False
    return True


def _skip_if_unsupported_amo(dtype, native_ib_supports_8byte):
    if not _is_cross_node_job():
        if nvshmem.core.n_pes() > 1 and not _has_native_peer_atomics():
            pytest.skip("Same-node GPUs do not support native peer atomics")
        return
    if native_ib_supports_8byte and np.dtype(dtype).itemsize == 8:
        return
    pytest.skip("Cross-node job without NVLink/P2P atomics; remote transport does not "
                f"support this AMO for dtype={np.dtype(dtype).name}")


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_atomic_add_on_array(nvshmem_init_fini, dtype):
    _skip_if_unsupported_amo(dtype, native_ib_supports_8byte=True)

    buf = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    @cuda.jit(lto=True)
    def kernel_atomic_add(arr, val, pe):
        nvshmem.core.device.numba_cuda_mlir.atomic_add(arr, val, pe)

    dev = Device()
    stream = dev.create_stream()

    # Launch kernel to add 5 atomically
    kernel_atomic_add[1, 1, stream](buf, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_add buf={buf}")
    assert buf[0] == dtype(5), f"Expected buf[0] == 5, got {buf[0]}"
    nvshmem.core.free_array(buf)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_atomic_fetch_add_on_array(nvshmem_init_fini, dtype):
    _skip_if_unsupported_amo(dtype, native_ib_supports_8byte=False)

    buf = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    out = nvshmem.core.array((1, ), dtype=dtype)
    out[:] = 0

    @cuda.jit(lto=True)
    def kernel_atomic_fetch_add(arr, out, val, pe):
        out[0] = nvshmem.core.device.numba_cuda_mlir.atomic_fetch_add(arr, val, pe)

    dev = Device()
    stream = dev.create_stream()

    # Launch kernel to add 5 atomically
    kernel_atomic_fetch_add[1, 1, stream](buf, out, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_fetch_add buf={buf}, out={out}")
    assert buf[0] == dtype(5), f"Expected buf[0] == 5, got {buf[0]}"
    assert out[0] == dtype(0), f"Expected out[0] == 0, got {out[0]}"
    nvshmem.core.free_array(buf)
    nvshmem.core.free_array(out)
