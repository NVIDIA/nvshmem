import pytest
import numpy as np

from numba import cuda
from cuda.core import Device

import nvshmem.core
import nvshmem.core.device.numba


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_atomic_add_on_array(nvshmem_init_fini, dtype):
    buf = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    @cuda.jit
    def kernel_atomic_add(arr, val, pe):
        nvshmem.core.device.numba.atomic_add(arr, val, pe)

    dev = Device()
    stream = dev.create_stream()

    # Launch kernel to add 5 atomically
    kernel_atomic_add[1, 1, stream](buf, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_add buf={buf}")


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_atomic_fetch_add_on_array(nvshmem_init_fini, dtype):
    buf = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    out = nvshmem.core.array((1, ), dtype=dtype)
    out[:] = 0

    @cuda.jit
    def kernel_atomic_fetch_add(arr, out, val, pe):
        out[0] = nvshmem.core.device.numba.atomic_fetch_add(arr, val, pe)

    dev = Device()
    stream = dev.create_stream()

    # Launch kernel to add 5 atomically
    kernel_atomic_fetch_add[1, 1, stream](buf, out, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_fetch_add buf={buf}, out={out}")
