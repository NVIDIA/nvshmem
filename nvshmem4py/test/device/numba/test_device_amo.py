from cuda.core import Device, Stream
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba

import pytest

amo_std_dtypes = ["int32", "int64", "uint64"]
amo_ext_dtypes = ["float32", "float64"] + amo_std_dtypes
amo_bit_dtypes = amo_std_dtypes + ["uint32"]


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_std_dtypes)
def test_atomic_add_on_array(nvshmem_init_fini, dtype):
    print("Testing atomic_add")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    @cuda.jit
    def kernel_atomic_add(arr, val, pe):
        nvshmem.core.device.numba.atomic_add(arr, val, pe)

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    # Launch kernel to add 5 atomically
    kernel_atomic_add[1, 1, nb_stream](buf, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=cu_stream_ref)
    cu_stream_ref.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_add buf={buf}")

    assert (buf == 5).all()

    nvshmem.core.free_array(buf)
    print("Done testing atomic_add")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", amo_std_dtypes)
def test_atomic_fetch_add_on_array(nvshmem_init_fini, dtype):
    print("Testing atomic_fetch_add")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf = nvshmem.core.array((1, ), dtype=dtype)
    out = nvshmem.core.array((1, ), dtype=dtype)
    buf[:] = 0

    @cuda.jit
    def kernel_atomic_fetch_add(arr, out, val, pe):
        result = nvshmem.core.device.numba.atomic_fetch_add(arr, val, pe)
        out[:] = result

    nb_stream = cuda.stream()
    cu_stream_ref = Stream.from_handle(int(nb_stream.handle))

    # Launch kernel to add 5 atomically
    kernel_atomic_fetch_add[1, 1, nb_stream](buf, out, 5, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=cu_stream_ref)
    cu_stream_ref.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER atomic_fetch_add buf={buf}, out={out}")

    assert (buf == 5).all()

    nvshmem.core.free_array(buf)
    nvshmem.core.free_array(out)
    print("Done testing atomic_fetch_add")
