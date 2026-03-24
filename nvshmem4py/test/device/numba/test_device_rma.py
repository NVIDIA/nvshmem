from cuda.core import Device
import numba.cuda as cuda
import nvshmem.core
import nvshmem.core.device.numba

import pytest

rma_dtypes = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_on_array(nvshmem_init_fini, dtype):
    print("Testing RMA on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf_src = nvshmem.core.array((4, 4), dtype=dtype)
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype=dtype)
    buf_dst[:] = 0

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst={buf_dst}, src={buf_src}")

    @cuda.jit
    def test_put(dst, src, pe):
        nvshmem.core.device.numba.put(dst, src, pe)

    dev = Device()
    stream = dev.create_stream()

    test_put[1, 1, stream](buf_dst, buf_src, (nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    assert (buf_dst == ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()) + 1).all()
    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(buf_src)
    print("Done testing put on Array")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_get_on_array(nvshmem_init_fini, dtype):
    print("Testing RMA on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf_src = nvshmem.core.array((4, 4), dtype=dtype)
    buf_src[:] = 0
    buf_dst = nvshmem.core.array((4, 4), dtype=dtype)
    buf_dst[:] = nvshmem.core.my_pe() + 1

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst={buf_dst}, src={buf_src}")

    @cuda.jit
    def test_get(dst, src, pe):
        nvshmem.core.device.numba.get(dst, src, pe)

    dev = Device()
    stream = dev.create_stream()

    test_get[1, 1, stream](buf_src, buf_dst, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    assert (buf_dst == nvshmem.core.my_pe() + 1).all()
    del nb_stream

    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(buf_src)
    print("Done testing get on Array")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_on_array(nvshmem_init_fini, dtype):
    print("Testing RMA on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf_src = nvshmem.core.array((4, 4), dtype=dtype)
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype=dtype)
    buf_dst[:] = 0
    signal_var = nvshmem.core.array((1, ), dtype="uint64")
    signal_var[:] = 0
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst={buf_dst}, src={buf_src}")

    @cuda.jit
    def test_put_signal(dst, src, signal_var, signal_val, signal_op, pe):
        nvshmem.core.device.numba.put_signal(dst, src, signal_var, signal_val, signal_op, pe)

    dev = Device()
    stream = dev.create_stream()

    test_put_signal[1, 1, stream](buf_dst, buf_src, signal_var, signal_val, signal_op, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    assert (buf_dst == nvshmem.core.my_pe() + 1).all()

    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(buf_src)
    del nb_stream
    print("Done testing put signal on Array")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_put_signal_with_wait_on_array(nvshmem_init_fini, dtype):
    print("Testing RMA on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf_src = nvshmem.core.array((4, 4), dtype=dtype)
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype=dtype)
    buf_dst[:] = 0
    signal_var = nvshmem.core.array((1, ), dtype="uint64")
    signal_var[:] = 0
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst={buf_dst}, src={buf_src}")

    @cuda.jit
    def test_put_signal_with_wait(dst, src, signal_var, signal_val, signal_op, pe):
        nvshmem.core.device.numba.put_signal(dst, src, signal_var, 1, signal_op, pe)
        nvshmem.core.device.numba.signal_wait(signal_var, nvshmem.core.ComparisonType.CMP_GE, signal_val)

    dev = Device()
    stream = dev.create_stream()

    test_put_signal_with_wait[1, 1, stream](buf_dst, buf_src, signal_var, signal_val, signal_op, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    if nvshmem.core.my_pe() == 1:
        assert (buf_dst == nvshmem.core.my_pe() + 1).all()
    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(buf_src)
    print("Done testing put signal with wait on Array")


@pytest.mark.mpi
def test_signal_op_signal_wait():
    print("Testing Signal Op and Signal Wait on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    signal_var = nvshmem.core.array((1, ), dtype="uint64")
    signal_var[:] = 0
    signal_val = 1
    signal_op = nvshmem.core.SignalOp.SIGNAL_SET

    @cuda.jit
    def test_signal_op_signal_wait(signal_var, signal_val, signal_op, pe):
        nvshmem.core.device.numba.signal_op(signal_var, 1, signal_op, pe)
        nvshmem.core.device.numba.signal_wait(signal_var, nvshmem.core.ComparisonType.CMP_GE, signal_val)

    dev = Device()
    stream = dev.create_stream()

    test_signal_op_signal_wait[1, 1, stream](signal_var, signal_val, signal_op, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    print("Done testing Signal Op and Signal Wait on Array")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_p(dtype):
    print("Testing shmem_p")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    var = nvshmem.core.array((1, ), dtype=dtype)
    var[:] = 0
    val = 1

    @cuda.jit
    def test_p(var, val, pe):
        nvshmem.core.device.numba.p(var, val, pe)

    dev = Device()
    stream = dev.create_stream()

    test_p[1, 1, stream](var, val, nvshmem.core.my_pe())

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    print(f"From PE {nvshmem.core.my_pe()} AFTER var={var}")
    assert (var == 1).all()

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    print("Done testing shmem_p")


@pytest.mark.mpi
@pytest.mark.parametrize("dtype", rma_dtypes)
def test_g(dtype):
    print("Testing shmem_g")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    var = nvshmem.core.array((1, ), dtype=dtype)
    var[:] = 1
    dest = nvshmem.core.array((1, ), dtype=dtype)
    dest[:] = 0

    @cuda.jit
    def test_g(dest, var, pe):
        local = nvshmem.core.device.numba.g(var, pe)
        dest[:] = local

    dev = Device()
    stream = dev.create_stream()

    test_g[1, 1, stream](dest, var, nvshmem.core.my_pe())
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    print(f"From PE {nvshmem.core.my_pe()} AFTER var={var}, dest={dest}")
    assert (dest == 1).all()

    print("Done testing shmem_g")
