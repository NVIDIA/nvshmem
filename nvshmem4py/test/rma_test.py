"""
Unit tests for memory management functionality in nvshmem.core
"""
try:
    import torch
    from torch import float32
    _torch_enabled = True
except:
    float32 = None
    _torch_enabled = False

try:
    import cupy
    _cupy_enabled = True
except:
    _cupy_enabled = False

from utils import uid_init, mpi_init
import argparse
import os

import nvshmem.core

from cuda.core import Device, system

from mpi4py import MPI


def test_rma_on_buffer():
    print("Testing RMA on buffer")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf_src = nvshmem.core.buffer(1024)
    buf_dst = nvshmem.core.buffer(1024)
    stream = dev.create_stream()

    nvshmem.core.put(buf_dst, buf_src, remote_pe=((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()), stream=stream)

    nvshmem.core.free(buf_src)
    nvshmem.core.free(buf_dst)
    print("Done testing RMA on buffer")


def test_rma_on_array():
    print("Testing RMA on Array")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf_src = nvshmem.core.array((4, 4), dtype="float32")
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype="float32")
    buf_dst[:] = 0
    stream = dev.create_stream()

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst 1={buf_dst}, src={buf_src}")

    nvshmem.core.put(buf_dst, buf_src, remote_pe=((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()), stream=stream)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    # After this, buf_dst on PE0 should show 1s and on PE1 should show 0s

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(buf_src)
    print("Done testing RMA on Array")


def test_rma_on_tensor():
    print("Testing RMA on tensor")
    if not _torch_enabled:
        print("Skipping test_rma_on_tensor because torch is not enabled")
        return
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf_src = nvshmem.core.tensor((4, 4), dtype=torch.float32)
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.tensor((4, 4), dtype=torch.float32)
    buf_dst[:] = 0
    stream = dev.create_stream()

    print(f"From PE {nvshmem.core.my_pe()} BEFORE dst 1={buf_dst}, src={buf_src}")

    nvshmem.core.put(buf_dst, buf_src, remote_pe=((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()), stream=stream)

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    # After this, buf_dst on PE0 should show 1s and on PE1 should show 0s

    print(f"From PE {nvshmem.core.my_pe()} AFTER dst={buf_dst}, src={buf_src}")

    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_tensor(buf_src)
    print("Done testing RMA on tensor")


def test_quiet():
    print("Testing quiet")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    stream = dev.create_stream()
    nvshmem.core.quiet(stream=stream)
    print("Done testing quiet quiet")


def test_signal_wait_array():
    print("Testing put/signal on Array")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    stream = dev.create_stream()

    buf_src = nvshmem.core.array((4, 4), dtype="float32")
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype="float32")
    buf_dst[:] = 0

    signal = nvshmem.core.array((1, ), dtype="uint64")
    signal[:] = 0
    buf_sig, sz, type = nvshmem.core.array_get_buffer(signal)

    if nvshmem.core.my_pe() == 0:
        # TODO: Expose signal ops as an enum
        nvshmem.core.put_signal(buf_dst,
                                buf_src,
                                buf_sig,
                                1,
                                nvshmem.core.SignalOp.SIGNAL_SET,
                                remote_pe=1,
                                stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} sent buf to remote PE 1 and set signal")

    if nvshmem.core.my_pe() == 1:
        nvshmem.core.signal_wait(buf_sig, 1, nvshmem.core.ComparisonType.CMP_EQ, stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} waited for signal")

    stream.sync()
    print(f"From PE {nvshmem.core.my_pe()} src={buf_src}, dst={buf_dst} signal={signal}")

    nvshmem.core.free_array(buf_src)
    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(signal)
    print("Done testing put/signal on Array")


def test_signal_wait_array_non_one():
    print("Testing put/signal on Array (with non-default signal_op)")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    stream = dev.create_stream()

    buf_src = nvshmem.core.array((4, 4), dtype="float32")
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.array((4, 4), dtype="float32")
    buf_dst[:] = 0

    signal = nvshmem.core.array((1, ), dtype="uint64")
    signal[:] = 0  # Start below threshold
    buf_sig, sz, type = nvshmem.core.array_get_buffer(signal)

    if nvshmem.core.my_pe() == 0:
        stream.sync()
        nvshmem.core.put_signal(
            buf_dst,
            buf_src,
            buf_sig,
            5,  # <-- Set signal to 5
            nvshmem.core.SignalOp.SIGNAL_SET,
            remote_pe=1,
            stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} sent buf and set signal to 5")

    if nvshmem.core.my_pe() == 1:
        # Use non-default comparison op and a value other than 1
        nvshmem.core.signal_wait(
            buf_sig,
            4,  # <-- Wait for signal to be >= 4
            nvshmem.core.ComparisonType.CMP_GE,
            stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} passed wait for signal >= 4")

    stream.sync()
    print(f"From PE {nvshmem.core.my_pe()} src={buf_src}, dst={buf_dst}, signal={signal}")

    nvshmem.core.free_array(buf_src)
    nvshmem.core.free_array(buf_dst)
    nvshmem.core.free_array(signal)

    print("Done testing put/signal on Array (non-default signal_op)")


def test_signal_wait_tensor():
    print("Testing put/signal on tensor")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    stream = dev.create_stream()

    buf_src = nvshmem.core.tensor((4, 4), dtype=torch.float32)
    buf_src[:] = nvshmem.core.my_pe() + 1
    buf_dst = nvshmem.core.tensor((4, 4), dtype=torch.float32)
    buf_dst[:] = 0

    # Torch doesn't have uint64_t so we need to use CuPy
    signal = nvshmem.core.array((1, ), dtype="uint64")
    signal[:] = 0
    buf_sig, sz, type = nvshmem.core.array_get_buffer(signal)

    if nvshmem.core.my_pe() == 0:
        # TODO: Expose signal ops as an enum
        nvshmem.core.put_signal(buf_dst,
                                buf_src,
                                buf_sig,
                                1,
                                nvshmem.core.SignalOp.SIGNAL_SET,
                                remote_pe=1,
                                stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} sent buf to remote PE 1 and set signal")

    if nvshmem.core.my_pe() == 1:
        nvshmem.core.signal_wait(buf_sig, 1, nvshmem.core.SignalOp.SIGNAL_SET, stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} waited for signal")

    stream.sync()
    print(f"From PE {nvshmem.core.my_pe()} src={buf_src}, dst={buf_dst} signal={signal}")

    nvshmem.core.free_tensor(buf_src)
    nvshmem.core.free_tensor(buf_dst)
    nvshmem.core.free_array(signal)
    print("Done testing put/signal on tensor")


def test_signalop_wait():
    print("Testing signal_op")
    dev = Device()
    local_rank_per_node = dev.device_id
    stream = dev.create_stream()

    signal = nvshmem.core.array((1, ), dtype="uint64")
    signal[:] = 0
    buf_sig, sz, type = nvshmem.core.array_get_buffer(signal)

    if nvshmem.core.my_pe() == 0:
        nvshmem.core.signal_op(buf_sig, 1, nvshmem.core.SignalOp.SIGNAL_SET, remote_pe=1, stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} sent buf to remote PE 1 and set signal")

    if nvshmem.core.my_pe() == 1:
        nvshmem.core.signal_wait(buf_sig, 1, nvshmem.core.ComparisonType.CMP_EQ, stream=stream)
        print(f"From PE {nvshmem.core.my_pe()} waited for signal")

    stream.sync()
    print(f"From PE {nvshmem.core.my_pe()} signal={signal}")

    nvshmem.core.free_array(signal)
    print("Done testing signal_op")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_rma_on_buffer()
    test_rma_on_array()
    test_rma_on_tensor()

    test_quiet()
    test_signal_wait_array()
    test_signal_wait_tensor()
    test_signal_wait_array_non_one()
    test_signalop_wait()

    nvshmem.core.finalize()
