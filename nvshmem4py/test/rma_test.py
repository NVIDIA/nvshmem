# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for memory management functionality in nvshmem.core
"""
try:
    import torch
    from torch import float32
    _torch_enabled = True
except:
    torch = None
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


def _requires_multiple_pes(test_name):
    """Return whether a cross-PE RMA test can run in this launch."""
    if nvshmem.core.n_pes() < 2:
        print(f"Skipping {test_name}: requires at least two PEs")
        return False
    return True


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

    dev.sync()

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

    dev.sync()

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


def test_host_atomic_memory_operations():
    """Exercise high-level host AMOs against a local symmetric target.

    A local target keeps this test independent of remote atomic-transport
    support while still validating high-level dtype dispatch and the
    return semantics of every host AMO family.
    """
    if not _cupy_enabled:
        print("Skipping test_host_atomic_memory_operations because CuPy is not enabled")
        return

    print("Testing host atomic memory operations")
    dev = Device()
    my_pe = nvshmem.core.my_pe()
    int_values = None
    uint_values = None
    float_values = None

    try:
        int_values = nvshmem.core.array((1, ), dtype="int32")
        uint_values = nvshmem.core.array((1, ), dtype="uint32")
        float_values = nvshmem.core.array((1, ), dtype="float32")
        int_values[0] = 10
        uint_values[0] = 0b1010
        float_values[0] = 1.25
        dev.sync()

        # Standard integral operations on one scalar target.
        assert nvshmem.core.atomic_fetch(int_values, my_pe) == 10
        nvshmem.core.atomic_set(int_values, 12, my_pe)
        assert nvshmem.core.atomic_fetch_add(int_values, 4, my_pe) == 12
        nvshmem.core.atomic_add(int_values, 3, my_pe)
        assert nvshmem.core.atomic_fetch_inc(int_values, my_pe) == 19
        nvshmem.core.atomic_inc(int_values, my_pe)
        assert nvshmem.core.atomic_swap(int_values, 30, my_pe) == 21
        assert nvshmem.core.atomic_compare_swap(int_values, 30, 25, my_pe) == 30
        assert nvshmem.core.atomic_compare_swap(int_values, 30, 99, my_pe) == 25

        # Bitwise AMOs cover the OpenSHMEM 1.4 family.
        assert nvshmem.core.atomic_fetch_or(uint_values, 0b0101, my_pe) == 0b1010
        nvshmem.core.atomic_and(uint_values, 0b1101, my_pe)
        assert nvshmem.core.atomic_fetch_xor(uint_values, 0b0011, my_pe) == 0b1101
        nvshmem.core.atomic_or(uint_values, 0b0001, my_pe)
        nvshmem.core.atomic_xor(uint_values, 0b0011, my_pe)
        assert nvshmem.core.atomic_fetch_and(uint_values, 0b1111, my_pe) == 0b1100

        # Floating-point add/fetch-add use the NVSHMEM extension bindings.
        assert abs(nvshmem.core.atomic_fetch(float_values, my_pe) - 1.25) < 1e-6
        nvshmem.core.atomic_set(float_values, 1.5, my_pe)
        assert abs(nvshmem.core.atomic_fetch_add(float_values, 0.5, my_pe) - 1.5) < 1e-6
        nvshmem.core.atomic_add(float_values, 0.25, my_pe)
        dev.sync()

        assert int(int_values.get()[0]) == 25
        assert int(uint_values.get()[0]) == 0b1100
        assert abs(float(float_values.get()[0]) - 2.25) < 1e-6
    finally:
        for values in (int_values, uint_values, float_values):
            if values is not None:
                nvshmem.core.free_array(values)

    print("Done testing host atomic memory operations")


def test_rma_nbi_on_array():
    """Exercise the host NBI put/get paths and their completion operations."""
    print("Testing NBI RMA on Array")
    if not _requires_multiple_pes("test_rma_nbi_on_array"):
        return

    dev = Device()
    stream = dev.create_stream()
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    next_pe = (my_pe + 1) % n_pes
    previous_pe = (my_pe - 1 + n_pes) % n_pes

    put_src = put_dst = get_src = get_dst = None
    try:
        put_src = nvshmem.core.array((4, 4), dtype="int32")
        put_dst = nvshmem.core.array((4, 4), dtype="int32")
        get_src = nvshmem.core.array((4, 4), dtype="int32")
        get_dst = nvshmem.core.array((4, 4), dtype="int32")
        put_src[:] = my_pe + 1
        put_dst[:] = 0
        get_src[:] = my_pe + 1
        get_dst[:] = 0

        # Initialize all symmetric buffers before either PE begins an NBI transfer.
        dev.sync()
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        stream.sync()

        # flush() establishes the source-reuse ordering for this put_nbi(). quiet()
        # then establishes the remote-completion guarantee used by the assertion.
        nvshmem.core.put_nbi(put_dst, put_src, remote_pe=next_pe, stream=stream)
        nvshmem.core.flush(stream=stream)
        nvshmem.core.quiet(stream=stream)

        # get_nbi() completes locally at quiet(); it reads the source on next_pe.
        nvshmem.core.get_nbi(get_dst, get_src, remote_pe=next_pe, stream=stream)
        nvshmem.core.quiet(stream=stream)

        # Do not free a source until every PE has completed its inbound get/put.
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        stream.sync()

        assert (put_dst == previous_pe + 1).all(), f"PE {my_pe}: put_nbi result is incorrect"
        assert (get_dst == next_pe + 1).all(), f"PE {my_pe}: get_nbi result is incorrect"
    finally:
        for array in (put_src, put_dst, get_src, get_dst):
            if array is not None:
                nvshmem.core.free_array(array)

    print("Done testing NBI RMA on Array")


def test_put_signal_nbi_and_signal_fetch():
    """Validate host put_signal_nbi, signal_wait, and synchronous signal_fetch."""
    print("Testing NBI put/signal and signal_fetch")
    if not _requires_multiple_pes("test_put_signal_nbi_and_signal_fetch"):
        return

    dev = Device()
    stream = dev.create_stream()
    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    next_pe = (my_pe + 1) % n_pes
    previous_pe = (my_pe - 1 + n_pes) % n_pes

    src = dst = signal = None
    try:
        src = nvshmem.core.array((4, 4), dtype="int32")
        dst = nvshmem.core.array((4, 4), dtype="int32")
        signal = nvshmem.core.array((1, ), dtype="uint64")
        src[:] = my_pe + 1
        dst[:] = 0
        signal[:] = 0
        signal_buf, _, _ = nvshmem.core.array_get_buffer(signal)
        signal_value = my_pe + 1

        dev.sync()
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        stream.sync()

        nvshmem.core.put_signal_nbi(dst,
                                    src,
                                    signal_buf,
                                    signal_value,
                                    nvshmem.core.SignalOp.SIGNAL_SET,
                                    remote_pe=next_pe,
                                    stream=stream)
        # Unlike flush(), quiet() is the completion operation required for the
        # signaling NBI operation before the remote PE consumes the signal/data.
        nvshmem.core.quiet(stream=stream)
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        stream.sync()

        expected_signal = previous_pe + 1
        nvshmem.core.signal_wait(signal_buf, expected_signal, nvshmem.core.ComparisonType.CMP_EQ, stream=stream)
        stream.sync()

        assert (dst == expected_signal).all(), f"PE {my_pe}: put_signal_nbi data is incorrect"
        assert nvshmem.core.signal_fetch(signal_buf) == expected_signal, f"PE {my_pe}: signal_fetch is incorrect"
    finally:
        for array in (src, dst, signal):
            if array is not None:
                nvshmem.core.free_array(array)

    print("Done testing NBI put/signal and signal_fetch")


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

    dev.sync()

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

    dev.sync()

    if nvshmem.core.my_pe() == 0:
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
    if not _torch_enabled:
        print("Skipping test_signal_wait_tensor because torch is not enabled")
        return
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

    dev.sync()

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

    dev.sync()

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
    parser.add_argument("--skip-signalop", action="store_true", help="Skip test_signalop_wait (proxy timeout on PCIe)")
    parser.add_argument("--skip-amo", action="store_true", help="Skip host AMO coverage on unsupported targets")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_rma_on_buffer()
    test_rma_on_array()
    test_rma_on_tensor()

    test_quiet()
    if args.skip_amo:
        print("Skipping test_host_atomic_memory_operations (--skip-amo)")
    else:
        test_host_atomic_memory_operations()
    test_rma_nbi_on_array()
    test_put_signal_nbi_and_signal_fetch()
    test_signal_wait_array()
    test_signal_wait_tensor()
    test_signal_wait_array_non_one()
    if not args.skip_signalop:
        test_signalop_wait()
    else:
        print("Skipping test_signalop_wait (--skip-signalop)")

    # Sync all PEs before finalization to avoid proxy timeout during cleanup
    dev = Device()
    stream = dev.create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()
    dev.sync()
    nvshmem.core.finalize()
