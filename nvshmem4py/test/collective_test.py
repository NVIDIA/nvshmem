"""
Unit tests for host collective on stream functionality in nvshmem.core
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

all_types_cupy = ["float16", "float32", "float64", "uint8", "int8", "int16", "int32", "int64", "bool"]
all_types_torch = [
    torch.float16, torch.bfloat16, torch.float32, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool
]
all_types_nvshmem = ["half", "bfloat16", "uint8", "int8", "int16", "int32", "int64", "double", "float"]
all_ops = ["sum", "min", "max"]

###
# Collectives with no data
###


def test_barrier():
    print("Testing barrier")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    dev.sync()
    # Print dst, src before
    print(f"Before barrier from {nvshmem.core.my_pe()}")
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream=stream)

    # Print dst, src after
    print(f"After barrier from PE {nvshmem.core.my_pe()}")
    print("Done testing barrier")


def test_team_sync():
    print("Testing team sync")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    dev.sync()
    # Print dst, src before
    print(f"Before team sync from {nvshmem.core.my_pe()}")
    nvshmem.core.sync(nvshmem.core.Teams.TEAM_NODE, stream=stream)

    # Print dst, src after
    print(f"After team sync from PE {nvshmem.core.my_pe()}")
    print("Done testing team sync")


def test_barrier_all():
    print("Testing barrier all")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    dev.sync()
    # Print dst, src before
    print(f"Before barrier from {nvshmem.core.my_pe()}")
    nvshmem.core.barrier_all(stream=stream)

    # Print dst, src after
    print(f"After barrier from PE {nvshmem.core.my_pe()}")
    print("Done testing barrier")


def test_all_sync():
    print("Testing all sync")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    dev.sync()
    # Print dst, src before
    print(f"Before all sync from {nvshmem.core.my_pe()}")
    nvshmem.core.sync_all(stream=stream)

    # Print dst, src after
    print(f"After all sync from PE {nvshmem.core.my_pe()}")
    print("Done testing all sync")


###
# CuPy Collectives
###


def test_alltoall_cupy():
    print("Testing CuPy array Alltoall")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_cupy:
        # For alltoall, each PE contributes nelems elements to each other PE
        # Source size = n_pes * nelems, Destination size = n_pes * nelems
        n_pes = nvshmem.core.n_pes()
        my_pe = nvshmem.core.my_pe()

        # Create source array with proper alltoall pattern
        # Each PE has data for all PEs: [PE0_data_for_PE0, PE0_data_for_PE1, PE0_data_for_PE2, ...]
        arr_src = nvshmem.core.array((n_pes, n_pes), dtype=dtype)
        arr_dst = nvshmem.core.array((n_pes, n_pes), dtype=dtype)

        arr_src[:] = my_pe + 1
        arr_dst[:] = 0

        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Print dst, src before
        print(f"PE {my_pe}: Dest before collective:", arr_dst)
        print(f"PE {my_pe}: Src before collective:", arr_src)

        nvshmem.core.alltoall(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, stream=stream)
        # Explicit sync because we didn't set a CuPy stream
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        if dtype == "bool":
            nvshmem.core.free_array(arr_src)
            nvshmem.core.free_array(arr_dst)
            print("Skipping bool correctness check for alltoall test")
            continue
        # Print dst, src after
        print(f"PE {my_pe}: Dest after collective:", arr_dst)
        print(f"PE {my_pe}: Src after collective:", arr_src)

        expected = cupy.zeros((n_pes, n_pes), dtype=dtype)
        for pe in range(n_pes):
            expected[pe, :] = pe + 1

        if not cupy.allclose(arr_dst, expected):
            print(f"ERROR: PE {my_pe}: Alltoall failed for dtype {dtype}")
            print(f"Expected:\n{expected}")
            print(f"Got:\n{arr_dst}")
            raise AssertionError(f"Alltoall data mismatch on PE {my_pe} for dtype {dtype}")
        nvshmem.core.free_array(arr_src)
        nvshmem.core.free_array(arr_dst)

    print("Done testing CuPy array Alltoall")


def test_fcollect_cupy():
    print("Testing CuPy array Fcollect")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()
    dev.sync()

    for dtype in all_types_cupy:
        arr_src = nvshmem.core.array((2, 2), dtype=dtype)
        # We're going to Fcollect from src to dst, so it needs to be N times as big
        arr_dst = nvshmem.core.array((nvshmem.core.n_pes(), 2, 2), dtype=dtype)
        # Zero out the dest and set src to my_pe
        arr_dst[:] = 0
        # as usual, local_rank_per_node + 1 so there's no zeroes
        arr_src[:] = nvshmem.core.my_pe() + 1

        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Print dst, src before
        print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)
        dev.sync()
        nvshmem.core.fcollect(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, stream=stream)
        # Explicit sync because we didn't set a CuPy stream
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()

        # Print dst, src after
        print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

        # Data validation - check that each PE's contribution is in the correct position
        validation_passed = True
        n_pes = nvshmem.core.n_pes()
        my_pe = nvshmem.core.my_pe()

        # The output should be a concatenation of all PEs' input arrays, in PE order.
        # Each PE's input is (2,2) filled with pe+1, so the expected output is
        # [ [1s], [2s], ..., [n_pes]s ] stacked along axis 0

        # Build the expected output array
        expected = cupy.empty_like(arr_dst)
        for pe in range(n_pes):
            expected[pe, :, :] = pe + 1

        if not cupy.all(arr_dst == expected):
            print(f"ERROR: Fcollect output is incorrect on PE {my_pe}")
            print(f"Expected:\n{expected}")
            print(f"Got:\n{arr_dst}")
            validation_passed = False
        else:
            print(f"Fcollect output is correct on PE {my_pe}")

        if not validation_passed:
            nvshmem.core.free_array(arr_src)
            nvshmem.core.free_array(arr_dst)
            raise Exception(f"Fcollect validation failed for dtype {dtype} on PE {my_pe}")

        # Free Buffers
        nvshmem.core.free_array(arr_src)
        nvshmem.core.free_array(arr_dst)
        print(f"Done testing CuPy array Fcollect with type={dtype}")

    print("Done testing CuPy array Fcollect")


def test_reduce_cupy():
    print("Testing CuPy array Reduce")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = nvshmem.core.NvshmemStream(torch.cuda.current_stream())

    for dtype in all_types_cupy:
        for op in all_ops:
            arr_src = nvshmem.core.array((2, 2), dtype=dtype)
            arr_dst = nvshmem.core.array((2, 2), dtype=dtype)
            # Zero out the dest and set src to my_pe
            arr_dst[:] = 0
            # as usual, local_rank_per_node + 1 so there's no zeroes
            arr_src[:] = nvshmem.core.my_pe() + 1

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

            nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, op, stream=stream)
            # Explicit sync because we didn't set a CuPy stream
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            if dtype == "bool":
                nvshmem.core.free_array(arr_src)
                nvshmem.core.free_array(arr_dst)
                print("Skipping bool correctness check for reduce test")
                continue
            # Data correctness checking for reduce
            # After the reduce, arr_dst should contain the reduction result across all PEs
            # For each element, the expected value is the reduction of (pe+1) over all PEs
            n_pes = nvshmem.core.n_pes()
            my_pe = nvshmem.core.my_pe()
            # Build the expected input values for all PEs
            input_values = cupy.arange(1, n_pes + 1, dtype=arr_dst.dtype)
            # For each element in arr_dst, the reduction is over all PEs' values
            if op == "sum":
                expected_value = input_values.sum()
            elif op == "max":
                expected_value = input_values.max()
            elif op == "min":
                expected_value = input_values.min()
            else:
                raise ValueError(f"Unknown reduction op: {op}")

            # Check all elements in arr_dst
            if not cupy.all(arr_dst == expected_value):
                print(f"ERROR: Reduce result incorrect on PE {my_pe} for dtype {dtype} op {op}")
                print(f"Expected all elements to be {expected_value}, got: {arr_dst}")
                raise Exception(f"Reduce validation failed for dtype {dtype} op {op} on PE {my_pe}")
            else:
                print(f"Reduce result correct on PE {my_pe} for dtype {dtype} op {op}")

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

            # Free Buffers
            nvshmem.core.free_array(arr_src)
            nvshmem.core.free_array(arr_dst)
            print(f"Done testing CuPy Array Reduce with dtype {dtype} op {op}")
    print("Done testing CuPy array Reduce")


def test_reducescatter_cupy():
    print("Testing CuPy array Reducescatter")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_cupy:
        for op in all_ops:
            arr_src = nvshmem.core.array((2, 2, nvshmem.core.n_pes()), dtype=dtype)
            # We're going to reducescatter from src to dst, so it needs to be N times as big
            arr_dst = nvshmem.core.array((2, 2), dtype=dtype)
            # Zero out the dest and set src to my_pe
            arr_dst[:] = 0
            arr_src[:] = nvshmem.core.my_pe() + 1

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

            nvshmem.core.reducescatter(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, op, stream=stream)
            # Explicit sync because we didn't set a CuPy stream
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Check correctness of reducescatter result
            # Each PE should get a chunk of the reduction result.
            # For reducescatter, the input is split along the last axis (n_pes), and each PE gets its chunk.
            # Since we filled arr_src with (my_pe + 1), and all PEs did the same,
            # the reduction (default op) is applied along the last axis, and each PE gets the result for its chunk.
            # For op, we need to check which operation is being used.

            n_pes = nvshmem.core.n_pes()
            my_pe = nvshmem.core.my_pe()
            # arr_src shape: (2,2,n_pes), arr_dst shape: (2,2)
            # For each element in arr_dst, it should be the reduction of arr_src[:,:,my_pe] across all PEs
            # But in NVSHMEM, reducescatter does reduction along the last axis, and scatters the result to each PE

            # Build the expected value for this PE
            # All arr_src on all PEs are filled with (their local_rank_per_node + 1)
            # So, for each PE, arr_src[:,:,i] is filled with (i+1)
            # The reduction is over axis=-1, and each PE gets the chunk at index my_pe

            # Build a fake "global" arr_src to compute the expected result
            # For each PE, arr_src[:,:,i] = (i+1)
            # So arr_src_global[:,:,i] = (i+1) for i in 0..n_pes-1
            arr_src_global = cupy.empty((2, 2, n_pes), dtype=arr_src.dtype)
            for i in range(n_pes):
                arr_src_global[:, :, i] = i + 1

            # Now, for reducescatter, the reduction is over axis=-1, and each PE gets its chunk
            # So, for op, we need to apply the reduction over axis=-1, then take the chunk for my_pe
            if op == "sum":
                expected_chunk = cupy.sum(arr_src_global, axis=-1)
            elif op == "max":
                expected_chunk = cupy.max(arr_src_global, axis=-1)
            elif op == "min":
                expected_chunk = cupy.min(arr_src_global, axis=-1)
            else:
                raise Exception(f"Unknown op {op} for reducescatter test")

            if dtype == "bool":
                nvshmem.core.free_array(arr_src)
                nvshmem.core.free_array(arr_dst)
                print("Skipping bool correctness check for reduce test")
                continue

            # But reducescatter splits the reduction result into n_pes chunks along a flattened axis
            # Since arr_dst is (2,2), and arr_src is (2,2,n_pes), and n_pes == arr_src.shape[-1]
            # Each PE gets a (2,2) chunk, which is just the reduction result for that chunk

            # So arr_dst should be equal to expected_chunk for all PEs
            if not cupy.all(arr_dst == expected_chunk):
                print(f"ERROR: Reducescatter result incorrect on PE {my_pe} for dtype {dtype} op {op}")
                print(f"Expected arr_dst to be:\n{expected_chunk}\nGot:\n{arr_dst}")
                raise Exception(f"Reducescatter validation failed for dtype {dtype} op {op} on PE {my_pe}")
            else:
                print(f"Reducescatter result correct on PE {my_pe} for dtype {dtype} op {op}")

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

            # Free Buffers
            nvshmem.core.free_array(arr_src)
            nvshmem.core.free_array(arr_dst)
            print(f"Done testing CuPy Array Reducescatter with dtype {dtype} op {op}")
    print("Done testing CuPy array Reducescatter")


def test_broadcast_cupy():
    print("Testing CuPy array broadcast")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_cupy:
        arr_src = nvshmem.core.array((2, 2), dtype=dtype)
        arr_dst = nvshmem.core.array((2, 2), dtype=dtype)
        # Zero out the dest and set src to my_pe
        arr_dst[:] = 0
        arr_src[:] = nvshmem.core.my_pe() + 1

        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Print dst, src before
        print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

        nvshmem.core.broadcast(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, root=0, stream=stream)
        # Explicit sync because we didn't set a CuPy stream
        # Important! Make sure you sync the stream that you launched the nvshmem collective on
        # (or the whole device like we do in this example)
        # before you use the results of the collective
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Check that after broadcast, all arr_dst values match the root's arr_src
        # Only root PE's arr_src is used for broadcast, which is root=0
        root = 0
        # Gather what the root's arr_src should be
        expected = cupy.full((2, 2), root + 1, dtype=arr_dst.dtype)
        # arr_dst should match expected on all PEs
        if not cupy.all(arr_dst == expected):
            print(f"ERROR: Broadcast result incorrect on PE {nvshmem.core.my_pe()} for dtype {dtype}")
            print(f"Expected arr_dst to be:\n{expected}\nGot:\n{arr_dst}")
            raise Exception(f"Broadcast validation failed for dtype {dtype} on PE {nvshmem.core.my_pe()}")
        else:
            print(f"Broadcast result correct on PE {nvshmem.core.my_pe()} for dtype {dtype}")

        # Print dst, src after
        print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

        # Free Buffers
        nvshmem.core.free_array(arr_src)
        nvshmem.core.free_array(arr_dst)
        print(f"Done testing CuPy Array broadcast with dtype {dtype}")
    print("Done testing CuPy array broadcast")


###
# Torch Collectives
###


def test_alltoall_torch():
    print("Testing Torch array Alltoall")
    n_pes = nvshmem.core.n_pes()
    my_pe = nvshmem.core.my_pe()
    if not _torch_enabled:
        print("WARNING: Torch not found. Not running Torch collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_torch:
        arr_src = nvshmem.core.tensor((n_pes, n_pes), dtype=dtype)
        # We're going to alltoall from src to dst, so it needs to be N times as big
        arr_dst = nvshmem.core.tensor((n_pes, n_pes), dtype=dtype)
        # Zero out the dest and set src to my_pe
        arr_dst[:] = 0
        arr_src[:] = nvshmem.core.my_pe() + 1

        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Print dst, src before
        print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

        nvshmem.core.alltoall(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, stream=stream)
        # Explicit sync because we didn't set a Torch stream
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()

        # Print dst, src after
        print(f"PE {my_pe}: Dest after collective:", arr_dst)
        print(f"PE {my_pe}: Src after collective:", arr_src)

        if dtype == torch.bool or dtype == torch.bfloat16:
            nvshmem.core.free_tensor(arr_src)
            nvshmem.core.free_tensor(arr_dst)
            print("Skipping bool/bf16 correctness check for alltoall test")
            continue

        # Data validation for alltoall
        # Each PE should receive data from all PEs in the destination
        validation_passed = True

        expected = torch.zeros((n_pes, n_pes), dtype=dtype, device=arr_dst.device)
        for pe in range(n_pes):
            expected[pe, :] = pe + 1

        if not torch.allclose(arr_dst, expected):
            validation_passed = False
            print(f"ERROR: Alltoall result incorrect on PE {my_pe} for dtype {dtype}")
            print(f"Expected:\n{expected}\nGot:\n{arr_dst}")

        if not validation_passed:
            raise Exception(f"Alltoall validation failed for dtype {dtype} on PE {nvshmem.core.my_pe()}")

        # Free Buffers
        nvshmem.core.free_tensor(arr_src)
        nvshmem.core.free_tensor(arr_dst)
        print(f"Done testing Torch array Alltoall with type={dtype}")


def test_fcollect_torch():
    print("Testing Torch array Fcollect")
    if not _torch_enabled:
        print("WARNING: Torch not found. Not running Torch collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_torch:
        arr_src = nvshmem.core.tensor((2, 2), dtype=dtype)
        # We're going to Fcollect from src to dst, so it needs to be N times as big
        arr_dst = nvshmem.core.tensor((nvshmem.core.n_pes(), 2, 2), dtype=dtype)
        # Zero out the dest and set src to my_pe
        arr_dst[:] = 0
        arr_src[:] = nvshmem.core.my_pe() + 1

        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()
        # Print dst, src before
        print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

        nvshmem.core.fcollect(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, stream=stream)
        # Explicit sync because we didn't set a Torch stream
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
        dev.sync()

        # Print dst, src after
        print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
        print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

        if dtype == torch.bool:
            nvshmem.core.free_tensor(arr_src)
            nvshmem.core.free_tensor(arr_dst)
            print("Skipping bool correctness check for fcollect test")
            continue

        # Data validation - check that each PE's contribution is in the correct position
        validation_passed = True
        n_pes = nvshmem.core.n_pes()

        for pe in range(n_pes):
            expected_value = pe + 1
            actual_data = arr_dst[pe, :, :]

            # Check if all elements in this PE's contribution are correct
            if not torch.all(actual_data == expected_value):
                print(f"ERROR: Data from PE {pe} is incorrect on PE {nvshmem.core.my_pe()}")
                print(f"Expected all elements to be {expected_value}, got: {actual_data}")
                validation_passed = False
            else:
                print(f"✓ Data from PE {pe} is correct on PE {nvshmem.core.my_pe()}")

        if not validation_passed:
            raise Exception(f"Fcollect validation failed for dtype {dtype} on PE {nvshmem.core.my_pe()}")

        # Free Buffers
        nvshmem.core.free_tensor(arr_src)
        nvshmem.core.free_tensor(arr_dst)
        print(f"Done testing Torch array Fcollect with type={dtype}")

    print("Done testing Torch array Fcollect")


def test_reduce_torch():
    print("Testing torch tensor Reduce")
    if not _torch_enabled:
        print("WARNING: torch not found. Not running torch collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_torch:
        for op in all_ops:
            arr_src = nvshmem.core.tensor((2, 2), dtype=dtype)
            arr_dst = nvshmem.core.tensor((2, 2), dtype=dtype)
            # Zero out the dest and set src to my_pe
            arr_dst[:] = 0
            # as usual, local_rank_per_node + 1 so there's no zeroes
            arr_src[:] = nvshmem.core.my_pe() + 1

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

            nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, op, stream=stream)
            # Explicit sync because we didn't set a torch stream
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Check data correctness for reduce
            # For reduce, arr_dst should contain the reduction (op) of all arr_src across all PEs
            # Since arr_src[:] = local_rank_per_node + 1 on each PE, and arr_dst is (2,2)
            # The expected value is reduce_op applied to [pe+1 for pe in range(n_pes())]
            import operator

            n_pes = nvshmem.core.n_pes()
            my_pe = nvshmem.core.my_pe()
            # Build the list of values contributed by each PE
            values = [pe + 1 for pe in range(n_pes)]

            # Map op string to python operator
            op_map = {
                "sum": sum,
                "min": min,
                "max": max,
            }
            # For prod, use functools.reduce for better compatibility
            import functools
            if op == "prod":
                expected_value = functools.reduce(operator.mul, values, 1)
            elif op in op_map:
                expected_value = op_map[op](values)
            else:
                raise Exception(f"Unknown reduction op: {op}")

            if dtype == torch.bool:
                nvshmem.core.free_tensor(arr_src)
                nvshmem.core.free_tensor(arr_dst)
                print("Skipping bool correctness check for reduce test")
                continue

            # Now check that all elements in arr_dst are equal to expected_value
            if not torch.all(arr_dst == expected_value):
                print(f"ERROR: Reduce result incorrect on PE {my_pe} for dtype {dtype} op {op}")
                print(f"Expected all elements to be {expected_value}, got: {arr_dst}")
                raise Exception(f"Reduce validation failed for dtype {dtype} op {op} on PE {my_pe}")
            else:
                print(f"✓ Reduce result correct on PE {my_pe} for dtype {dtype} op {op}")

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

            # Free Buffers
            nvshmem.core.free_tensor(arr_src)
            nvshmem.core.free_tensor(arr_dst)
            print(f"Done testing torch tensor Reduce with dtype {dtype} op {op}")
    print("Done testing torch tensor Reduce")


def test_reducescatter_torch():
    print("Testing torch tensor reducescatter")
    if not _torch_enabled:
        print("WARNING: torch not found. Not running torch collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()
    n_pes = nvshmem.core.n_pes()

    for dtype in all_types_torch:
        for op in all_ops:
            arr_src = nvshmem.core.tensor((2, 2, nvshmem.core.n_pes()), dtype=dtype)
            arr_dst = nvshmem.core.tensor((2, 2), dtype=dtype)
            # Zero out the dest and set src to my_pe
            arr_dst[:] = 0
            # as usual, local_rank_per_node + 1 so there's no zeroes
            arr_src[:] = nvshmem.core.my_pe() + 1

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

            nvshmem.core.reducescatter(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, op, stream=stream)
            # Explicit sync because we didn't set a torch stream
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Check data correctness for reducescatter
            # For reducescatter, arr_dst should contain the reduction (op) of all arr_src across all PEs
            # Since arr_src[:] = local_rank_per_node + 1 on each PE, and arr_dst is (2,2)
            # The expected value is reducescatter_op applied to [pe+1 for pe in range(n_pes())]
            import operator

            n_pes = nvshmem.core.n_pes()
            my_pe = nvshmem.core.my_pe()
            # Build the list of values contributed by each PE
            values = [pe + 1 for pe in range(n_pes)]

            # Map op string to python operator
            op_map = {
                "sum": sum,
                "min": min,
                "max": max,
            }
            if op in op_map:
                expected_value = op_map[op](values)
            else:
                raise Exception(f"Unknown reduction op: {op}")

            if dtype == torch.bool:
                nvshmem.core.free_tensor(arr_src)
                nvshmem.core.free_tensor(arr_dst)
                print("Skipping bool correctness check for reducescatter test")
                continue

            # Now check that all elements in arr_dst are equal to expected_value
            if not torch.all(arr_dst == expected_value):
                print(f"ERROR: reducescatter result incorrect on PE {my_pe} for dtype {dtype} op {op}")
                print(f"Expected all elements to be {expected_value}, got: {arr_dst}")
                raise Exception(f"reducescatter validation failed for dtype {dtype} op {op} on PE {my_pe}")
            else:
                print(f"✓ reducescatter result correct on PE {my_pe} for dtype {dtype} op {op}")

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

            # Free Buffers
            nvshmem.core.free_tensor(arr_src)
            nvshmem.core.free_tensor(arr_dst)
            print(f"Done testing torch tensor reducescatter with dtype {dtype} op {op}")
    print("Done testing torch tensor reducescatter")


def test_broadcast_torch():
    print("Testing torch tensor broadcast")
    if not _torch_enabled:
        print("WARNING: torch not found. Not running torch collective test")
        return
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    dev = Device(local_rank_per_node)
    stream = dev.create_stream()

    for dtype in all_types_torch:
        for op in all_ops:
            arr_src = nvshmem.core.tensor((2, 2), dtype=dtype)
            arr_dst = nvshmem.core.tensor((2, 2), dtype=dtype)
            # Zero out the dest and set src to my_pe
            arr_dst[:] = 0
            # as usual, local_rank_per_node + 1 so there's no zeroes
            arr_src[:] = local_rank_per_node + 1

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", arr_src)

            nvshmem.core.broadcast(nvshmem.core.Teams.TEAM_WORLD, arr_dst, arr_src, root=0, stream=stream)
            # Explicit sync because we didn't set a torch stream
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()

            if dtype == torch.bool or dtype == torch.bfloat16:
                nvshmem.core.free_tensor(arr_src)
                nvshmem.core.free_tensor(arr_dst)
                print("Skipping bool/bf16 correctness check for broadcast test")
                continue

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", arr_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", arr_src)

            # On root, arr_src is filled with local_rank_per_node + 1, which is 1
            expected = torch.full_like(arr_dst, 1)
            if not torch.allclose(arr_dst, expected):
                raise Exception(f"Broadcast validation failed for dtype {dtype} on PE {local_rank_per_node}. "
                                f"Expected {expected}, got {arr_dst}")
            else:
                print(f"Broadcast result correct on PE {local_rank_per_node} for dtype {dtype}")
            # Free Buffers
            nvshmem.core.free_tensor(arr_src)
            nvshmem.core.free_tensor(arr_dst)
            print(f"Done testing torch tensor broadcast with dtype {dtype} op {op}")
    print("Done testing torch tensor broadcast")


###
# Collectives on raw buffers
###


def test_collectives_on_raw_buffers():
    print("Testing collectives on raw buffers")
    # Get a buffer
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)
    stream = dev.create_stream()
    for coll in ["reduce", "reducescatter", "fcollect", "alltoall", "broadcast"]:
        for dtype in all_types_nvshmem:
            buf_src = nvshmem.core.buffer(nvshmem.core.utils.get_size((2, 2, nvshmem.core.n_pes()), dtype))
            buf_dst = nvshmem.core.buffer(nvshmem.core.n_pes() * nvshmem.core.utils.get_size(
                (2, 2, nvshmem.core.n_pes()), dtype))

            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()
            # Print dst, src before
            print(f"Dest before collective from PE {nvshmem.core.my_pe()}:", buf_dst)
            print(f"Src before collective from PE {nvshmem.core.my_pe()}:", buf_src)
            if coll in ["reduce", "reducescatter"]:
                for op in all_ops:
                    nvshmem.core.collective_on_buffer(coll,
                                                      nvshmem.core.Teams.TEAM_WORLD,
                                                      buf_dst,
                                                      buf_src,
                                                      dtype=dtype,
                                                      op=op,
                                                      stream=stream)
            else:
                nvshmem.core.collective_on_buffer(coll,
                                                  nvshmem.core.Teams.TEAM_WORLD,
                                                  buf_dst,
                                                  buf_src,
                                                  dtype=dtype,
                                                  op=None,
                                                  stream=stream)
            # Important! Make sure you sync the stream that you launched the nvshmem collective on
            # (or the whole device like we do in this example)
            # before you use the results of the collective
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            dev.sync()

            # Print dst, src after
            print(f"Dest after collective from PE {nvshmem.core.my_pe()}:", buf_dst)
            print(f"Src after collective from PE {nvshmem.core.my_pe()}:", buf_src)

            # Free Buffers
            nvshmem.core.free(buf_src)
            nvshmem.core.free(buf_dst)
            print(f"Done testing {coll} with dtype {dtype} on raw buffer")
    print("Done testing on raw buffers")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    dev = Device()
    local_rank_per_node = dev.device_id

    # Cupy Interop
    test_fcollect_cupy()
    dev.sync()
    test_reduce_cupy()
    dev.sync()
    test_alltoall_cupy()
    dev.sync()
    test_reducescatter_cupy()
    dev.sync()
    test_broadcast_cupy()

    # Torch interop
    test_alltoall_torch()
    dev.sync()
    test_fcollect_torch()
    dev.sync()
    test_reduce_torch()
    dev.sync()
    test_reducescatter_torch()
    dev.sync()
    test_broadcast_torch()
    dev.sync()

    # Raw Buffers
    test_collectives_on_raw_buffers()
    dev.sync()

    # No arrays
    test_barrier()
    dev.sync()
    test_team_sync()
    dev.sync()
    test_barrier_all()
    dev.sync()
    test_all_sync()
    dev.sync()

    nvshmem.core.finalize()
