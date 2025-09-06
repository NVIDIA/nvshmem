"""
Utility functions for NVSHMEM Python Perftests
"""

import argparse
import re

import nvshmem.core
from nvshmem.core.utils import dtype_nbytes

from mpi4py import MPI
import numpy as np
from cuda.core.experimental import Device, system

def run_coll_benchmark(args, coll):
    """
    Run the COLL_on_stream performance test using NVSHMEM collectives and CUDA events.
    """
    dev = Device()
    stream = dev.create_stream()

    # Log device info
    print(f"mype: {nvshmem.core.my_pe()} mype_node: {nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)} "
          f"device name: {dev.name} bus id: {dev.pci_bus_id}")

    if coll in ("reduce", "reducescatter") and args.datatype is None or args.reduce_op is None:
        raise ValueError("You must specify both --datatype and --reduce_op for this benchmark.")

    min_size = args.min_size or 4
    max_size = args.max_size or (1 << 22)
    step = args.step or 2
    iters = args.iters or 10
    warmup = args.warmup_iters or 5
    dtype = args.datatype
    op = args.reduce_op

    # Hack because the python function and C function are named `reduce_` but the perftest is named `reduction_`
    print_header(coll if coll != "reduce" else "reduction")

    if coll not in ("reduce", "reducescatter"):
        op = None

    min_size = max(dtype_nbytes(dtype), min_size)

    if min_size < nvshmem.core.n_pes() and coll in ("reducescatter", "alltoall", "fcollect"):
        min_size = nvshmem.core.n_pes() * dtype_nbytes(dtype)

    if coll in ("broadcast", "reduce"):
        min_elems = max(1, min_size // dtype_nbytes(dtype))
        max_elems = max(1, max_size // dtype_nbytes(dtype))
    elif coll in ("reducescatter", "alltoall", "fcollect"):
        min_elems = max(1, min_size // (nvshmem.core.n_pes() * dtype_nbytes(dtype)))
        max_elems = max(1, max_size // (nvshmem.core.n_pes() * dtype_nbytes(dtype)))


    n_elems = min_elems
    while n_elems <= max_elems:

        size_bytes = n_elems * dtype_nbytes(dtype)
        display_size = size_bytes
        display_elems = n_elems
        if coll in ("alltoall", "fcollect", "reducescatter"):
            display_size *= nvshmem.core.n_pes()
            display_elems *= nvshmem.core.n_pes()
        if coll in ("reduce", "broadcast"):
            buf_dst = nvshmem.core.buffer(size_bytes)
            buf_src = nvshmem.core.buffer(size_bytes)
        elif coll in ("alltoall"):
            buf_dst = nvshmem.core.buffer(size_bytes * nvshmem.core.n_pes())
            buf_src = nvshmem.core.buffer(size_bytes * nvshmem.core.n_pes())
        elif coll in ("reducescatter"):
            buf_dst = nvshmem.core.buffer(size_bytes)
            buf_src = nvshmem.core.buffer(size_bytes * nvshmem.core.n_pes())
        elif coll in ("fcollect"):
            buf_src = nvshmem.core.buffer(size_bytes * nvshmem.core.n_pes())
            buf_dst = nvshmem.core.buffer(size_bytes)
             

        # Warmup
        for _ in range(warmup):
            nvshmem.core.collective_on_buffer(
                coll,
                nvshmem.core.Teams.TEAM_WORLD,
                buf_dst, buf_src,
                dtype=dtype,
                op=op,
                stream=stream
            )
        dev.sync()

        # Timed iterations
        latencies = []
        for _ in range(iters):
            # If barrier/sync are removed, sometimes unwanted time spent on prev task shows up on PE0
            nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
            stream.sync()
            

            time_ms = nvshmem.core.collective_on_buffer(
                coll,
                nvshmem.core.Teams.TEAM_WORLD,
                buf_dst, buf_src,
                dtype=dtype,
                op=op,
                stream=stream,
                enable_timing=True
            )
            latencies.append(time_ms)

        latencies_us = [x * 1000 for x in latencies]  # convert ms -> us
        if nvshmem.core.my_pe() == 0:
            print_result(display_size, display_elems, dtype, latencies_us, coll, op=op)

        nvshmem.core.free(buf_src)
        nvshmem.core.free(buf_dst)
        n_elems *= step

def uid_init():
    # This will use mpi4py to perform a UID based init with bcast.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # Broadcast UID to all ranks
    comm.Bcast(uniqueid._data.view(np.int8), root=0)

    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks,
                      mpi_comm=None, initializer_method="uid")

def parse_size(size_str):
    """
    Parse a human-readable size string like '1G', '512K', '2M', '1.5MiB' into bytes.
    Case-insensitive. Supports K, M, G, T (with or without iB or B suffix).
    """
    size_str = size_str.strip().lower()
    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([kmgt]?)(i?b?)$', size_str)

    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number, unit, _ = match.groups()
    number = float(number)

    unit_multipliers = {
        '': 1,
        'k': 1024,
        'm': 1024 ** 2,
        'g': 1024 ** 3,
        't': 1024 ** 4,
    }

    multiplier = unit_multipliers.get(unit, None)
    if multiplier is None:
        raise ValueError(f"Unknown size unit: {unit}")

    return int(number * multiplier)

def mpi_init():
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None,
                      mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

def print_header(coll):
    """
    Print the column headers for the benchmark results table.
    """
    print(f"#{coll}_on_stream.py")
    print(f"{'size(B)':<12}"
          f"{'count':<12}"
          f"{'type':<10}"
          f"{'latency(us)':<18}"
          f"{'min_lat(us)':<18}"
          f"{'max_lat(us)':<18}"
          f"{'algbw(GB/s)':<15}"
          f"{'busbw(GB/s)':<15}")

def adjusted_logical_size(size_bytes, coll):
    """
    Return the logical total communication size (in bytes) for a given collective operation.
    Parameters:
        size_bytes (int): The size of the message buffer per PE, in bytes.
        coll (str): The collective operation type.

    Returns:
        float: The adjusted logical size (communication volume) in bytes.
    """
    n_pes = nvshmem.core.n_pes()

    if coll in ("reduce"):
        factor = 2 * (n_pes - 1) / n_pes
    elif coll in ("broadcast"):
        factor = 1.0
    elif coll in ("alltoall", "fcollect", "reducescatter"):
        factor = (n_pes - 1) / n_pes
    else:
        print(f"Warning: Job Name {coll} bandwidth factor not set. Using factor = 1.")
        factor = 1.0

    return size_bytes * factor

def print_result(size, n_elems, dtype, latencies_us, coll, op=None):
    """
    Print a single row of benchmark results in a tabular format.
    """
    import numpy as np

    avg_us = np.mean(latencies_us)
    min_us = np.min(latencies_us)
    max_us = np.max(latencies_us)

    # Convert to seconds for bandwidth calculation
    avg_s = avg_us / 1e6
    size_gb = size / ( 1e9)

    # This size is already in Bytes
    adj_size = (adjusted_logical_size(size, coll) / ( 1e9))
    bus_bandwidth = adj_size / avg_s
    alg_bandwidth = (size_gb / (avg_s)) 

    print(f"{size:<12}{n_elems:<12}{dtype + f'-{op}' if op else dtype:<10}"
          f"{avg_us:<18.7f}{min_us:<18.3f}{max_us:<18.3f}"
          f"{alg_bandwidth:<15.3f}{bus_bandwidth:<15.3f}")


def print_runtime_options(args):
    print("Runtime options after parsing command line arguments")
    print(
        "min_size: {}, max_size: {}, step_factor: {}, iterations: {}, warmup iterations: {}, "
        "number of ctas: {}, threads per cta: {}, stride: {}, datatype: {}, reduce_op: {}, "
        "threadgroup_scope: {}, atomic_op: {}, dir: {}, report_msgrate: {}, bidirectional: {}, "
        "putget_issue: {}, use_graph: {}".format(
            args.min_size or 0,
            args.max_size or 0,
            args.step or 0,
            args.iters or 0,
            args.warmup_iters or 0,
            args.ctas or 0,
            args.threads_per_cta or 0,
            args.stride or 0,
            args.datatype or "None",
            args.reduce_op or "None",
            args.scope or "None",
            args.atomic_op or "None",
            args.dir or "None",
            int(args.msgrate),
            int(args.bidir),
            args.issue or "None",
            int(args.cudagraph)
        )
    )
    print("Note: Above is full list of options, any given test will use only a subset of these variables.\n")
    print("Note: Python perftests do not yet reach full parity with C library perftests. Some options may be ignored.\n")



def build_parser():
    parser = argparse.ArgumentParser(description="Run NCCL-like device point-to-point tests")

    parser.add_argument("-b", "--min_size", type=str, default="8", help="Minimum message size in bytes")
    parser.add_argument("-e", "--max_size", type=str, default="4194304", help="Maximum message size in bytes")
    parser.add_argument("-f", "--step", type=float, default=2, help="Step factor for message sizes")
    parser.add_argument("-n", "--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument("-w", "--warmup_iters", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("-c", "--ctas", type=int, help="Number of CTAs to launch (used in some device pt-to-pt tests)")
    parser.add_argument("-t", "--threads_per_cta", type=int, help="Number of threads per block (used in some device pt-to-pt tests)")

    parser.add_argument(
        "-d", "--datatype",
        choices=[
            "int", "int32", "uint32", "int64", "uint64", "long", "longlong",
            "ulonglong", "size", "ptrdiff", "float", "double", "fp16", "bf16"
        ],
        default="float",
        help="Datatype to use"
    )

    parser.add_argument(
        "-o", "--reduce_op",
        choices=["min", "max", "sum"],
        default="sum",
        help="Reduction operation"
    )

    parser.add_argument(
        "-s", "--scope",
        choices=["thread", "warp", "block", "all"],
        help="Scope of the operation"
    )

    parser.add_argument("-i", "--stride", type=int, help="Stride between elements")

    parser.add_argument(
        "-a", "--atomic_op",
        choices=[
            "inc", "add", "and", "or", "xor", "set", "swap",
            "fetch_inc", "fetch_add", "fetch_and", "fetch_or", "fetch_xor",
            "compare_swap"
        ],
        help="Atomic operation"
    )

    parser.add_argument("--bidir", action="store_true", help="Run bidirectional test")
    parser.add_argument("--msgrate", action="store_true", help="Report message rate (MMPs)")

    parser.add_argument(
        "--dir",
        choices=["read", "write"],
        help="Whether to run put or get operations"
    )

    parser.add_argument(
        "--issue",
        choices=["on_stream", "host"],
        help="Applicable in some host pt-to-pt tests"
    )

    parser.add_argument("--cudagraph", action="store_true", help="Use CUDA graph to amortize launch overhead")
    args = parser.parse_args()

    # Convert string sizes to int sizes here
    args.min_size = parse_size(args.min_size)
    args.max_size =parse_size(args.max_size)
    return args

