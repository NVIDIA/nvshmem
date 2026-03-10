# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information
"""
This file implements a program representative of an MoE (mixture of experts) application using only Numba-CUDA and NVSHMEM4Py.
"""
import numpy as np
import cupy as cp
from numba import cuda, int32, float32
import nvshmem.core
import nvshmem.core.device.numba as shmem_dev
from nvshmem.core import SignalOp, ComparisonType
from nvshmem.core.device.numba import my_pe, n_pes
from cuda.core import Device, system
from mpi4py import MPI

# Constants for the MoE pattern
# Simple baseline used by the original example
N_EXPERTS = 4
N_INPUTS = 16
N_FEATURES = 8
BATCH_SIZE = N_INPUTS // N_EXPERTS

# MoE shuffle-style parameters (mirroring moe_shuffle.cu)
# Many experts, top-k routing per token
TOP_K = 2
NUM_SRC_ROWS = 32
NUM_ROWS = NUM_SRC_ROWS * TOP_K

signal_op = SignalOp.SIGNAL_ADD
comparison_type = ComparisonType.CMP_GE


@cuda.jit
def dispatch_inputs(inputs, expert_inputs, expert_signals, batch_size, nfeatures):
    """
    Each PE dispatches its batch to the corresponding expert (PE).
    """
    mype = my_pe()
    npes = n_pes()
    tid = cuda.threadIdx.x

    # Each PE is an expert, receives a batch from all PEs
    for src_pe in range(npes):
        # Only one thread per PE does the put_signal
        if tid == 0:
            # Calculate offset for this expert in the global input
            src_offset = src_pe * batch_size * nfeatures
            dst_offset = mype * batch_size * nfeatures
            # Put the batch from src_pe to this expert's input buffer and signal
            shmem_dev.put_signal_nbi(
                expert_inputs[dst_offset:dst_offset + batch_size * nfeatures],
                inputs[src_offset:src_offset + batch_size * nfeatures],
                expert_signals[mype:mype + 1],
                np.uint64(1),
                signal_op,
                mype,
            )


@cuda.jit
def expert_kernel(expert_inputs, expert_outputs, expert_signals, batch_size, nfeatures):
    """
    Each expert waits for all batches, processes its batch, and writes output.
    """
    mype = my_pe()
    tid = cuda.threadIdx.x

    # Wait until all batches have arrived (one from each PE)
    if tid == 0:
        shmem_dev.signal_wait(expert_signals[mype:mype + 1], comparison_type, np.uint64(n_pes()))
    cuda.syncthreads()

    # Simple expert computation: sum along features
    for i in range(tid, batch_size, cuda.blockDim.x):
        acc = 0.0
        for j in range(nfeatures):
            acc += expert_inputs[mype * batch_size * nfeatures + i * nfeatures + j]
        expert_outputs[mype * batch_size + i] = acc


@cuda.jit
def combine_outputs(expert_outputs, combined_outputs, root_signals, batch_size):
    """
    Gather expert outputs from all PEs to the root PE (PE 0).
    Each expert sends its contiguous batch slice to the root and signals completion.
    The root waits for all signals and assembles the final combined output.
    """
    mype = my_pe()
    npes = n_pes()
    tid = cuda.threadIdx.x

    root = 0

    # Each PE contributes its [mype * batch_size : (mype + 1) * batch_size] slice
    start = mype * batch_size
    end = start + batch_size

    if mype == root:
        # Root copies its own local slice directly
        for i in range(tid, batch_size, cuda.blockDim.x):
            combined_outputs[start + i] = expert_outputs[start + i]

        # One thread waits for the remaining experts' signals
        if tid == 0:
            for pe in range(npes):
                if pe == root:
                    continue
                shmem_dev.signal_wait(root_signals[pe:pe + 1], comparison_type, 1)
    else:
        # Non-root sends its slice to root and signals
        if tid == 0:
            shmem_dev.put_signal_nbi(
                combined_outputs[start:end],
                expert_outputs[start:end],
                root_signals[mype:mype + 1],
                1,
                signal_op,
                root,
            )


@cuda.jit
def exchange_offsets(expert_counts, expert_pos_out, num_experts):
    """
    Device-side prefix-sum style exchange of expert counts across PEs.
    Fills expert_pos_out[r + e * npes] = cumulative count up to and including PE r for expert e.
    """
    tid = cuda.threadIdx.x
    if tid == 0:
        npes = n_pes()
        # For each expert, compute prefix over PEs by reading per-PE counts via immediate get
        for e in range(num_experts):
            prev = 0
            for r in range(npes):
                val = shmem_dev.g(expert_counts[e:e + 1], r)
                prev += val
                expert_pos_out[r + e * npes] = prev
    cuda.syncthreads()


@cuda.jit
def token_shuffle_two_step_allpush(send_data, recv_data, expanded_src_row, expert_for_expanded_src_row, expert_offsets,
                                   k, num_rows, nfeatures, expert_pos_out, num_experts):
    """
    Shuffle tokens to experts across ranks using CTA-level puts.
    Mirrors moe_shuffle.cu two-step allpush at a high level using arrays.
    """
    mype = my_pe()
    npes = n_pes()
    num_experts_per_rank = num_experts // npes

    # Block-strided loop over rows
    block_offset = cuda.blockIdx.x
    num_blocks = cuda.gridDim.x
    rows_per_expert = num_rows // num_experts if num_experts > 0 else 0
    true_block_offset = rows_per_expert * mype * num_experts_per_rank + block_offset
    rounded_block_offset = true_block_offset % num_rows if num_rows > 0 else 0

    while block_offset < num_rows:
        if num_rows == 0:
            return

        # Select source row and destination expert
        src_row = expanded_src_row[rounded_block_offset] % (num_rows // k)
        expert = expert_for_expanded_src_row[rounded_block_offset]
        peer = expert // num_experts_per_rank
        first_expert_in_rank = (expert % num_experts_per_rank) == 0

        # expert starting position for this rank on the destination rank
        expert_start = 0 if (mype == 0 and first_expert_in_rank) else expert_pos_out[mype + expert * npes - 1]
        # relative position inside expert bucket
        pos_in_expert = rounded_block_offset - (expert_offsets[expert - 1] if expert > 0 else 0)

        # Compute contiguous slices (flattened)
        dst_row = expert_start + pos_in_expert
        # CTA-level non-blocking put of one row
        shmem_dev.put_nbi_block(
            recv_data[dst_row * nfeatures:(dst_row + 1) * nfeatures],
            send_data[src_row * nfeatures:(src_row + 1) * nfeatures],
            peer,
        )

        true_block_offset += num_blocks
        rounded_block_offset = true_block_offset % num_rows
        block_offset += num_blocks


@cuda.jit
def build_routing(expanded_src_row, expert_for_expanded_src_row, num_src_rows, top_k, n_experts):
    idx = cuda.grid(1)
    total = num_src_rows * top_k
    if idx >= total:
        return
    i = idx // top_k
    # round-robin expert selection
    expert = idx % n_experts
    expanded_src_row[idx] = i
    expert_for_expanded_src_row[idx] = expert


@cuda.jit
def count_experts(expert_for_expanded_src_row, expert_counts, n_rows):
    idx = cuda.grid(1)
    if idx < n_rows:
        exp = expert_for_expanded_src_row[idx]
        cuda.atomic.add(expert_counts, exp, 1)


@cuda.jit
def prefix_expert_offsets(expert_counts, expert_offsets, n_experts):
    if cuda.threadIdx.x == 0:
        total = 0
        for e in range(n_experts):
            total += expert_counts[e]
            expert_offsets[e] = total


def main():
    # Set device based on local rank
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    mype = nvshmem.core.my_pe()
    npes = nvshmem.core.n_pes()

    # Allocate input and output arrays
    inputs = nvshmem.core.array((N_INPUTS, N_FEATURES), dtype="float32")
    expert_inputs = nvshmem.core.array((N_EXPERTS * BATCH_SIZE, N_FEATURES), dtype="float32")
    expert_outputs = nvshmem.core.array((N_EXPERTS * BATCH_SIZE, ), dtype="float32")
    expert_signals = nvshmem.core.array((N_EXPERTS, ), dtype="uint64")
    combined_outputs = nvshmem.core.array((N_EXPERTS * BATCH_SIZE, ), dtype="float32")
    root_signals = nvshmem.core.array((nvshmem.core.n_pes(), ), dtype="uint64")

    # Buffers for moe_shuffle-like path
    # Source rows and expert mapping for top-k routing
    send_data = nvshmem.core.array((NUM_SRC_ROWS, N_FEATURES), dtype="float32")
    recv_data = nvshmem.core.array((NUM_ROWS, N_FEATURES), dtype="float32")
    expanded_src_row = nvshmem.core.array((NUM_ROWS, ), dtype="int32")
    expert_for_expanded_src_row = nvshmem.core.array((NUM_ROWS, ), dtype="int32")
    expert_counts = nvshmem.core.array((N_EXPERTS, ), dtype="int64")
    expert_offsets = nvshmem.core.array((N_EXPERTS, ), dtype="int64")
    expert_pos_out = nvshmem.core.array((nvshmem.core.n_pes() * N_EXPERTS, ), dtype="int64")

    # Initialize input data (each PE initializes its own batch)
    if mype == 0:
        data = cp.asarray(np.arange(N_INPUTS * N_FEATURES, dtype=np.float32).reshape(N_INPUTS, N_FEATURES))
        inputs[:] = data
        # Initialize send_data for shuffle path and create simple round-robin routing
        send = cp.asarray(np.arange(NUM_SRC_ROWS * N_FEATURES, dtype=np.float32).reshape(NUM_SRC_ROWS, N_FEATURES))
        send_data[:] = send
    # Create a CUDA stream and use it for barriers and kernels
    stream = dev.create_stream()
    nvshmem.core.barrier_all(stream=stream)

    # Zero signals and expert buffers
    expert_signals[:] = 0
    expert_inputs[:] = 0
    expert_outputs[:] = 0
    combined_outputs[:] = 0
    root_signals[:] = 0
    recv_data[:] = 0
    expert_pos_out[:] = 0
    expanded_src_row[:] = 0
    expert_for_expanded_src_row[:] = 0
    expert_counts[:] = 0
    expert_offsets[:] = 0

    # Launch dispatch kernel: each PE sends its batch to all experts
    threads_per_block = 32
    dispatch_inputs[1, threads_per_block, stream](inputs.reshape(-1), expert_inputs.reshape(-1), expert_signals,
                                                  BATCH_SIZE, N_FEATURES)
    nvshmem.core.barrier_all(stream=stream)

    # Launch expert kernel: each expert processes its batch
    expert_kernel[1, threads_per_block, stream](expert_inputs.reshape(-1), expert_outputs, expert_signals, BATCH_SIZE,
                                                N_FEATURES)
    nvshmem.core.barrier_all(stream=stream)

    # Build routing and expert counts/offsets on device
    threads = 128
    rows_total = NUM_ROWS
    grid_rows = (rows_total + threads - 1) // threads
    build_routing[grid_rows, threads, stream](
        expanded_src_row,
        expert_for_expanded_src_row,
        NUM_SRC_ROWS,
        TOP_K,
        N_EXPERTS,
    )
    nvshmem.core.barrier_all(stream=stream)
    count_experts[grid_rows, threads, stream](
        expert_for_expanded_src_row,
        expert_counts,
        NUM_ROWS,
    )
    nvshmem.core.barrier_all(stream=stream)
    prefix_expert_offsets[1, 1, stream](
        expert_counts,
        expert_offsets,
        N_EXPERTS,
    )
    nvshmem.core.barrier_all(stream=stream)

    # Two-step offset exchange for shuffle-style path
    exchange_offsets[1, 1, stream](expert_counts, expert_pos_out, N_EXPERTS)
    nvshmem.core.barrier_all(stream=stream)

    # Token shuffle (by-peer style loop with block puts)
    gridsize = max(1, min(NUM_ROWS, 32))
    threads = 128
    token_shuffle_two_step_allpush[gridsize, threads, stream](
        send_data.reshape(-1),
        recv_data.reshape(-1),
        expanded_src_row,
        expert_for_expanded_src_row,
        expert_offsets,
        TOP_K,
        NUM_ROWS,
        N_FEATURES,
        expert_pos_out,
        N_EXPERTS,
    )
    nvshmem.core.barrier_all(stream=stream)

    # Launch combine kernel: gather expert outputs to root
    combine_outputs[1, threads_per_block, stream](
        expert_outputs,
        combined_outputs,
        root_signals,
        BATCH_SIZE,
    )
    nvshmem.core.barrier_all(stream=stream)
    dev.sync()

    # Print combined results (from PE 0)
    if mype == 0:
        print("Combined outputs:")
        print(combined_outputs)

    nvshmem.core.free_array(inputs)
    nvshmem.core.free_array(expert_inputs)
    nvshmem.core.free_array(expert_outputs)
    nvshmem.core.free_array(expert_signals)
    nvshmem.core.free_array(combined_outputs)
    nvshmem.core.free_array(root_signals)
    nvshmem.core.free_array(send_data)
    nvshmem.core.free_array(recv_data)
    nvshmem.core.free_array(expanded_src_row)
    nvshmem.core.free_array(expert_for_expanded_src_row)
    nvshmem.core.free_array(expert_counts)
    nvshmem.core.free_array(expert_offsets)
    nvshmem.core.free_array(expert_pos_out)
    nvshmem.core.finalize()


if __name__ == "__main__":
    main()
