import cffi
from mpi4py import MPI
from cuda.core import Device, system

from numba import cuda, uint64

import nvshmem
import nvshmem.bindings
from nvshmem.core import SignalOp, ComparisonType
from nvshmem.bindings.device.numba import int_put_signal_nbi, signal_wait_until, my_pe, n_pes

ffi = cffi.FFI()


@cuda.jit(lto=True)
def ring_reduce(dst, src, nreduce, signal, chunk_size):
    mype = my_pe()
    npes = n_pes()
    peer = (mype + 1) % npes

    thread_id = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    num_blocks = cuda.gridDim.x
    block_idx = cuda.blockIdx.x
    elems_per_block = nreduce // num_blocks

    # Change src, dst, nreduce, signal to what this block is going to process
    # Each CTA will work independently
    if elems_per_block * (block_idx + 1) > nreduce:
        return

    # Adjust pointers for this block
    src_offset = block_idx * elems_per_block
    dst_offset = block_idx * elems_per_block

    # Use ffi.from_buffer to get array access
    src_block = ffi.from_buffer(src[src_offset:])
    dst_block = ffi.from_buffer(dst[dst_offset:])
    signal_block = ffi.from_buffer(signal[block_idx:block_idx + 1])

    chunk_elems = chunk_size
    num_chunks = elems_per_block // chunk_elems

    # Reduce phase
    for chunk in range(num_chunks):
        if mype != 0:
            if thread_id == 0:
                signal_wait_until(signal_block, ComparisonType.CMP_GE, chunk + 1)

            cuda.syncthreads()
            for i in range(thread_id, chunk_elems, num_threads):
                dst_block[i] = dst_block[i] + src_block[i]
            cuda.syncthreads()

        if thread_id == 0:
            src_data = src_block if mype == 0 else dst_block
            int_put_signal_nbi(dst_block, src_data, chunk_elems, signal_block, uint64(1), SignalOp.SIGNAL_ADD, peer)

        # Move to next chunk
        src_offset += chunk_elems
        dst_offset += chunk_elems

        src_block = ffi.from_buffer(src[src_offset:])
        dst_block = ffi.from_buffer(dst[dst_offset:])

    # if signal is printed here, it will be 0 for first and last PE, num_chunks for other PEs.

    # Reset dst pointer for broadcast phase
    dst_offset = block_idx * elems_per_block
    dst_block = ffi.from_buffer(dst[dst_offset:])

    # Broadcast phase
    if thread_id == 0:
        for chunk in range(num_chunks):
            if mype < npes - 1:  # Last pe already has the final result
                expected_val = (chunk + 1) if mype == 0 else (num_chunks + chunk + 1)
                signal_wait_until(signal_block, ComparisonType.CMP_GE, expected_val)

            if mype < npes - 2:
                int_put_signal_nbi(dst_block, dst_block, chunk_elems, signal_block, uint64(1), SignalOp.SIGNAL_ADD,
                                   peer)

            dst_offset += chunk_elems
            dst_block = ffi.from_buffer(dst[dst_offset:])

        # Reset signal for next iteration
        signal_block[0] = 0


# Initialize MPI and NVSHMEM
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
dev = Device(local_rank_per_node)
dev.set_current()

stream = dev.create_stream()

nvshmem.core.init(
    device=dev,
    uid=None,
    rank=None,
    nranks=None,
    mpi_comm=MPI.COMM_WORLD,
    initializer_method="mpi",
)

mype = nvshmem.bindings.my_pe()
npes = nvshmem.bindings.n_pes()

# Test parameters
nreduce = 1024

num_blocks = 32
elems_per_block = nreduce // num_blocks
num_chunk_per_block = 4
chunk_size = elems_per_block // num_chunk_per_block

threads_per_block = 512

# Allocate arrays
src = nvshmem.core.array((nreduce, ), dtype="int32")
dst = nvshmem.core.array((nreduce, ), dtype="int32")
signal = nvshmem.core.array((num_blocks, ), dtype="uint64")

# Initialize data
for i in range(nreduce):
    src[i] = mype + 1

dst[:] = 0

# Initialize signal
for i in range(num_blocks):
    signal[i] = 0

# Launch kernel
ring_reduce[num_blocks, threads_per_block, stream, 0](dst, src, nreduce, signal, chunk_size)

nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
dev.sync()

# Check results
expected_result = sum(range(1, npes + 1))
for i in range(nreduce):
    assert dst[i] == expected_result, f"PE {mype}: Mismatch at index {i}: got {dst[i]}, expected {expected_result}"
print(f"PE {mype}: Ring allreduce test passed")

# Clean up
nvshmem.core.free_array(src)
nvshmem.core.free_array(dst)
nvshmem.core.free_array(signal)
nvshmem.core.finalize()
