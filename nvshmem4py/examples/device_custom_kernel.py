from numba import cuda
import nvshmem.core
from nvshmem.bindings.device.numba import my_pe, n_pes, put_signal_nbi, signal_wait_until
from nvshmem.core import SignalOp, ComparisonType
import cuda.core
from cuda.core.experimental import Device, system
from mpi4py import MPI

signal_op = SignalOp.ADD
comparison_type = ComparisonType.GE

SIGNAL_ADD = signal_op.value
CMP_GE = comparison_type.value

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
    signal_block = signal[block_idx:block_idx+1]

    chunk_elems = chunk_size 
    num_chunks = elems_per_block // chunk_elems

    # Reduce phase
    block_base_offset = block_idx * elems_per_block
    for chunk, offset in enumerate(range(block_base_offset, block_base_offset + elems_per_block, chunk_elems)):
        src_block = src[offset:offset+chunk_elems]
        dst_block = dst[offset:offset+chunk_elems]

        if mype != 0:
            if thread_id == 0:
                signal_wait_until(signal_block, CMP_GE, chunk + 1)

            cuda.syncthreads()
            for i in range(thread_id, chunk_elems, num_threads):
                dst_block[i] = dst_block[i] + src_block[i]
            cuda.syncthreads()

        if thread_id == 0:
            src_data = src_block if mype == 0 else dst_block
            put_signal_nbi(dst_block, src_data, chunk_elems, 
                              signal_block, 1, SIGNAL_ADD, peer)

    # Broadcast phase
    if thread_id == 0:
        for chunk, offset in enumerate(range(block_base_offset, block_base_offset + elems_per_block, chunk_elems)):
            dst_block = dst[offset:offset+chunk_elems]

            if mype < npes - 1:  # Last pe already has the final result
                expected_val = (chunk + 1) if mype == 0 else (num_chunks + chunk + 1)
                signal_wait_until(signal_block, CMP_GE, expected_val)
            
            if mype < npes - 2:
                put_signal_nbi(dst_block, dst_block, chunk_elems,
                                  signal_block, 1, SIGNAL_ADD, peer)

        # Reset signal for next iteration
        signal_block[0] = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    size = args.size
    chunk_size = args.chunk_size

    rank = MPI.COMM_WORLD.Get_rank()
    dev = cuda.Device(rank % system.num_devices)
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    src = nvshmem.core.array((size,), dtype="float32")
    dst = nvshmem.core.array((size,), dtype="float32")
    src[:] = 1
    dst[:] = 0
    signal = nvshmem.core.array((1,), dtype="uint64")

    ring_reduce[1, 1](dst, src, size, signal, chunk_size)

    print(dst)